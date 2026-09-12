// mem::index-fill-missing - vector fill and repair pass for inproc mode
// (plan step 5). Compares what the content rows call for with what the
// `vectors` table holds and closes the gap in both directions:
//
//   - a job with no row, or with a `verified` row whose `input_hash` differs
//     from the current text, is embedded and committed through the same
//     revalidating store method every live completion uses;
//   - a row whose id no current job claims is pruned, inside a transaction
//     that re-reads the content row at that moment, so an interleaved save
//     between enumeration and apply is respected.
//
// `legacy` rows (imported from the iii shards, no input text to hash) count
// as complete until their content is rewritten, which the store flips to
// `verified` with an empty hash in the rewrite's own transaction.
//
// Single-flight: a second invocation while one runs returns immediately with
// `skipped: true`.
import type { ISdk } from "iii-sdk";
import type { SqliteState } from "../engine/inproc/state.js";
import type { SqliteVectorStore } from "../engine/inproc/vectors.js";
import type { VectorIndex } from "../state/vector-index.js";
import { parentIdOf } from "../state/vector-index.js";
import { KV } from "../state/schema.js";
import {
  embedInputHash,
  enumerateIndexCorpus,
  memoryEmbedJobs,
  observationEmbedJobs,
  type EmbedJob,
} from "../state/index-corpus.js";
import type { CompressedObservation, Memory } from "../types.js";
import { getEmbeddingProvider, vectorIndexAddBatchGuarded } from "./search.js";
import { logger } from "../logger.js";

export type FillReport = {
  skipped: boolean;
  expected: number;
  present: number;
  missing: number;
  embedded: number;
  failed: number;
  pruned: number;
  aborted: boolean;
  durationMs: number;
};

const BATCH = 32;
// Whole-batch failures in a row before the pass gives up for this run. A
// provider outage would otherwise produce one warning per batch across the
// entire backlog; the next hourly run picks up where this one stopped.
const MAX_CONSECUTIVE_BATCH_FAILURES = 3;

// Whether any content row calls for the vector `vectorId` RIGHT NOW. Both
// kinds are checked: a memory and an observation may share a parent id, and a
// memory may carry a real session id, so neither lookup can exclude the
// other. `sessionId` is the vector row's current session id.
export function claimedNow(state: SqliteState, sessionId: string, vectorId: string): boolean {
  const parent = parentIdOf(vectorId);
  try {
    const jobs: EmbedJob[] = [
      ...memoryEmbedJobs(state.get(KV.memories, parent) as Memory | null),
      ...observationEmbedJobs(state.get(KV.observations(sessionId), parent) as CompressedObservation | null),
    ];
    return jobs.some((j) => j.id === vectorId);
  } catch {
    // A row we cannot evaluate is not a row we may prune.
    return true;
  }
}

export function createIndexFill(
  state: SqliteState,
  vectors: SqliteVectorStore,
  index: VectorIndex,
): { run: () => Promise<FillReport> } {
  let running: Promise<FillReport> | null = null;

  async function pass(): Promise<FillReport> {
    const t0 = performance.now();
    const report: FillReport = {
      skipped: false,
      expected: 0,
      present: 0,
      missing: 0,
      embedded: 0,
      failed: 0,
      pruned: 0,
      aborted: false,
      durationMs: 0,
    };

    const rows = new Map<string, { sessionId: string; inputHash: string; hashState: string }>();
    for (const r of state.db
      .prepare("SELECT id, session_id, input_hash, hash_state FROM vectors")
      .iterate() as Iterable<{ id: string; session_id: string; input_hash: string; hash_state: string }>) {
      rows.set(r.id, { sessionId: r.session_id, inputHash: r.input_hash, hashState: r.hash_state });
    }

    const expectedIds = new Set<string>();
    const missing: EmbedJob[] = [];
    let n = 0;
    for (const item of enumerateIndexCorpus(state.db)) {
      for (const job of item.jobs) {
        report.expected++;
        expectedIds.add(job.id);
        const row = rows.get(job.id);
        if (row && (row.hashState === "legacy" || row.inputHash === embedInputHash(job.text))) {
          report.present++;
        } else {
          missing.push(job);
        }
      }
      if (++n % 2000 === 0) await new Promise((r) => setImmediate(r));
    }
    report.missing = missing.length;

    // Prune rows nothing claims. Each delete re-reads the vector row (its
    // session id may have moved with a rewrite) and the content behind it
    // inside its own transaction: a save that raced the enumeration wins.
    // Exact removal only - a stale bare-parent row must not cascade into the
    // memory's current chunk rows.
    const liveRow = state.db.prepare("SELECT session_id FROM vectors WHERE id = ?");
    for (const id of rows.keys()) {
      if (expectedIds.has(id)) continue;
      const removed = state.transaction(() => {
        const live = liveRow.get(id) as { session_id: string } | undefined;
        if (!live || claimedNow(state, live.session_id, id)) return false;
        index.remove(id, true);
        return true;
      });
      if (removed) report.pruned++;
    }

    if (missing.length > 0 && getEmbeddingProvider()) {
      let consecutiveFailures = 0;
      for (let i = 0; i < missing.length; i += BATCH) {
        const batch = missing.slice(i, i + BATCH).map((job) => ({
          id: job.id,
          sessionId: job.sessionId,
          text: job.text,
          context: { kind: job.kind, logId: job.id },
        }));
        const { ok, fail, rejected } = await vectorIndexAddBatchGuarded(batch);
        report.embedded += ok;
        report.failed += fail + rejected;
        // Only a whole batch lost on the provider side counts towards the
        // abort; rejected completions say nothing about the provider.
        consecutiveFailures = ok === 0 && fail === batch.length ? consecutiveFailures + 1 : 0;
        if (consecutiveFailures >= MAX_CONSECUTIVE_BATCH_FAILURES) {
          report.aborted = true;
          report.failed += missing.length - i - batch.length;
          break;
        }
      }
    } else if (missing.length > 0) {
      report.failed = missing.length;
    }

    report.durationMs = Math.round(performance.now() - t0);
    // A pass that lost rows is a warning, so it still surfaces at AGENTMEMORY_LOG_LEVEL=warn.
    const level = report.failed > 0 || report.aborted ? "warn" : "info";
    logger[level]("index-fill-missing finished", { ...report, rows: vectors.count() });
    return report;
  }

  return {
    run: () => {
      if (running) {
        return Promise.resolve({
          skipped: true,
          expected: 0,
          present: 0,
          missing: 0,
          embedded: 0,
          failed: 0,
          pruned: 0,
          aborted: false,
          durationMs: 0,
        });
      }
      running = pass().finally(() => {
        running = null;
      });
      return running;
    },
  };
}

export function registerIndexFillFunction(
  sdk: ISdk,
  fill: { run: () => Promise<FillReport> },
): void {
  sdk.registerFunction("mem::index-fill-missing", async () => fill.run());
}
