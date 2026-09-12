import type { CompressedObservation, Memory, Session } from "../types.js";
import { KV } from "../state/schema.js";
import type { StateKV } from "../state/kv.js";
import { SearchIndex } from "../state/search-index.js";
import { VectorIndex } from "../state/vector-index.js";
import type { IndexPersistence } from "../state/index-persistence.js";
import {
  memoryToIndexDoc,
  memoryChunkJobs,
  isLatestEligible,
  MEMORY_SESSION,
} from "../state/memory-utils.js";
import {
  getSearchIndex,
  getVectorIndex,
  getEmbeddingProvider,
  clipEmbedInput,
  isIndexExcluded,
} from "./search.js";
import { logger } from "../logger.js";

// A2 - baseline reconcile.
//
// rebuildIndex() cannot be used for this. It clears the LIVE singletons
// before repopulating them (search.ts:313), soft-fails its embedding work
// item by item, and hands the result to a persistence path that publishes
// BM25 and vector separately. Any one of those turns a failed rebuild into
// a degraded live index. This builds into throwaway objects instead, fails
// closed on the first embedding loss, and publishes both indexes as one
// retained-predecessor checkpoint. The live singletons are swapped only
// after the checkpoint is on disk.
//
// Deliberately NOT incremental. Reusing embeddings for ids already in the
// vector index would carry forward exactly the drift this is meant to
// clear: no writer re-indexes on content change, so an id being present
// says nothing about the vector matching the row's current text.

export type ReconcileReport = {
  published: boolean;
  generation: string | null;
  durationMs: number;
  memories: { scanned: number; indexed: number; skippedNonLatest: number };
  observations: { scanned: number; indexed: number; excluded: number };
  embeddings: { attempted: number; batches: number; failed: number };
  bm25: { before: number; after: number; added: number; dropped: number };
  // Ids visited more than once by the corpus walk. Nonzero means a row is
  // reachable from two places (e.g. an observation present in two session
  // scopes); the index is unharmed because add() replaces, but the delta
  // arithmetic is only correct once each id is counted once.
  duplicateIds: number;
  vectors: { before: number; after: number };
  failedSessions: string[];
};

const DEFAULT_BATCH = 32;
// One progress line per this many embed batches. A reconcile of this corpus
// runs for tens of minutes with no output otherwise, which makes it
// impossible to tell a working run from a wedged one - the operator's only
// signal would be CPU on the embedding service.
const PROGRESS_EVERY_BATCHES = 25;

// Ops already tunes the rebuild path with REBUILD_EMBED_BATCH_SIZE (set to 8
// in the drop-in for embed reliability). The reconcile drives the same
// endpoint, so it honours the same knob rather than inventing a second one -
// and it matters more here, because a batch that times out fails the whole
// run closed instead of being skipped.
function envBatchSize(): number {
  const raw = process.env.REBUILD_EMBED_BATCH_SIZE;
  if (!raw) return DEFAULT_BATCH;
  const n = parseInt(raw, 10);
  return Number.isFinite(n) && n > 0 ? n : DEFAULT_BATCH;
}

// Invocation is a boot token, not an HTTP route, and deliberately so.
// A full reconcile embeds the whole corpus; served synchronously it would
// hit the same iii invocation timeout that already makes GET /export
// return 500 on this corpus while the work completes underneath. The
// rebuild token (search.ts:91) is the pattern this mirrors - idempotent,
// safe to leave in a drop-in, consumed once it succeeds.
//
// The token is consumed only by a run that PUBLISHES, so a dry run can be
// repeated and the publishing run is the one that spends it.
const RECONCILE_TOKEN_KEY = "index-reconcile-token";

export async function pendingReconcileToken(
  kv: StateKV,
): Promise<string | null> {
  const want = process.env.AGENTMEMORY_RECONCILE?.trim();
  if (!want) return null;
  const stored = await kv
    .get<{ token: string }>(KV.state, RECONCILE_TOKEN_KEY)
    .catch(() => null);
  return stored?.token === want ? null : want;
}

export async function markReconcileTokenDone(
  kv: StateKV,
  token: string,
): Promise<void> {
  await kv
    .set(KV.state, RECONCILE_TOKEN_KEY, {
      token,
      at: new Date().toISOString(),
    })
    .catch(() => undefined);
}

// Publishing is opt-in on top of the token. A token alone dry-runs: it
// reads and embeds the entire corpus and reports the delta without
// touching a manifest, which is what sizes the maintenance window.
export function isReconcilePublishEnabled(): boolean {
  return process.env.AGENTMEMORY_RECONCILE_PUBLISH?.trim() === "true";
}

type EmbedJob = {
  id: string;
  sessionId: string;
  text: string;
};

// Fail-closed counterpart to vectorIndexAddBatchGuarded. That helper logs
// and counts failures so a live save is never blocked by a flaky embed;
// here the same tolerance would let the reconcile publish a baseline that
// is quietly missing vectors and still call it verified. Any loss throws.
async function embedBatchStrict(
  jobs: EmbedJob[],
  target: VectorIndex,
): Promise<void> {
  if (jobs.length === 0) return;
  const ep = getEmbeddingProvider();
  if (!ep) throw new Error("reconcile: no embedding provider configured");

  const embeddings = await ep.embedBatch(
    jobs.map((j) => clipEmbedInput(j.text)),
  );
  if (embeddings.length !== jobs.length) {
    throw new Error(
      `reconcile: provider ${ep.name} returned ${embeddings.length} embeddings for ${jobs.length} inputs`,
    );
  }
  for (let i = 0; i < jobs.length; i++) {
    const embedding = embeddings[i];
    if (embedding.length !== ep.dimensions) {
      throw new Error(
        `reconcile: ${jobs[i].id} embedded at ${embedding.length} dimensions, provider declares ${ep.dimensions}`,
      );
    }
    target.add(jobs[i].id, jobs[i].sessionId, embedding);
  }
}

export async function reconcileIndexes(
  kv: StateKV,
  persistence: IndexPersistence,
  opts: { publish: boolean; batchSize?: number } = { publish: false },
): Promise<ReconcileReport> {
  const startedAt = Date.now();
  const batchSize =
    opts.batchSize && opts.batchSize > 0 ? opts.batchSize : envBatchSize();

  const liveBm25 = getSearchIndex();
  const liveVector = getVectorIndex();
  const ep = getEmbeddingProvider();

  // A vector index exists but no provider can serve it: building BM25 alone
  // and publishing would leave the old vector generation live beside a new
  // BM25 one, which is the mixed generation this whole protocol exists to
  // avoid. Refuse rather than half-reconcile.
  if (liveVector && !ep) {
    throw new Error(
      "reconcile: vector index is active but no embedding provider is configured; refusing to publish a BM25-only generation",
    );
  }

  const bm25 = new SearchIndex();
  const vector = liveVector ? new VectorIndex() : null;

  const report: ReconcileReport = {
    published: false,
    generation: null,
    durationMs: 0,
    memories: { scanned: 0, indexed: 0, skippedNonLatest: 0 },
    observations: { scanned: 0, indexed: 0, excluded: 0 },
    embeddings: { attempted: 0, batches: 0, failed: 0 },
    bm25: { before: liveBm25.size, after: 0, added: 0, dropped: 0 },
    duplicateIds: 0,
    vectors: { before: liveVector?.size ?? 0, after: 0 },
    failedSessions: [],
  };

  // Ids present in the fresh build that the live index also has. Everything
  // else is a delta in one direction or the other, and the live index has no
  // id enumeration, so the overlap count is what makes both directions
  // computable: dropped = live.size - overlap.
  let overlap = 0;
  // SearchIndex.add() REPLACES on a duplicate id, so bm25.size counts each id
  // once while the walk may visit it more than once. Counting overlap per
  // visit therefore let it exceed the live index size and drove `dropped`
  // negative (-71 on the first dry run, which is how the duplicates were
  // noticed at all). Count each id once, and report the duplicates rather
  // than hiding them: 71 of them means something is walked twice.
  const freshIds = new Set<string>();
  const duplicateSample: string[] = [];
  const pending: EmbedJob[] = [];
  const flush = async (): Promise<void> => {
    if (pending.length === 0 || !vector) {
      pending.length = 0;
      return;
    }
    report.embeddings.batches++;
    report.embeddings.attempted += pending.length;
    try {
      await embedBatchStrict(pending, vector);
    } catch (err) {
      report.embeddings.failed += pending.length;
      throw err;
    }
    pending.length = 0;
    if (report.embeddings.batches % PROGRESS_EVERY_BATCHES === 0) {
      const elapsedS = Math.round((Date.now() - startedAt) / 1000);
      const rate = elapsedS > 0 ? report.embeddings.attempted / elapsedS : 0;
      logger.info("reconcile: progress", {
        embedded: report.embeddings.attempted,
        batches: report.embeddings.batches,
        bm25Docs: bm25.size,
        elapsedS,
        perSec: Math.round(rate * 10) / 10,
      });
    }
  };
  const enqueue = async (job: EmbedJob): Promise<void> => {
    if (!vector) return;
    pending.push(job);
    if (pending.length >= batchSize) await flush();
  };
  const noteIndexed = (id: string): void => {
    if (freshIds.has(id)) {
      report.duplicateIds++;
      if (duplicateSample.length < 10) duplicateSample.push(id);
      return;
    }
    freshIds.add(id);
    if (liveBm25.has(id)) overlap++;
  };

  // Memories first, same predicate as rebuildIndex and A1 so all three
  // agree on what "latest" means. Any divergence here would publish an
  // index the read path disagrees with.
  const memories = await kv.list<Memory>(KV.memories);
  for (const memory of memories) {
    report.memories.scanned++;
    if (!isLatestEligible(memory)) {
      report.memories.skippedNonLatest++;
      continue;
    }
    if (!memory.title || !memory.content) continue;
    bm25.add(memoryToIndexDoc(memory));
    noteIndexed(memory.id);
    report.memories.indexed++;
    for (const job of memoryChunkJobs(memory)) {
      await enqueue({
        id: job.id,
        sessionId: memory.sessionIds?.[0] ?? MEMORY_SESSION,
        text: job.text,
      });
    }
  }

  const sessions = await kv.list<Session>(KV.sessions);
  for (let batch = 0; batch < sessions.length; batch += 10) {
    const chunk = sessions.slice(batch, batch + 10);
    const results = await Promise.all(
      chunk.map(async (s) => {
        try {
          return await kv.list<CompressedObservation>(KV.observations(s.id));
        } catch {
          report.failedSessions.push(s.id);
          return [] as CompressedObservation[];
        }
      }),
    );
    for (const observations of results) {
      for (const obs of observations) {
        report.observations.scanned++;
        if (!obs.title || !obs.narrative) continue;
        if (isIndexExcluded(obs)) {
          report.observations.excluded++;
          continue;
        }
        bm25.add(obs);
        noteIndexed(obs.id);
        report.observations.indexed++;
        await enqueue({
          id: obs.id,
          sessionId: obs.sessionId,
          text: obs.title + " " + obs.narrative,
        });
      }
    }
  }
  await flush();

  // A session whose observations could not be read would silently shrink
  // the baseline. Fail closed for the same reason the embed path does.
  if (report.failedSessions.length > 0) {
    throw new Error(
      `reconcile: ${report.failedSessions.length} session(s) failed to load; refusing to publish a short index (${report.failedSessions.slice(0, 5).join(", ")})`,
    );
  }

  report.bm25.after = bm25.size;
  report.vectors.after = vector?.size ?? 0;
  report.bm25.added = bm25.size - overlap;
  report.bm25.dropped = report.bm25.before - overlap;
  report.durationMs = Date.now() - startedAt;

  // Catch-up pass. The build takes tens of minutes on this corpus, and the
  // daemon keeps serving throughout: remember.ts indexes live, auto-forget
  // and consolidation run on timers. Those rows land in the LIVE index, not
  // the fresh one - and restoreFrom() below replaces the live index
  // wholesale, so without this they would be silently dropped from search
  // while remaining in KV. Exactly the KV/index divergence this step exists
  // to remove, re-introduced by the step itself.
  //
  // Scope is bounded rather than a second full scan: every memory (one list,
  // and has() makes it a cheap skip for the ones already built) plus the
  // observations of sessions that were active or touched during the window.
  // A completed session gains no new observations.
  //
  // Residual window: writes landing between this pass and the swap, on the
  // order of seconds against a run of tens of minutes. Closing it entirely
  // needs the writer-side lock protocol, which is 4B.
  if (opts.publish) {
    const cutoff = new Date(startedAt).toISOString();
    let caught = 0;
    for (const memory of await kv.list<Memory>(KV.memories)) {
      if (bm25.has(memory.id)) continue;
      if (!isLatestEligible(memory) || !memory.title || !memory.content) continue;
      bm25.add(memoryToIndexDoc(memory));
      noteIndexed(memory.id);
      caught++;
      for (const job of memoryChunkJobs(memory)) {
        await enqueue({
          id: job.id,
          sessionId: memory.sessionIds?.[0] ?? MEMORY_SESSION,
          text: job.text,
        });
      }
    }
    const sessions2 = await kv.list<Session>(KV.sessions);
    const touched = sessions2.filter(
      (s) =>
        s.status === "active" ||
        s.startedAt > cutoff ||
        (s.endedAt !== undefined && s.endedAt > cutoff),
    );
    for (const s of touched) {
      const observations = await kv
        .list<CompressedObservation>(KV.observations(s.id))
        .catch(() => [] as CompressedObservation[]);
      for (const obs of observations) {
        if (bm25.has(obs.id)) continue;
        if (!obs.title || !obs.narrative) continue;
        if (isIndexExcluded(obs)) continue;
        bm25.add(obs);
        noteIndexed(obs.id);
        caught++;
        await enqueue({
          id: obs.id,
          sessionId: obs.sessionId,
          text: obs.title + " " + obs.narrative,
        });
      }
    }
    await flush();
    report.bm25.after = bm25.size;
    report.vectors.after = vector?.size ?? 0;
    report.bm25.added = bm25.size - overlap;
    report.bm25.dropped = report.bm25.before - overlap;
    logger.info("reconcile: catch-up pass complete", {
      caught,
      sessionsRescanned: touched.length,
      bm25Docs: bm25.size,
    });
  }

  if (!opts.publish) {
    if (report.duplicateIds > 0) {
      logger.warn("reconcile: corpus walk visited some ids more than once", {
        duplicateIds: report.duplicateIds,
        sample: duplicateSample,
      });
    }
    logger.info("reconcile: dry run complete, nothing published", {
      bm25Before: report.bm25.before,
      bm25After: report.bm25.after,
      added: report.bm25.added,
      dropped: report.bm25.dropped,
      durationMs: report.durationMs,
    });
    return report;
  }

  const checkpoint = await persistence.saveCheckpoint(
    bm25.serialize(),
    vector ? vector.serialize() : null,
  );
  report.generation = checkpoint.generation;
  report.published = true;

  // Only now does live traffic see the new generation. Swapping before the
  // checkpoint is durable would leave a restart serving the old shards
  // while the process served the new ones.
  liveBm25.restoreFrom(bm25);
  if (liveVector && vector) liveVector.restoreFrom(vector);

  report.durationMs = Date.now() - startedAt;
  logger.info("reconcile: checkpoint published", {
    generation: checkpoint.generation,
    previousBm25Generation: checkpoint.previous.bm25?.generation ?? null,
    bm25Docs: report.bm25.after,
    vectors: report.vectors.after,
    durationMs: report.durationMs,
  });
  return report;
}
