// The one place that says which content rows are indexed and what text is
// embedded for them. Every consumer of that rule routes through here so they
// cannot drift apart:
//
//   - the live write paths (observe / compress / remember) build their embed
//     jobs from it;
//   - the inproc vector store re-derives the expected input for an id from the
//     CURRENT content row before it commits an embedding, so a completion for
//     text that has since been rewritten or deleted is dropped as stale;
//   - the boot BM25 rebuild and the vector fill pass enumerate the corpus with
//     the same eligibility.
import { createHash } from "node:crypto";
import type { DatabaseSync } from "node:sqlite";
import type { CompressedObservation, Memory } from "../types.js";
import { KV } from "./schema.js";
import { parentIdOf } from "./vector-index.js";
import {
  MEMORY_SESSION,
  isLatestEligible,
  memoryChunkJobs,
  memoryToIndexDoc,
} from "./memory-utils.js";

export type EmbedKind = "memory" | "observation" | "synthetic";

export type EmbedJob = {
  id: string;
  sessionId: string;
  text: string;
  kind: EmbedKind;
};

// Hard cap on embedding input length. Most providers cap input around
// 8k tokens (~32k chars at ~4 chars/token). Truncate defensively so a
// huge memory.content can't 400 the embed call or blow context budget
// on a single doc. 16k chars ≈ 4k tokens, safely under every provider.
const EMBED_MAX_CHARS = 16_000;

export function clipEmbedInput(text: string): string {
  if (text.length <= EMBED_MAX_CHARS) return text;
  return text.slice(0, EMBED_MAX_CHARS);
}

// Fingerprint of the exact text handed to the provider. Stored on the vector
// row so "a vector exists" means "a vector for THIS content exists".
export function embedInputHash(text: string): string {
  return createHash("sha256").update(clipEmbedInput(text)).digest("hex");
}

// Tool-name prefixes whose observations are STORED but never INDEXED.
//
// Retrieval echoes — the daemon's own record of "the agent ran
// memory_smart_search for <query>" — contain the query verbatim, so they
// are near-perfect BM25 and vector matches for exactly the query that
// created them, and they crowd out the answer (upstream #993). Live
// measurement: for one query, ranks 1, 2 and 6 were all records of prior
// searches, scoring up to 74.89 against the real memory's 42.23 at rank 13.
//
// The KV writes are deliberately untouched. These rows stay in the
// timeline, session history, the viewer and expandIds — they just stop
// competing in search. That is how "observations are kept" is satisfied
// structurally rather than by promise.
let excludeCache: { raw: string; prefixes: string[] } | null = null;

function excludedToolPrefixes(): string[] {
  const raw = process.env.AGENTMEMORY_INDEX_EXCLUDE_TOOL_PREFIXES ?? "";
  if (!excludeCache || excludeCache.raw !== raw) {
    excludeCache = {
      raw,
      prefixes: raw
        .split(",")
        .map((s) => s.trim())
        .filter((s) => s.length > 0),
    };
  }
  return excludeCache.prefixes;
}

export function isIndexExcluded(obs: { toolName?: string }): boolean {
  if (typeof obs.toolName !== "string" || !obs.toolName) return false;
  const prefixes = excludedToolPrefixes();
  if (prefixes.length === 0) return false;
  return prefixes.some((p) => obs.toolName!.startsWith(p));
}

// Where the content row for a vector id lives. Memories may carry a real
// session id as their first `sessionIds` entry, so the session alone does not
// identify the scope; the kind does.
export function contentLocation(
  kind: EmbedKind,
  sessionId: string,
  vectorId: string,
): { scope: string; key: string } {
  const parent = parentIdOf(vectorId);
  return kind === "memory"
    ? { scope: KV.memories, key: parent }
    : { scope: KV.observations(sessionId), key: parent };
}

// The embed jobs a stored row is entitled to right now. An empty array means
// the row is not indexed (missing, superseded, incomplete, or an excluded
// tool), so any vector for it is stale.
// Field checks are `typeof`, not truthiness: a stored row may carry a
// non-string where the type says string (the import handler accepts such
// rows), and string concatenation on an object throws. Such a row is simply
// not indexable.
export function memoryEmbedJobs(memory: Memory | null | undefined): EmbedJob[] {
  if (!memory || typeof memory.id !== "string" || !memory.id || !isLatestEligible(memory)) return [];
  if (typeof memory.title !== "string" || !memory.title) return [];
  if (typeof memory.content !== "string" || !memory.content) return [];
  const first = Array.isArray(memory.sessionIds) ? memory.sessionIds[0] : undefined;
  const sessionId = typeof first === "string" && first ? first : MEMORY_SESSION;
  return memoryChunkJobs(memory).map((job) => ({
    id: job.id,
    sessionId,
    text: job.text,
    kind: "memory" as const,
  }));
}

export function observationEmbedJobs(
  obs: CompressedObservation | null | undefined,
  kind: "observation" | "synthetic" = "observation",
): EmbedJob[] {
  if (!obs || typeof obs.id !== "string" || typeof obs.sessionId !== "string") return [];
  if (typeof obs.title !== "string" || !obs.title) return [];
  if (typeof obs.narrative !== "string" || !obs.narrative) return [];
  if (isIndexExcluded(obs)) return [];
  return [{ id: obs.id, sessionId: obs.sessionId, text: obs.title + " " + obs.narrative, kind }];
}

export function embedJobsFor(kind: EmbedKind, row: unknown): EmbedJob[] {
  return kind === "memory"
    ? memoryEmbedJobs(row as Memory | null)
    : observationEmbedJobs(row as CompressedObservation | null, kind);
}

// ---------------------------------------------------------------- corpus

// SearchIndex spreads `facts`, `concepts` and `files`; an imported or
// hand-written observation may lack them (the import handler accepts such
// rows and only skips their indexing). Normalise here so the rebuild indexes
// the row instead of throwing on it.
const arr = (v: unknown): string[] => (Array.isArray(v) ? v : []);
function indexableObservation(obs: CompressedObservation): CompressedObservation {
  return { ...obs, facts: arr(obs.facts), concepts: arr(obs.concepts), files: arr(obs.files) };
}

// `mem:obs:<sid>` scopes as a half-open range (';' follows ':'), so the
// (scope, seq) index serves it and no session id needs escaping.
const OBS_SCOPE_LO = "mem:obs:";
const OBS_SCOPE_HI = "mem:obs;";

export type CorpusItem = {
  kind: "memory" | "observation";
  scope: string;
  key: string;
  seq: number;
  row: Memory | CompressedObservation;
  // The BM25 document, or null when the row is not indexed.
  doc: CompressedObservation | null;
  // The embedding jobs the row is entitled to (empty when not indexed).
  jobs: EmbedJob[];
  // Set when building the doc or the jobs threw: the row is treated as not
  // indexed and consumers count and log it instead of failing the walk.
  error?: string;
};

// Walks every memory and every observation on disk - including observation
// scopes whose session record is gone (evict.ts removes sessions but keeps
// their observations, which are content and stay indexed) - in `seq` order,
// and applies the eligibility rules above once for both legs. The boot BM25
// rebuild and the vector fill pass consume this same iterator, so the two can
// never disagree about what belongs in the index.
export function* enumerateIndexCorpus(db: DatabaseSync): IterableIterator<CorpusItem> {
  const stmt = db.prepare(
    "SELECT seq, scope, key, value FROM kv WHERE scope = ? OR (scope >= ? AND scope < ?) ORDER BY seq",
  );
  for (const r of stmt.iterate(KV.memories, OBS_SCOPE_LO, OBS_SCOPE_HI) as Iterable<{
    seq: number;
    scope: string;
    key: string;
    value: string;
  }>) {
    let row: unknown;
    try {
      row = JSON.parse(r.value);
    } catch {
      continue;
    }
    const kind = r.scope === KV.memories ? "memory" : "observation";
    const item: CorpusItem = { kind, scope: r.scope, key: r.key, seq: r.seq, row: row as Memory, doc: null, jobs: [] };
    // One malformed row must never end the walk (the boot rebuild awaits it).
    try {
      if (kind === "memory") {
        const memory = row as Memory;
        item.jobs = memoryEmbedJobs(memory);
        item.doc = item.jobs.length > 0 ? memoryToIndexDoc(memory) : null;
      } else {
        const obs = row as CompressedObservation;
        item.jobs = observationEmbedJobs(obs);
        item.doc = item.jobs.length > 0 ? indexableObservation(obs) : null;
      }
    } catch (err) {
      item.doc = null;
      item.jobs = [];
      item.error = err instanceof Error ? err.message : String(err);
    }
    yield item;
  }
}
