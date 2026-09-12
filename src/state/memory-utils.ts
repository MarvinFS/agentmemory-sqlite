import type { CompressedObservation, Lesson, Memory } from "../types.js";
import { chunkText } from "./chunker.js";

// Synthetic sessionId every saved memory is indexed under. Memories live
// in KV.memories, outside any per-session observation scope, so there is
// no real session to attribute them to. Callers that need to recognise
// "this row is a memory, not an observation" key off this.
export const MEMORY_SESSION = "memory";

// A1: is this memory eligible to be surfaced to a caller?
//
// DO NOT "simplify" this to `Boolean(memory.isLatest)` or `memory.isLatest`.
// types.ts declares `isLatest: boolean` as REQUIRED, so a required boolean
// compared against `false` reads like a redundant check - it is not. Rows
// written before the field existed carry NO value at runtime, and the type
// declaration does not retroactively give them one. The corpus is ~2765
// memories of which only ~67 are explicitly demoted; truthiness would hide
// every legacy row, i.e. suppress the entire corpus to remove 67 rows.
//
// The predicate is therefore total by construction: exclude ONLY rows
// explicitly marked non-latest, treat absent as visible.
export function isLatestEligible(memory: Pick<Memory, "isLatest">): boolean {
  return memory.isLatest !== false;
}

// Distinct-date guard, shared by the two paths that hide one memory behind
// another on a similarity score: `remember` (supersede above 0.7 Jaccard) and
// `auto-forget` (demote above 0.9, and without recording any lineage at all).
//
// Every failure we have measured has the same shape: two RECURRING REPORTS of
// two different days, written from the same template, whose wording collides
// above the threshold. The 2026-07-12 digest ate 2026-07-11 at 0.774; the
// 07-15 run ate 07-14 at 0.908; and after the step 5 repair unwound the second
// pair, auto-forget re-demoted it within the hour, because unwinding the edge
// did nothing about the similarity. Two daily reports of two different days
// are not a paraphrase and not a contradiction - they are two facts.
//
// So: if BOTH rows carry ISO dates and they have NO date in common, they are
// about different days and neither may replace the other. Deliberately narrow.
// It does not fire when only one side is dated (a dated note genuinely can
// supersede an undated one), nor when the rows share a date (a same-day
// correction is exactly what supersession is for).
const ISO_DATE = /\b(\d{4})-(0[1-9]|1[0-2])-(0[1-9]|[12]\d|3[01])\b/g;

export function datesIn(text: string): Set<string> {
  const found = new Set<string>();
  for (const m of text.matchAll(ISO_DATE)) found.add(m[0]);
  return found;
}

export function refersToDifferentDates(a: string, b: string): boolean {
  const da = datesIn(a);
  if (da.size === 0) return false;
  const db = datesIn(b);
  if (db.size === 0) return false;
  for (const d of da) if (db.has(d)) return false;
  return true;
}

// Wraps a Memory record in the CompressedObservation shape that
// SearchIndex / VectorIndex / enrichment paths consume. Memories share
// the same searchable fields as observations (title + content +
// concepts + files); type is normalized to "decision" so memories stay
// distinguishable in result metadata without colliding with observation
// enums (file_read, command_run, …). The synthetic sessionId
// ("memory" or memory.sessionIds[0]) is what enrich-side fallbacks key
// off of when looking up the source record in KV.memories.
//
// NOTE: this output is also returned to API callers (search.ts,
// hybrid-search.ts, smart-search.ts all hand it back as the result
// payload), so its fields are part of the wire contract. Indexing-only
// adjustments belong in memoryToIndexDoc below, not here.
export function memoryToObservation(memory: Memory): CompressedObservation {
  return {
    id: memory.id,
    sessionId: memory.sessionIds?.[0] ?? MEMORY_SESSION,
    timestamp: memory.createdAt,
    type: "decision",
    title: memory.title,
    facts: [memory.content],
    narrative: memory.content,
    concepts: memory.concepts,
    files: memory.files,
    importance: memory.strength,
    // Carry the owning agent through so agent-scoped search filters see
    // memories, not just raw observations. Dropping it made every memory
    // invisible to any agentId-scoped query.
    ...(memory.agentId ? { agentId: memory.agentId } : {}),
  };
}

// Same adapter for lessons, kept beside memoryToObservation so a new
// CompressedObservation field has one obvious place to be threaded
// through both record kinds.
export function lessonToObservation(l: Lesson): CompressedObservation {
  return {
    id: l.id,
    sessionId: "lesson",
    timestamp: l.createdAt,
    type: "decision",
    title: l.content.slice(0, 120),
    facts: [l.content],
    narrative: l.context || "",
    concepts: l.tags,
    files: [],
    importance: l.confidence,
  };
}

// Indexing-only projection of a Memory.
//
// memoryToObservation puts the content in THREE fields — title (its
// first 80 chars), narrative, and facts[0] — and SearchIndex.extractTerms
// flattens all of them with no field weighting. The doc therefore looks
// ~2.2x longer than it is, and BM25's b=0.75 length normalisation
// roughly halves its score against genuinely short observations.
// Dropping the duplicated facts[0] removes one full copy; title stays
// because it is only a prefix.
export function memoryToIndexDoc(memory: Memory): CompressedObservation {
  const doc = memoryToObservation(memory);
  if (process.env.AGENTMEMORY_MEMORY_DOC_SLIM === "true") doc.facts = [];
  return doc;
}

function envNumber(key: string, fallback: number): number {
  const raw = process.env[key];
  if (!raw) return fallback;
  const n = Number(raw);
  return Number.isFinite(n) && n > 0 ? n : fallback;
}

export function isMemoryChunkingEnabled(): boolean {
  return process.env.AGENTMEMORY_MEMORY_CHUNKING === "true";
}

/**
 * The vector-index writes for one memory: one job per chunk.
 *
 * Chunk ids are `${memoryId}#${i}` and are only used when the memory
 * actually produced more than one chunk, so single-chunk memories keep
 * their bare id and stay wire-identical to the unchunked path. `#` is
 * collision-free against generateId's `prefix_base36_hex`.
 *
 * Every chunk is prefixed with the title and the concepts. Concepts are
 * the hand-curated tags and are invisible to the vector leg otherwise —
 * the unchunked path embeds `title + " " + content` only. Files are
 * deliberately excluded: paths are noise to an embedder, and they
 * already reach BM25 via SearchIndex.extractTerms.
 */
export function memoryChunkJobs(
  memory: Memory,
): Array<{ id: string; text: string }> {
  const flat = memory.title + " " + memory.content;
  if (!isMemoryChunkingEnabled()) return [{ id: memory.id, text: flat }];

  const header = [memory.title, (memory.concepts ?? []).join(", ")]
    .filter((p) => p.length > 0)
    .join(" | ");
  const chunks = chunkText(
    memory.content,
    envNumber("AGENTMEMORY_CHUNK_MAX_CHARS", 1200),
    envNumber("AGENTMEMORY_CHUNK_OVERLAP_CHARS", 150),
  );
  return chunks.map((chunk, i) => ({
    id: chunks.length > 1 ? `${memory.id}#${i}` : memory.id,
    text: header + "\n\n" + chunk,
  }));
}
