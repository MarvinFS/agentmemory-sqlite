// The acceptance harness's "new side" (plan step 9.7a), built as its own
// dist entry: over a snapshot of the published agentmemory.sqlite it runs the
// same BM25 rebuild the daemon runs at boot and the same vector hydration, and
// hands back the two indexes so raw legs can be compared offline against the
// old build's deserialized shards. The index classes are re-exported so the
// harness can `remove()` delta ids and read `size` without a second import.
import { DatabaseSync } from "node:sqlite";

import { SqliteState } from "../engine/inproc/state.js";
import { SqliteVectorStore } from "../engine/inproc/vectors.js";
import { getSearchIndex, rebuildBm25FromContent } from "../functions/search.js";
import { enumerateIndexCorpus } from "../state/index-corpus.js";
import { SearchIndex } from "../state/search-index.js";
import { VectorIndex } from "../state/vector-index.js";
import type { CompressedObservation } from "../types.js";

export { SearchIndex, VectorIndex };

// The BM25 documents (as the rebuild produces them) for a set of ids, so the
// harness can re-tokenize a mismatching doc with the old build's index class
// and tell a stale old entry from a tokenizer change. Read-only.
export function docsFor(snapshotPath: string, ids: Iterable<string>): Map<string, CompressedObservation> {
  const want = new Set(ids);
  const out = new Map<string, CompressedObservation>();
  if (want.size === 0) return out;
  const db = new DatabaseSync(snapshotPath, { readOnly: true });
  try {
    for (const item of enumerateIndexCorpus(db)) {
      if (item.doc && want.has(item.doc.id)) out.set(item.doc.id, item.doc);
    }
  } finally {
    db.close();
  }
  return out;
}

export type NewIndexes = {
  bm25: SearchIndex;
  vector: VectorIndex;
  rebuild: { docs: number; rows: number; skipped: number; readMs: number; indexMs: number };
  hydrated: number;
  // The vector ids the content claims today - what the fill pass keeps. A
  // row outside this set is a stale one the fill prunes (a bare parent whose
  // memory is chunked now); a row inside it that vanished is a loss.
  expectedVectorIds: Set<string>;
};

// `snapshotPath` must be a copy (mem::backup's snapshot), not the live file:
// opening it here puts it in WAL mode and checkpoints it on close.
export async function loadNewIndexes(snapshotPath: string): Promise<NewIndexes> {
  const state = new SqliteState(snapshotPath);
  try {
    const rebuild = await rebuildBm25FromContent(state.db);
    const vector = new VectorIndex();
    const hydrated = new SqliteVectorStore(state).hydrate(vector);
    const expectedVectorIds = new Set<string>();
    for (const item of enumerateIndexCorpus(state.db)) for (const job of item.jobs) expectedVectorIds.add(job.id);
    return { bm25: getSearchIndex(), vector, rebuild, hydrated, expectedVectorIds };
  } finally {
    state.close();
  }
}
