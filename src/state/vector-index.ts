// Pass byteOffset + byteLength explicitly so the round-trip survives
// Node's Buffer pool. Buffer.from(b64, "base64") returns a slice of a
// shared 8KB pool (poolSize), and `new Float32Array(buf.buffer)` ignores
// the slice metadata — it would mint a 2048-element view over the whole
// pool. Same risk on the encode side if the input Float32Array is itself
// a sliced view. Reported as a phantom "2048 dimensions on disk" crash
// in #455 / #469 / #584 / #587.
function float32ToBase64(arr: Float32Array): string {
  return Buffer.from(arr.buffer, arr.byteOffset, arr.byteLength).toString(
    "base64",
  );
}

function base64ToFloat32(b64: string): Float32Array {
  const buf = Buffer.from(b64, "base64");
  return new Float32Array(
    buf.buffer,
    buf.byteOffset,
    buf.byteLength / Float32Array.BYTES_PER_ELEMENT,
  );
}

function cosineSimilarity(a: Float32Array, b: Float32Array): number {
  if (a.length !== b.length) return 0;
  let dot = 0;
  let normA = 0;
  let normB = 0;
  for (let i = 0; i < a.length; i++) {
    dot += a[i] * b[i];
    normA += a[i] * a[i];
    normB += b[i] * b[i];
  }
  const denom = Math.sqrt(normA) * Math.sqrt(normB);
  return denom === 0 ? 0 : dot / denom;
}

// Chunked documents store one vector per chunk under `${parentId}#${i}`.
// `#` cannot appear in a generateId output (`prefix_base36_hex`), so the
// split is unambiguous.
export function parentIdOf(vectorId: string): string {
  const hash = vectorId.indexOf("#");
  return hash === -1 ? vectorId : vectorId.slice(0, hash);
}

// How far past `limit` to scan before collapsing chunks back to parents.
// A 10-chunk memory occupying 10 raw slots would otherwise underfill the
// page. Only applied when the index actually contains chunks.
const CHUNK_FANOUT = 4;

export type VectorRow = {
  id: string;
  sessionId: string;
  embedding: Float32Array;
  // SHA-256 of the clipped text this vector was computed for; "" when the
  // caller has none (non-inproc paths, which never reach a store anyway).
  inputHash: string;
};

// Durable rows behind the in-memory map (src/engine/inproc/vectors.ts). Each
// mutation of the map first goes to the store, and the map is changed only
// after the store's enclosing transaction commits, so the map never holds a
// vector the disk does not.
export interface VectorStore {
  // Upsert one row. A parent id (no `#`) also drops every `${id}#*` row.
  put(row: VectorRow): void;
  // Delete `id` and every `${id}#*` chunk row; with `exact`, only `id`.
  remove(id: string, exact?: boolean): void;
  clear(): void;
  // Run `fn` once the enclosing transaction has committed; immediately when
  // no transaction is open. Never runs after a rollback.
  afterCommit(fn: () => void): void;
}

export class VectorIndex {
  private vectors: Map<string, { embedding: Float32Array; sessionId: string }> =
    new Map();
  // parentId -> its chunk ids. Lets remove(parent) drop every `parent#*`
  // without an O(N) scan of the whole index on every delete.
  private chunkIds: Map<string, Set<string>> = new Map();
  private store: VectorStore | null = null;

  attachStore(store: VectorStore | null): void {
    this.store = store;
  }

  add(obsId: string, sessionId: string, embedding: Float32Array, inputHash = ""): void {
    this.mutate(
      (s) => s.put({ id: obsId, sessionId, embedding, inputHash }),
      () => this.applyAdd(obsId, sessionId, embedding),
    );
  }

  // Content deletion removes a parent and all its chunks. The fill pass prunes
  // one row at a time (`exact`): a stale bare-parent row must not take the
  // memory's current chunk rows with it.
  remove(obsId: string, exact = false): void {
    this.mutate(
      (s) => s.remove(obsId, exact),
      () => this.applyRemove(obsId, exact),
    );
  }

  // Fill the map from persisted rows without any store write. `rows` is in
  // `seq` order, so insertion order matches what was persisted.
  hydrate(rows: Iterable<Pick<VectorRow, "id" | "sessionId" | "embedding">>): void {
    for (const row of rows) this.applyAdd(row.id, row.sessionId, row.embedding);
  }

  private mutate(persist: (s: VectorStore) => void, apply: () => void): void {
    if (!this.store) {
      apply();
      return;
    }
    persist(this.store);
    this.store.afterCommit(apply);
  }

  private applyAdd(obsId: string, sessionId: string, embedding: Float32Array): void {
    this.vectors.set(obsId, { embedding, sessionId });
    const parent = parentIdOf(obsId);
    if (parent === obsId) {
      // Re-adding a document as a SINGLE vector must drop any chunk
      // vectors it previously had, or the stale chunks keep matching and
      // can outrank the current content. Same re-add hazard SearchIndex
      // has; keeping add() idempotent closes it here too.
      const stale = this.chunkIds.get(obsId);
      if (stale) {
        for (const id of stale) this.vectors.delete(id);
        this.chunkIds.delete(obsId);
      }
    } else {
      let set = this.chunkIds.get(parent);
      if (!set) {
        set = new Set();
        this.chunkIds.set(parent, set);
      }
      set.add(obsId);
    }
  }

  private applyRemove(obsId: string, exact = false): void {
    this.vectors.delete(obsId);
    const parent = parentIdOf(obsId);
    if (parent !== obsId) {
      const set = this.chunkIds.get(parent);
      set?.delete(obsId);
      if (set?.size === 0) this.chunkIds.delete(parent);
      return;
    }
    if (exact) return;
    // Callers only ever know the parent id (a memory id, an observation
    // id). Without this, re-saving or forgetting a chunked memory leaves
    // its chunk vectors behind as orphans that still match queries.
    const chunks = this.chunkIds.get(obsId);
    if (chunks) {
      for (const id of chunks) this.vectors.delete(id);
      this.chunkIds.delete(obsId);
    }
  }

  search(
    query: Float32Array,
    limit = 20,
  ): Array<{ obsId: string; sessionId: string; score: number }> {
    // UNGATED collapse, on purpose. This is the rollback contract: the
    // WRITING of chunks is flag-gated, the READING of them never is. If
    // this were gated too, turning the chunking flag off against an
    // already-chunked index would return raw `id#3` rows that no KV
    // lookup can resolve — a config-only rollback would silently gut the
    // vector leg.
    const chunked = this.chunkIds.size > 0;
    const scanLimit = chunked ? limit * CHUNK_FANOUT : limit;

    const results: Array<{
      obsId: string;
      sessionId: string;
      score: number;
    }> = [];
    let minScore = -Infinity;

    for (const [obsId, entry] of this.vectors) {
      const score = cosineSimilarity(query, entry.embedding);
      if (results.length < scanLimit) {
        results.push({ obsId, sessionId: entry.sessionId, score });
        if (results.length === scanLimit) {
          results.sort((a, b) => a.score - b.score);
          minScore = results[0].score;
        }
      } else if (score > minScore) {
        results[0] = { obsId, sessionId: entry.sessionId, score };
        results.sort((a, b) => a.score - b.score);
        minScore = results[0].score;
      }
    }

    results.sort((a, b) => b.score - a.score);
    if (!chunked) return results.slice(0, limit);

    // One row per parent, carrying its best-scoring chunk's score.
    const best = new Map<string, { obsId: string; sessionId: string; score: number }>();
    for (const r of results) {
      const parent = parentIdOf(r.obsId);
      const existing = best.get(parent);
      if (!existing || r.score > existing.score) {
        best.set(parent, {
          obsId: parent,
          sessionId: r.sessionId,
          score: r.score,
        });
      }
    }
    return Array.from(best.values())
      .sort((a, b) => b.score - a.score)
      .slice(0, limit);
  }

  get size(): number {
    return this.vectors.size;
  }

  // Dimension of the stored vectors (null while empty). A query vector of
  // another length scores 0 against every row and looks like a valid leg.
  get dims(): number | null {
    const first = this.vectors.values().next().value;
    return first ? first.embedding.length : null;
  }

  // Walks every stored vector and returns the obsIds whose dimension
  // doesn't match `expected`, plus the set of distinct dimensions seen.
  // Used by the persistence-restore guard in src/index.ts to refuse
  // loading any index containing wrong-dimension vectors — including
  // legacy on-disk indexes written before the live-API dimension guard
  // existed (where a mid-session provider swap could mix dimensions
  // inside a single index). Empty `mismatches` plus a single-entry
  // `seenDimensions` matching `expected` is the only clean state.
  validateDimensions(
    expected: number,
  ): { mismatches: Array<{ obsId: string; dim: number }>; seenDimensions: Set<number> } {
    const mismatches: Array<{ obsId: string; dim: number }> = [];
    const seenDimensions = new Set<number>();
    for (const [obsId, entry] of this.vectors) {
      const dim = entry.embedding.length;
      seenDimensions.add(dim);
      if (dim !== expected) {
        mismatches.push({ obsId, dim });
      }
    }
    return { mismatches, seenDimensions };
  }

  clear(): void {
    this.mutate(
      (s) => s.clear(),
      () => {
        this.vectors.clear();
        this.chunkIds.clear();
      },
    );
  }

  // Snapshot restore for the iii-persisted index. Not used with a store: the
  // inproc boot path hydrates from rows instead.
  restoreFrom(other: VectorIndex): void {
    const src = (other as any).vectors as Map<
      string,
      { embedding: Float32Array; sessionId: string }
    >;
    this.vectors = new Map();
    this.chunkIds = new Map();
    for (const [obsId, entry] of src) {
      this.applyAdd(obsId, entry.sessionId, new Float32Array(entry.embedding));
    }
  }

  serialize(): string {
    const data: Array<[string, { embedding: string; sessionId: string }]> = [];
    for (const [obsId, entry] of this.vectors) {
      data.push([
        obsId,
        {
          embedding: float32ToBase64(entry.embedding),
          sessionId: entry.sessionId,
        },
      ]);
    }
    return JSON.stringify(data);
  }

  static deserialize(json: string): VectorIndex {
    const idx = new VectorIndex();
    let data: unknown;
    try {
      data = JSON.parse(json);
    } catch {
      return idx;
    }
    if (!Array.isArray(data)) return idx;
    for (const row of data) {
      try {
        if (!Array.isArray(row) || row.length < 2) continue;
        const [obsId, entry] = row;
        if (
          typeof obsId !== "string" ||
          typeof entry?.embedding !== "string" ||
          typeof entry?.sessionId !== "string"
        )
          continue;
        // Via add(), not vectors.set(), so the chunk-id side map is
        // rebuilt from a persisted chunked index. Ids are opaque strings
        // on disk, so the serialised format needs no version bump.
        idx.add(obsId, entry.sessionId, base64ToFloat32(entry.embedding));
      } catch {
        continue;
      }
    }
    return idx;
  }
}
