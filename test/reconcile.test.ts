import { describe, it, expect, beforeEach, vi } from "vitest";
import { IndexPersistence } from "../src/state/index-persistence.js";
import { SearchIndex } from "../src/state/search-index.js";
import { VectorIndex } from "../src/state/vector-index.js";
import { KV } from "../src/state/schema.js";

// A minimal in-memory StateKV stand-in. Only get/set/delete/list are used
// by the checkpoint path.
function makeKv(seed: Record<string, Record<string, unknown>> = {}) {
  const store = new Map<string, Map<string, unknown>>();
  for (const [scope, entries] of Object.entries(seed)) {
    store.set(scope, new Map(Object.entries(entries)));
  }
  const failures = new Set<string>();
  const kv = {
    failOn: (scope: string, key: string) => failures.add(`${scope}\0${key}`),
    dump: () => store,
    async get<T>(scope: string, key: string): Promise<T | null> {
      return (store.get(scope)?.get(key) as T) ?? null;
    },
    async set<T>(scope: string, key: string, value: T): Promise<void> {
      if (failures.has(`${scope}\0${key}`)) {
        throw new Error(`injected write failure: ${scope}/${key}`);
      }
      if (!store.has(scope)) store.set(scope, new Map());
      store.get(scope)!.set(key, value);
    },
    async delete(scope: string, key: string): Promise<void> {
      store.get(scope)?.delete(key);
    },
    async list<T>(scope: string): Promise<T[]> {
      return [...(store.get(scope)?.values() ?? [])] as T[];
    },
  };
  return kv;
}

function makeIndex(ids: string[]): SearchIndex {
  const idx = new SearchIndex();
  for (const id of ids) {
    idx.add({
      id,
      sessionId: "s1",
      title: `title ${id}`,
      narrative: `narrative body for ${id}`,
      timestamp: "2026-08-09T00:00:00.000Z",
      type: "decision",
      facts: [],
      concepts: [],
      files: [],
    } as never);
  }
  return idx;
}

describe("saveCheckpoint - A2 publication protocol", () => {
  let kv: ReturnType<typeof makeKv>;

  beforeEach(() => {
    kv = makeKv();
    vi.spyOn(console, "warn").mockImplementation(() => undefined);
  });

  it("publishes BM25 and vector under ONE generation", async () => {
    const bm25 = makeIndex(["a", "b"]);
    const vector = new VectorIndex();
    vector.add("a", "s1", new Float32Array([1, 0]));
    const p = new IndexPersistence(kv as never, bm25, vector);

    const checkpoint = await p.saveCheckpoint(
      bm25.serialize(),
      vector.serialize(),
    );

    const bm25Manifest = await kv.get<{ generation: string }>(
      KV.bm25Index,
      "data:manifest",
    );
    const vectorManifest = await kv.get<{ generation: string }>(
      KV.bm25Index,
      "vectors:manifest",
    );
    expect(bm25Manifest?.generation).toBe(checkpoint.generation);
    expect(vectorManifest?.generation).toBe(checkpoint.generation);
  });

  it("retains the predecessor generation's shards so rollback has an artifact", async () => {
    const first = makeIndex(["a"]);
    const p = new IndexPersistence(kv as never, first, null);
    const gen1 = await p.saveCheckpoint(first.serialize(), null);
    const gen1Shards = (
      await kv.get<{ shards: Array<{ scope: string; key: string }> }>(
        KV.bm25Index,
        "data:manifest",
      )
    )!.shards;

    const second = makeIndex(["a", "b", "c"]);
    const gen2 = await p.saveCheckpoint(second.serialize(), null);

    expect(gen2.generation).not.toBe(gen1.generation);
    expect(gen2.previous.bm25?.generation).toBe(gen1.generation);
    // The whole point: gen1's shards are still readable after gen2 publishes.
    for (const shard of gen1Shards) {
      expect(await kv.get<string>(shard.scope, shard.key)).toBeTypeOf("string");
    }
    const rollback = await p.readCheckpointRollback();
    expect(rollback?.previous.bm25?.generation).toBe(gen1.generation);
  });

  it("does not half-publish: a failed vector manifest restores the BM25 manifest", async () => {
    const before = makeIndex(["old"]);
    const p = new IndexPersistence(kv as never, before, null);
    await p.saveCheckpoint(before.serialize(), null);
    const originalManifest = await kv.get<{ generation: string }>(
      KV.bm25Index,
      "data:manifest",
    );

    const bm25 = makeIndex(["a", "b"]);
    const vector = new VectorIndex();
    vector.add("a", "s1", new Float32Array([1, 0]));
    kv.failOn(KV.bm25Index, "vectors:manifest");

    await expect(
      p.saveCheckpoint(bm25.serialize(), vector.serialize()),
    ).rejects.toThrow(/injected write failure/);

    const after = await kv.get<{ generation: string }>(
      KV.bm25Index,
      "data:manifest",
    );
    expect(after?.generation).toBe(originalManifest?.generation);
  });
});

describe("reconcile - fail-closed embedding", () => {
  it("aborts rather than publishing a baseline missing vectors", async () => {
    vi.resetModules();
    const kv = makeKv({
      [KV.memories]: {
        m1: {
          id: "m1",
          title: "t",
          content: "some content",
          isLatest: true,
          sessionIds: [],
          concepts: [],
          files: [],
          type: "fact",
        },
      },
      [KV.sessions]: {},
    });

    vi.doMock("../src/functions/search.js", () => ({
      getSearchIndex: () => new SearchIndex(),
      getVectorIndex: () => new VectorIndex(),
      getEmbeddingProvider: () => ({
        name: "stub",
        dimensions: 2,
        // The live path tolerates this and counts a failure. The reconcile
        // must not.
        embedBatch: async () => {
          throw new Error("embed endpoint down");
        },
      }),
      clipEmbedInput: (t: string) => t,
      isIndexExcluded: () => false,
    }));

    const { reconcileIndexes } = await import("../src/functions/reconcile.js");
    const persistence = {
      saveCheckpoint: vi.fn(),
    };

    await expect(
      reconcileIndexes(kv as never, persistence as never, { publish: true }),
    ).rejects.toThrow(/embed endpoint down/);
    expect(persistence.saveCheckpoint).not.toHaveBeenCalled();
  });
});
