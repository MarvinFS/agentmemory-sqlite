import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";

vi.mock("../src/logger.js", () => ({
  logger: { info: vi.fn(), warn: vi.fn(), error: vi.fn() },
}));

vi.mock("../src/state/keyed-mutex.js", () => ({
  withKeyedLock: <T>(_key: string, fn: () => Promise<T>) => fn(),
}));

import { registerRememberFunction } from "../src/functions/remember.js";
import {
  getSearchIndex,
  setVectorIndex,
  setEmbeddingProvider,
  getVectorIndex,
} from "../src/functions/search.js";
import { VectorIndex } from "../src/state/vector-index.js";
import type { EmbeddingProvider, Memory } from "../src/types.js";

function mockKV() {
  const store = new Map<string, Map<string, unknown>>();
  return {
    get: async <T>(scope: string, key: string): Promise<T | null> =>
      (store.get(scope)?.get(key) as T) ?? null,
    set: async <T>(scope: string, key: string, data: T): Promise<T> => {
      if (!store.has(scope)) store.set(scope, new Map());
      store.get(scope)!.set(key, data);
      return data;
    },
    delete: async (scope: string, key: string): Promise<void> => {
      store.get(scope)?.delete(key);
    },
    list: async <T>(scope: string): Promise<T[]> =>
      Array.from(store.get(scope)?.values() ?? []) as T[],
  };
}

function mockSdk() {
  const functions = new Map<string, Function>();
  return {
    registerFunction: (id: string, handler: Function) => functions.set(id, handler),
    registerTrigger: () => {},
    trigger: async (input: { function_id: string; payload: unknown }) => {
      const fn = functions.get(input.function_id);
      // mem::cascade-update is not registered in this harness; the
      // supersede path fires it and must not care that it is absent.
      if (!fn) return undefined;
      return fn(input.payload);
    },
  };
}

const embedder: EmbeddingProvider = {
  name: "test",
  dimensions: 3,
  embed: async () => new Float32Array([0.1, 0.2, 0.3]),
  embedBatch: async (t: string[]) =>
    t.map(() => new Float32Array([0.1, 0.2, 0.3])),
};

describe("superseding a memory de-indexes the old version", () => {
  beforeEach(() => {
    getSearchIndex().clear();
    setVectorIndex(new VectorIndex());
    setEmbeddingProvider(embedder);
  });
  afterEach(() => {
    getSearchIndex().clear();
    setVectorIndex(null);
    setEmbeddingProvider(null);
  });

  it("drops the superseded id from BM25 and the vector index", async () => {
    const sdk = mockSdk();
    const kv = mockKV();
    registerRememberFunction(sdk as never, kv as never);

    const content =
      "The proxy terminates TLS on port 443 and forwards to the app on 3080";
    const first = (await sdk.trigger({
      function_id: "mem::remember",
      payload: { content, type: "fact" },
    })) as { success: boolean; memory: Memory };
    expect(first.success).toBe(true);

    // >0.7 Jaccard against the first — this is what triggers supersede.
    const second = (await sdk.trigger({
      function_id: "mem::remember",
      payload: {
        content: content + " after the certificate rotation",
        type: "fact",
      },
    })) as { success: boolean; memory: Memory };
    expect(second.memory.supersedes).toEqual([first.memory.id]);

    // The stale version is gone from both live indexes…
    expect(getSearchIndex().has(first.memory.id)).toBe(false);
    expect(getSearchIndex().has(second.memory.id)).toBe(true);
    const vectorIds = getVectorIndex()!
      .search(new Float32Array([0.1, 0.2, 0.3]), 10)
      .map((r) => r.obsId);
    expect(vectorIds).toEqual([second.memory.id]);

    // …but the record itself is retained in KV, flagged isLatest: false.
    const stored = await kv.get<Memory>("mem:memories", first.memory.id);
    expect(stored).not.toBeNull();
    expect(stored!.isLatest).toBe(false);
  });

  it("leaves unrelated memories indexed", async () => {
    const sdk = mockSdk();
    const kv = mockKV();
    registerRememberFunction(sdk as never, kv as never);

    const a = (await sdk.trigger({
      function_id: "mem::remember",
      payload: { content: "Postgres runs on the storage VLAN", type: "fact" },
    })) as { memory: Memory };
    const b = (await sdk.trigger({
      function_id: "mem::remember",
      payload: { content: "Grafana dashboards live in monitoring", type: "fact" },
    })) as { memory: Memory };

    expect(b.memory.supersedes).toEqual([]);
    expect(getSearchIndex().has(a.memory.id)).toBe(true);
    expect(getSearchIndex().has(b.memory.id)).toBe(true);
  });
});
