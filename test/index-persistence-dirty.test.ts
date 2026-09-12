import { describe, it, expect, vi, beforeEach, afterEach } from "vitest";
import { IndexPersistence } from "../src/state/index-persistence.js";
import { SearchIndex } from "../src/state/search-index.js";
import { VectorIndex } from "../src/state/vector-index.js";
import type { CompressedObservation } from "../src/types.js";

function obs(id: string): CompressedObservation {
  return {
    id,
    sessionId: "ses_1",
    timestamp: "2026-01-01T00:00:00Z",
    type: "decision",
    title: id,
    facts: [],
    narrative: "some indexed content " + id,
    concepts: [],
    files: [],
    importance: 5,
  };
}

function mockKV() {
  const store = new Map<string, Map<string, unknown>>();
  return {
    writes: 0,
    get: async <T>(scope: string, key: string): Promise<T | null> =>
      (store.get(scope)?.get(key) as T) ?? null,
    set: async function <T>(scope: string, key: string, data: T): Promise<T> {
      // eslint-disable-next-line @typescript-eslint/no-this-alias
      (this as unknown as { writes: number }).writes++;
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

describe("IndexPersistence periodic dirty flush", () => {
  const saved = process.env.AGENTMEMORY_INDEX_SAVE_INTERVAL_MS;

  beforeEach(() => vi.useFakeTimers());
  afterEach(() => {
    vi.useRealTimers();
    if (saved === undefined) delete process.env.AGENTMEMORY_INDEX_SAVE_INTERVAL_MS;
    else process.env.AGENTMEMORY_INDEX_SAVE_INTERVAL_MS = saved;
  });

  it("flushes a dirty index on the interval, and not before", async () => {
    process.env.AGENTMEMORY_INDEX_SAVE_INTERVAL_MS = "1000";
    const kv = mockKV();
    const bm25 = new SearchIndex();
    const p = new IndexPersistence(kv as never, bm25, new VectorIndex());

    bm25.add(obs("obs_1"));
    p.markDirty();

    expect(kv.writes).toBe(0); // nothing written yet
    await vi.advanceTimersByTimeAsync(1100);
    const afterFirst = kv.writes;
    expect(afterFirst).toBeGreaterThan(0);

    // Clean index: the timer keeps ticking but must not re-serialize.
    await vi.advanceTimersByTimeAsync(3000);
    expect(kv.writes).toBe(afterFirst);

    // Dirty again -> flushes again.
    bm25.add(obs("obs_2"));
    p.markDirty();
    await vi.advanceTimersByTimeAsync(1100);
    expect(kv.writes).toBeGreaterThan(afterFirst);

    p.stop();
  });

  it("stop() clears the periodic timer", async () => {
    process.env.AGENTMEMORY_INDEX_SAVE_INTERVAL_MS = "1000";
    const kv = mockKV();
    const bm25 = new SearchIndex();
    const p = new IndexPersistence(kv as never, bm25, null);

    bm25.add(obs("obs_1"));
    p.markDirty();
    p.stop();

    await vi.advanceTimersByTimeAsync(5000);
    expect(kv.writes).toBe(0);
  });

  it("round-trips a live-added document through save and load", async () => {
    process.env.AGENTMEMORY_INDEX_SAVE_INTERVAL_MS = "1000";
    const kv = mockKV();
    const bm25 = new SearchIndex();
    const p = new IndexPersistence(kv as never, bm25, null);

    bm25.add(obs("obs_live"));
    p.markDirty();
    await vi.advanceTimersByTimeAsync(1100);
    p.stop();

    // A fresh boot reading the same KV must see the live-added document.
    const restored = await new IndexPersistence(
      kv as never,
      new SearchIndex(),
      null,
    ).load();
    expect(restored.bm25?.has("obs_live")).toBe(true);
  });
});
