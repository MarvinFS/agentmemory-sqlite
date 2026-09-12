import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { HybridSearch } from "../src/state/hybrid-search.js";
import { SearchIndex } from "../src/state/search-index.js";
import { VectorIndex } from "../src/state/vector-index.js";
import type { CompressedObservation, EmbeddingProvider } from "../src/types.js";

const KNOBS = [
  "AGENTMEMORY_RETRIEVAL_DEPTH",
  "AGENTMEMORY_RRF_K",
  "AGENTMEMORY_MAX_PER_SESSION",
  "AGENTMEMORY_DIVERSITY_EXEMPT_MEMORY_SESSION",
  "AGENTMEMORY_MEMORY_LAYER_BOOST",
  "AGENTMEMORY_FUSION",
] as const;

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

function obs(
  id: string,
  sessionId: string,
  narrative: string,
): CompressedObservation {
  return {
    id,
    sessionId,
    timestamp: new Date().toISOString(),
    type: "decision",
    title: id,
    facts: [],
    narrative,
    concepts: [],
    files: [],
    importance: 5,
  };
}

// Deterministic 2-d embedder: the query and doc A point the same way,
// doc B is orthogonal. Lets the fused arithmetic be asserted exactly.
const embedder: EmbeddingProvider = {
  name: "test",
  dimensions: 2,
  embed: async () => new Float32Array([1, 0]),
  embedBatch: async (t: string[]) => t.map(() => new Float32Array([1, 0])),
};

describe("hybrid ranking knobs", () => {
  const saved: Record<string, string | undefined> = {};
  let kv: ReturnType<typeof mockKV>;

  beforeEach(() => {
    for (const k of KNOBS) {
      saved[k] = process.env[k];
      delete process.env[k];
    }
    kv = mockKV();
  });
  afterEach(() => {
    for (const k of KNOBS) {
      if (saved[k] === undefined) delete process.env[k];
      else process.env[k] = saved[k]!;
    }
  });

  // Two docs, both in BM25 and both in the vector index. Doc A wins BM25
  // (higher tf in a shorter doc); both are cosine-1 against the query.
  async function twoDocFixture() {
    const bm25 = new SearchIndex();
    const vector = new VectorIndex();
    const a = obs("obs_a", "ses_1", "kappa kappa kappa");
    const b = obs("obs_b", "ses_2", "kappa lambda mu nu xi omicron pi rho");
    for (const o of [a, b]) {
      bm25.add(o);
      vector.add(o.id, o.sessionId, new Float32Array([1, 0]));
      await kv.set(`mem:obs:${o.sessionId}`, o.id, o);
    }
    return { bm25, vector };
  }

  it("RRF is the default and uses K=60", async () => {
    const { bm25, vector } = await twoDocFixture();
    const hybrid = new HybridSearch(bm25, vector, embedder, kv as never);
    const results = await hybrid.search("kappa", 10);

    expect(results.map((r) => r.observation.id)).toEqual(["obs_a", "obs_b"]);
    // rank 1 in both legs: 0.4/(60+1) + 0.6/(60+1) === 1/61
    expect(results[0].combinedScore).toBe(0.4 * (1 / 61) + 0.6 * (1 / 61));
    // rank 2 in BM25, rank 2 in vector (ties break on insertion order)
    expect(results[1].combinedScore).toBe(0.4 * (1 / 62) + 0.6 * (1 / 62));
  });

  it("AGENTMEMORY_RRF_K changes the rank constant", async () => {
    process.env.AGENTMEMORY_RRF_K = "10";
    const { bm25, vector } = await twoDocFixture();
    const hybrid = new HybridSearch(bm25, vector, embedder, kv as never);
    const results = await hybrid.search("kappa", 10);

    expect(results[0].combinedScore).toBe(0.4 * (1 / 11) + 0.6 * (1 / 11));
    // Smaller K sharpens the gap between adjacent ranks — that is the
    // whole point of exposing it.
    const gapK10 = results[0].combinedScore - results[1].combinedScore;
    expect(gapK10).toBeGreaterThan(1 / 61 - 1 / 62);
  });

  it("layer boost at 1.0 leaves scores bit-identical", async () => {
    const f1 = await twoDocFixture();
    const plain = await new HybridSearch(
      f1.bm25,
      f1.vector,
      embedder,
      kv as never,
    ).search("kappa", 10);

    process.env.AGENTMEMORY_MEMORY_LAYER_BOOST = "1.0";
    const boosted = await new HybridSearch(
      f1.bm25,
      f1.vector,
      embedder,
      kv as never,
    ).search("kappa", 10);

    expect(boosted.map((r) => r.combinedScore)).toEqual(
      plain.map((r) => r.combinedScore),
    );
  });

  it("layer boost multiplies mem_* rows and can reorder them above observations", async () => {
    const bm25 = new SearchIndex();
    const vector = new VectorIndex();
    // The observation is the STRONGER lexical match; the memory is the
    // weaker one. Without a boost the observation wins.
    const o = obs("obs_x", "ses_1", "kappa kappa kappa");
    const m = obs("mem_x", "memory", "kappa lambda mu nu");
    for (const d of [o, m]) {
      bm25.add(d);
      vector.add(d.id, d.sessionId, new Float32Array([1, 0]));
    }
    await kv.set("mem:obs:ses_1", "obs_x", o);
    await kv.set("mem:memories", "mem_x", {
      id: "mem_x",
      createdAt: new Date().toISOString(),
      updatedAt: new Date().toISOString(),
      type: "fact",
      title: "mem_x",
      content: "kappa lambda mu nu",
      concepts: [],
      files: [],
      sessionIds: [],
      strength: 7,
      version: 1,
      isLatest: true,
    });

    const before = await new HybridSearch(bm25, vector, embedder, kv as never)
      .search("kappa", 10);
    expect(before[0].observation.id).toBe("obs_x");

    process.env.AGENTMEMORY_MEMORY_LAYER_BOOST = "1.25";
    const after = await new HybridSearch(bm25, vector, embedder, kv as never)
      .search("kappa", 10);

    expect(after[0].observation.id).toBe("mem_x");
    const mem = after.find((r) => r.observation.id === "mem_x")!;
    const obsRow = after.find((r) => r.observation.id === "obs_x")!;
    // Memory was rank 2 in BM25, rank 2 in vector, then boosted.
    expect(mem.combinedScore).toBe((0.4 * (1 / 62) + 0.6 * (1 / 62)) * 1.25);
    // The observation is untouched.
    expect(obsRow.combinedScore).toBe(0.4 * (1 / 61) + 0.6 * (1 / 61));
  });

  it("norm fusion is exact: min-max on BM25, absolute map on cosine", async () => {
    process.env.AGENTMEMORY_FUSION = "norm";
    const { bm25, vector } = await twoDocFixture();
    const hybrid = new HybridSearch(bm25, vector, embedder, kv as never);
    const results = await hybrid.search("kappa", 10);

    // Top BM25 hit normalises to 1, bottom to 0; cosine is 1 for both, so
    // vecNorm = (1 + 1) / 2 = 1.
    expect(results[0].combinedScore).toBe(0.4 * 1 + 0.6 * 1);
    expect(results[1].combinedScore).toBe(0.4 * 0 + 0.6 * 1);
    // Magnitudes are an order of magnitude above the RRF band (0.004-0.017).
    expect(results[0].combinedScore).toBeGreaterThan(0.5);
  });

  it("norm fusion keeps the layer boost multiplicative", async () => {
    process.env.AGENTMEMORY_FUSION = "norm";
    process.env.AGENTMEMORY_MEMORY_LAYER_BOOST = "1.25";
    const bm25 = new SearchIndex();
    const vector = new VectorIndex();
    const m = obs("mem_y", "memory", "kappa kappa kappa");
    const o = obs("obs_y", "ses_1", "kappa lambda mu nu xi omicron");
    for (const d of [m, o]) {
      bm25.add(d);
      vector.add(d.id, d.sessionId, new Float32Array([1, 0]));
      await kv.set(`mem:obs:${d.sessionId}`, d.id, d);
    }
    await kv.set("mem:obs:memory", "mem_y", m);

    const results = await new HybridSearch(bm25, vector, embedder, kv as never)
      .search("kappa", 10);
    const mem = results.find((r) => r.observation.id === "mem_y")!;
    expect(mem.combinedScore).toBe((0.4 * 1 + 0.6 * 1) * 1.25);
  });

  // 12 memories all sharing sessionId "memory" plus 30 observations
  // spread over 10 sessions — enough candidates that the cap actually
  // binds instead of the fallback fill papering over it.
  async function crowdedFixture() {
    const bm25 = new SearchIndex();
    for (let i = 0; i < 12; i++) {
      const m = obs(`mem_${i}`, "memory", "kappa kappa kappa kappa");
      bm25.add(m);
      await kv.set("mem:memories", m.id, {
        id: m.id,
        createdAt: m.timestamp,
        updatedAt: m.timestamp,
        type: "fact",
        title: m.id,
        content: m.narrative,
        concepts: [],
        files: [],
        sessionIds: [],
        strength: 7,
        version: 1,
        isLatest: true,
      });
    }
    for (let s = 0; s < 10; s++) {
      for (let i = 0; i < 3; i++) {
        const o = obs(`obs_${s}_${i}`, `ses_${s}`, "kappa lambda mu");
        bm25.add(o);
        await kv.set(`mem:obs:ses_${s}`, o.id, o);
      }
    }
    return bm25;
  }

  it("caps the memory session at maxPerSession by default", async () => {
    const bm25 = await crowdedFixture();
    const results = await new HybridSearch(bm25, null, null, kv as never)
      .search("kappa", 20);

    const memoryRows = results.filter((r) => r.sessionId === "memory");
    expect(results).toHaveLength(20);
    expect(memoryRows).toHaveLength(3);
  });

  it("AGENTMEMORY_MAX_PER_SESSION raises the cap", async () => {
    process.env.AGENTMEMORY_MAX_PER_SESSION = "5";
    const bm25 = await crowdedFixture();
    const results = await new HybridSearch(bm25, null, null, kv as never)
      .search("kappa", 20);

    expect(results.filter((r) => r.sessionId === "memory")).toHaveLength(5);
  });

  it("AGENTMEMORY_DIVERSITY_EXEMPT_MEMORY_SESSION exempts the memory layer entirely", async () => {
    process.env.AGENTMEMORY_DIVERSITY_EXEMPT_MEMORY_SESSION = "true";
    const bm25 = await crowdedFixture();
    const results = await new HybridSearch(bm25, null, null, kv as never)
      .search("kappa", 20);

    // All 12 memories now compete per-document, not for 3 shared slots.
    expect(results.filter((r) => r.sessionId === "memory")).toHaveLength(12);
    // Real sessions are still capped.
    for (let s = 0; s < 10; s++) {
      expect(
        results.filter((r) => r.sessionId === `ses_${s}`).length,
      ).toBeLessThanOrEqual(3);
    }
  });

  it("returns results in descending combinedScore even when the cap forces a fallback fill", async () => {
    const bm25 = await crowdedFixture();
    const results = await new HybridSearch(bm25, null, null, kv as never)
      .search("kappa", 25);

    for (let i = 1; i < results.length; i++) {
      expect(results[i - 1].combinedScore).toBeGreaterThanOrEqual(
        results[i].combinedScore,
      );
    }
  });

  it("finds a doc that is deep in the vector leg even at a small limit", async () => {
    // The defect: both legs were fetched at `limit * 2`, so a document strong
    // in BM25 but deep in the vector leg fell outside the vector fetch at
    // small limits. Under RRF that is unrecoverable — the vector term is
    // worth up to 0.6/61 while the entire BM25 term tops out at 0.4/61 — so
    // results were non-monotonic in `limit`: absent at 15, rank 5 at 60.
    // 80 filler docs, each a weaker BM25 match than the target and each
    // nearer in the vector leg. The target is the BEST lexical match but sits
    // at vector rank 81 — far outside `limit * 2` for a small limit.
    async function deepFixture() {
      const bm25 = new SearchIndex();
      const vector = new VectorIndex();
      for (let i = 0; i < 80; i++) {
        const o = obs(`obs_${i}`, `ses_${i}`, "lambda mu nu xi omicron");
        bm25.add(o);
        const angle = ((i + 1) / 400) * Math.PI;
        vector.add(o.id, o.sessionId, new Float32Array([Math.cos(angle), Math.sin(angle)]));
        await kv.set(`mem:obs:${o.sessionId}`, o.id, o);
      }
      const target = obs("obs_target", "ses_target", "kappa kappa kappa");
      bm25.add(target);
      // Orthogonal to the query — dead last in the vector leg.
      vector.add(target.id, target.sessionId, new Float32Array([0, 1]));
      await kv.set("mem:obs:ses_target", target.id, target);
      return { bm25, vector };
    }

    const { bm25, vector } = await deepFixture();
    const results = await new HybridSearch(
      bm25,
      vector,
      embedder,
      kv as never,
    ).search("kappa", 5);

    // Under the old `limit * 2` fetch the vector leg saw only the 10 nearest
    // vectors, so obs_target was absent from it and could not compete.
    expect(results.map((r) => r.observation.id)).toContain("obs_target");
  });

  it("AGENTMEMORY_RETRIEVAL_DEPTH=0 restores the limit*2 fetch", async () => {
    process.env.AGENTMEMORY_RETRIEVAL_DEPTH = "0";
    const bm25 = new SearchIndex();
    const vector = new VectorIndex();
    for (let i = 0; i < 80; i++) {
      const o = obs(`obs_${i}`, `ses_${i}`, "lambda mu nu xi omicron");
      bm25.add(o);
      const angle = ((i + 1) / 400) * Math.PI;
      vector.add(o.id, o.sessionId, new Float32Array([Math.cos(angle), Math.sin(angle)]));
      await kv.set(`mem:obs:${o.sessionId}`, o.id, o);
    }
    const target = obs("obs_target", "ses_target", "kappa kappa kappa");
    bm25.add(target);
    vector.add(target.id, target.sessionId, new Float32Array([0, 1]));
    await kv.set("mem:obs:ses_target", target.id, target);

    const results = await new HybridSearch(bm25, vector, embedder, kv as never)
      .search("kappa", 5);

    // The defect, deliberately reproduced: the target is the best lexical
    // match but invisible to a 10-deep vector fetch.
    expect(results.map((r) => r.observation.id)).not.toContain("obs_target");
  });
});
