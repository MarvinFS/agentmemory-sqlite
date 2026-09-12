import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";

vi.mock("../src/logger.js", () => ({
  logger: { info: vi.fn(), warn: vi.fn(), error: vi.fn() },
}));

import { HybridSearch } from "../src/state/hybrid-search.js";
import {
  registerSearchFunction,
  getSearchIndex,
} from "../src/functions/search.js";
import { SearchIndex } from "../src/state/search-index.js";
import { memoryToObservation, MEMORY_SESSION } from "../src/state/memory-utils.js";
import { KV } from "../src/state/schema.js";
import { initMetrics } from "../src/telemetry/setup.js";
import type { Memory } from "../src/types.js";

const FLAG = "AGENTMEMORY_NONLATEST_FILTER";

// A Memory literal. `isLatest` is deliberately NOT defaulted: a test that
// omits it is exercising a legacy row written before the field existed,
// which is the case the predicate must let through.
function makeMemory(overrides: Partial<Memory> & { id: string }): Memory {
  return {
    content: "The deploy runbook lives in ops and the tripwire verifies it",
    title: "deploy runbook",
    concepts: ["deploy", "runbook"],
    files: [],
    strength: 7,
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    sessionIds: [],
    sourceObservationIds: [],
    supersedes: [],
    type: "architecture",
    version: 1,
    ...overrides,
  } as Memory;
}

function mockKV() {
  const store = new Map<string, Map<string, unknown>>();
  let getCalls = 0;
  return {
    getCallCount: () => getCalls,
    resetCallCount: () => {
      getCalls = 0;
    },
    get: async <T>(scope: string, key: string): Promise<T | null> => {
      getCalls++;
      return (store.get(scope)?.get(key) as T) ?? null;
    },
    set: async <T>(scope: string, key: string, data: T): Promise<T> => {
      if (!store.has(scope)) store.set(scope, new Map());
      store.get(scope)!.set(key, data);
      return data;
    },
    delete: async (scope: string, key: string): Promise<void> => {
      store.get(scope)?.delete(key);
    },
    list: async <T>(scope: string): Promise<T[]> => {
      const entries = store.get(scope);
      return entries ? (Array.from(entries.values()) as T[]) : [];
    },
  };
}

// Counting meter so the A3 counters can be asserted. Before initMetrics the
// accessors hand back no-ops, so without this the counter calls are silent.
function countingMeter() {
  const totals = new Map<string, number>();
  return {
    totals,
    getMeter: () => ({
      createCounter: (name: string) => ({
        add: (n: number) => totals.set(name, (totals.get(name) ?? 0) + n),
      }),
      createHistogram: () => ({ record: () => {} }),
    }),
  };
}

describe("A1 non-latest suppression", () => {
  let bm25: SearchIndex;
  let kv: ReturnType<typeof mockKV>;
  let meter: ReturnType<typeof countingMeter>;

  // Index a memory the way production does: the BM25 entry carries the
  // memory's id + synthetic sessionId, and the record lives in KV.memories
  // with NO observation under KV.observations(sessionId). That absence is
  // what forces the enrichment fallback that A1 gates.
  async function indexMemory(mem: Memory) {
    const doc = memoryToObservation(mem);
    bm25.add(doc);
    await kv.set(KV.memories, mem.id, mem);
  }

  beforeEach(() => {
    bm25 = new SearchIndex();
    kv = mockKV();
    meter = countingMeter();
    initMetrics(meter.getMeter as never);
    delete process.env[FLAG];
  });

  afterEach(() => {
    delete process.env[FLAG];
  });

  // THE CATASTROPHIC CASE. types.ts declares `isLatest: boolean` as
  // required, so the predicate `isLatest !== false` looks redundant and
  // invites simplification to truthiness. Rows written before the field
  // existed carry no value at runtime: truthiness would hide the entire
  // corpus in order to suppress a handful of demoted rows.
  it("returns a legacy memory with no isLatest field, filter ON", async () => {
    process.env[FLAG] = "true";
    await indexMemory(makeMemory({ id: "mem_legacy" }));

    const hybrid = new HybridSearch(bm25, null, null, kv as never);
    const results = await hybrid.search("runbook");

    expect(results.map((r) => r.observation.id)).toEqual(["mem_legacy"]);
  });

  it("returns a memory explicitly marked isLatest:true, filter ON", async () => {
    process.env[FLAG] = "true";
    await indexMemory(makeMemory({ id: "mem_latest", isLatest: true }));

    const hybrid = new HybridSearch(bm25, null, null, kv as never);
    const results = await hybrid.search("runbook");

    expect(results.map((r) => r.observation.id)).toEqual(["mem_latest"]);
  });

  it("excludes a memory explicitly marked isLatest:false, filter ON", async () => {
    process.env[FLAG] = "true";
    await indexMemory(makeMemory({ id: "mem_stale", isLatest: false }));

    const hybrid = new HybridSearch(bm25, null, null, kv as never);
    const results = await hybrid.search("runbook");

    expect(results).toEqual([]);
    expect(meter.totals.get("nonlatest_filtered_total")).toBe(1);
    expect(meter.totals.get("nonlatest_leaked_total")).toBeUndefined();
  });

  it("leaks the stale row and counts it when the filter is OFF", async () => {
    await indexMemory(makeMemory({ id: "mem_stale", isLatest: false }));

    const hybrid = new HybridSearch(bm25, null, null, kv as never);
    const results = await hybrid.search("runbook");

    // Pre-A1 behaviour, unchanged: the stale row is still returned. Only
    // the counter moves, which is how the live incidence gets measured
    // before the enable.
    expect(results.map((r) => r.observation.id)).toEqual(["mem_stale"]);
    expect(meter.totals.get("nonlatest_leaked_total")).toBe(1);
    expect(meter.totals.get("nonlatest_filtered_total")).toBeUndefined();
  });

  // The fill-loop path specifically. diversifyBySession caps each session
  // at maxPerSession (3) in its first pass, then RE-ADMITS the skipped
  // rows whenever the page underfills. Every memory shares the synthetic
  // "memory" sessionId, so a 5-memory corpus always exercises that path:
  // 3 rows come from the capped pass and 2 from the fill loop. If the
  // exclusion were implemented as a skip inside that pipeline, the fill
  // loop would re-admit exactly what A1 removed.
  it("keeps a stale row out even when the fill loop re-admits it", async () => {
    process.env[FLAG] = "true";
    for (const id of ["mem_a", "mem_b", "mem_c", "mem_d"]) {
      await indexMemory(makeMemory({ id }));
    }
    await indexMemory(makeMemory({ id: "mem_stale", isLatest: false }));

    const hybrid = new HybridSearch(bm25, null, null, kv as never);
    const results = await hybrid.search("runbook");
    const ids = results.map((r) => r.observation.id);

    // All five share one session, so more than maxPerSession=3 came back
    // only because the fill loop ran.
    expect(ids.length).toBe(4);
    expect(ids).not.toContain("mem_stale");
    expect(new Set(ids)).toEqual(
      new Set(["mem_a", "mem_b", "mem_c", "mem_d"]),
    );
  });

  it("refills the page from lower-ranked eligible rows instead of underfilling", async () => {
    process.env[FLAG] = "true";
    // Three stale rows interleaved with eligible ones. The page must come
    // back with every eligible row, not short by the number rejected.
    for (const id of ["mem_a", "mem_b", "mem_c"]) {
      await indexMemory(makeMemory({ id }));
    }
    for (const id of ["mem_x", "mem_y", "mem_z"]) {
      await indexMemory(makeMemory({ id, isLatest: false }));
    }

    const hybrid = new HybridSearch(bm25, null, null, kv as never);
    const results = await hybrid.search("runbook");
    const ids = results.map((r) => r.observation.id);

    expect(new Set(ids)).toEqual(new Set(["mem_a", "mem_b", "mem_c"]));
    expect(meter.totals.get("nonlatest_filtered_total")).toBe(3);
  });

  // Inertness as a COUNTED property, not an intention. Deploy one is only
  // "inert" if the disabled path issues the same number of KV reads it did
  // before A1 - that is the claim the whole rollout rests on.
  it("issues the same number of KV reads with the filter off as with no rejections", async () => {
    for (const id of ["mem_a", "mem_b", "mem_c"]) {
      await indexMemory(makeMemory({ id }));
    }

    const hybrid = new HybridSearch(bm25, null, null, kv as never);

    kv.resetCallCount();
    await hybrid.search("runbook");
    const off = kv.getCallCount();

    process.env[FLAG] = "true";
    kv.resetCallCount();
    await hybrid.search("runbook");
    const onNoRejections = kv.getCallCount();

    // Two reads per row (observation miss, then the memories fallback),
    // and enabling the filter must not add any when nothing is rejected.
    expect(off).toBe(6);
    expect(onNoRejections).toBe(off);
  });

  it("resolves only as far as the page needs, not the whole candidate pool", async () => {
    process.env[FLAG] = "true";
    // 30 eligible rows, page size 2. A full-pool materialisation would
    // read all 30 (60 KV calls); progressive refill must stop at the page.
    for (let i = 0; i < 30; i++) {
      await indexMemory(makeMemory({ id: `mem_${i}` }));
    }

    const hybrid = new HybridSearch(bm25, null, null, kv as never);
    kv.resetCallCount();
    const results = await hybrid.search("runbook", 2);

    expect(results.length).toBe(2);
    // retrievalDepth floors at 20, so the first batch is 20 rows / 40
    // reads. The point is that it does not scale with the 3x candidate
    // pool the filter asks diversification for.
    expect(kv.getCallCount()).toBeLessThanOrEqual(40);
  });

  it("terminates when every candidate is ineligible", async () => {
    process.env[FLAG] = "true";
    for (let i = 0; i < 5; i++) {
      await indexMemory(makeMemory({ id: `mem_${i}`, isLatest: false }));
    }

    const hybrid = new HybridSearch(bm25, null, null, kv as never);
    const results = await hybrid.search("runbook");

    // Regression: if the batch cursor advanced only past ACCEPTED rows,
    // an all-ineligible pool would re-resolve its own head forever.
    // Verified by mutation: that bug HANGS this test rather than failing
    // it - the retry loop is pure microtasks, so it starves the macrotask
    // queue and vitest's own testTimeout never gets to fire. A hang here
    // means the cursor advance in enrichResults was broken.
    expect(results).toEqual([]);
    expect(meter.totals.get("nonlatest_filtered_total")).toBe(5);
  });

  it("never filters observations, only memories", async () => {
    process.env[FLAG] = "true";
    const obs = {
      id: "obs_1",
      sessionId: "ses_1",
      timestamp: new Date().toISOString(),
      type: "file_edit" as const,
      title: "deploy runbook edit",
      facts: ["touched the runbook"],
      narrative: "Edited the deploy runbook",
      concepts: ["deploy", "runbook"],
      files: ["ops/runbook.md"],
      importance: 7,
    };
    bm25.add(obs);
    await kv.set(KV.observations("ses_1"), "obs_1", obs);

    const hybrid = new HybridSearch(bm25, null, null, kv as never);
    const results = await hybrid.search("runbook");

    expect(results.map((r) => r.observation.id)).toEqual(["obs_1"]);
    expect(meter.totals.get("nonlatest_filtered_total")).toBeUndefined();
  });

  it("indexes memories under the synthetic session the fallback keys off", async () => {
    // Guards the fixture itself: if memories stopped resolving through the
    // KV.memories fallback, every test above would pass vacuously.
    const mem = makeMemory({ id: "mem_probe" });
    expect(memoryToObservation(mem).sessionId).toBe(MEMORY_SESSION);
  });
});

function mockSdk() {
  const functions = new Map<string, Function>();
  return {
    registerFunction: (idOrOpts: string | { id: string }, handler: Function) => {
      functions.set(
        typeof idOrOpts === "string" ? idOrOpts : idOrOpts.id,
        handler,
      );
    },
    registerTrigger: () => {},
    trigger: async (id: string, payload: unknown) => {
      const fn = functions.get(id);
      if (!fn) throw new Error(`No function: ${id}`);
      return fn(payload);
    },
  };
}

// The second read path. mem::search is what memory_recall reaches through
// server.ts, so fixing only hybrid-search would leave recall still handing
// back stale rows - which is the whole reason A1 applies one shared
// predicate at both sites.
describe("A1 non-latest suppression - mem::search path", () => {
  let sdk: ReturnType<typeof mockSdk>;
  let kv: ReturnType<typeof mockKV>;
  let meter: ReturnType<typeof countingMeter>;

  async function indexMemory(mem: Memory) {
    getSearchIndex().add(memoryToObservation(mem));
    await kv.set(KV.memories, mem.id, mem);
  }

  beforeEach(() => {
    sdk = mockSdk();
    kv = mockKV();
    meter = countingMeter();
    initMetrics(meter.getMeter as never);
    registerSearchFunction(sdk as never, kv as never);
    // Module-level SearchIndex singleton would leak across tests.
    getSearchIndex().clear();
    delete process.env[FLAG];
  });

  afterEach(() => {
    delete process.env[FLAG];
    getSearchIndex().clear();
  });

  async function search(query: string) {
    const result = (await sdk.trigger("mem::search", { query })) as {
      results: Array<{ observation: { id: string } }>;
    };
    return result.results.map((r) => r.observation.id);
  }

  it("returns a legacy memory with no isLatest field, filter ON", async () => {
    process.env[FLAG] = "true";
    await indexMemory(makeMemory({ id: "mem_legacy" }));

    expect(await search("runbook")).toEqual(["mem_legacy"]);
  });

  it("excludes an explicitly non-latest memory, filter ON", async () => {
    process.env[FLAG] = "true";
    await indexMemory(makeMemory({ id: "mem_stale", isLatest: false }));

    expect(await search("runbook")).toEqual([]);
    expect(meter.totals.get("nonlatest_filtered_total")).toBe(1);
  });

  it("leaks the stale row and counts it when the filter is OFF", async () => {
    await indexMemory(makeMemory({ id: "mem_stale", isLatest: false }));

    expect(await search("runbook")).toEqual(["mem_stale"]);
    expect(meter.totals.get("nonlatest_leaked_total")).toBe(1);
  });

  it("refills rather than underfilling when rows are rejected", async () => {
    process.env[FLAG] = "true";
    for (const id of ["mem_a", "mem_b"]) {
      await indexMemory(makeMemory({ id }));
    }
    for (const id of ["mem_x", "mem_y"]) {
      await indexMemory(makeMemory({ id, isLatest: false }));
    }

    expect(new Set(await search("runbook"))).toEqual(
      new Set(["mem_a", "mem_b"]),
    );
    expect(meter.totals.get("nonlatest_filtered_total")).toBe(2);
  });
});
