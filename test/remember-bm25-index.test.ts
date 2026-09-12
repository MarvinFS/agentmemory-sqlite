import { describe, it, expect, afterEach } from "vitest";
import { SearchIndex } from "../src/state/search-index.js";
import { memoryToIndexDoc, MEMORY_SESSION } from "../src/state/memory-utils.js";
import type { Memory } from "../src/types.js";

// The real projection used by remember.ts, rebuildIndex() and the boot
// backfill. Was duplicated inline here; three copies of the same shape
// is exactly how they drift apart.
const memoryAsIndexable = memoryToIndexDoc;

const SLIM = "AGENTMEMORY_MEMORY_DOC_SLIM";
const savedSlim = process.env[SLIM];
afterEach(() => {
  if (savedSlim === undefined) delete process.env[SLIM];
  else process.env[SLIM] = savedSlim;
});

function makeMemory(overrides: Partial<Memory> = {}): Memory {
  return {
    id: "mem_test_001",
    createdAt: new Date().toISOString(),
    updatedAt: new Date().toISOString(),
    type: "fact",
    title: "BM25 test memory",
    content: "BM25 search returns this memory by keyword match",
    concepts: ["bm25", "search", "test"],
    files: [],
    sessionIds: [],
    strength: 7,
    version: 1,
    isLatest: true,
    ...overrides,
  };
}

describe("SearchIndex.has()", () => {
  it("returns false for unknown ids", () => {
    expect(new SearchIndex().has("mem_unknown")).toBe(false);
  });

  it("returns true after add()", () => {
    const idx = new SearchIndex();
    idx.add(memoryAsIndexable(makeMemory()));
    expect(idx.has("mem_test_001")).toBe(true);
  });
});

describe("memory indexing into SearchIndex (closes #257)", () => {
  it("makes a saved memory findable by keyword search", () => {
    const idx = new SearchIndex();
    idx.add(memoryAsIndexable(makeMemory({
      id: "mem_user_001",
      title: "JWT middleware uses jose for Edge compatibility",
      content: "Chose jose over jsonwebtoken because Cloudflare Workers don't ship Node crypto",
      concepts: ["auth", "jose", "edge"],
    })));

    const hits = idx.search("jose middleware", 5);
    expect(hits).toHaveLength(1);
    expect(hits[0].obsId).toBe("mem_user_001");
  });

  it("returns the memory when the issue's reproduction query is run", () => {
    // From issue #257: user saved a memory containing 'BM25 test'
    // keywords and the search returned empty — recall failure.
    const idx = new SearchIndex();
    idx.add(memoryAsIndexable(makeMemory({
      id: "mem_moy3u6ua_8c6962b668e7",
      title: "BM25 test",
      content: "Confirmed BM25 indexing works for memories saved via memory_save",
      concepts: [],
    })));

    const hits = idx.search("BM25 test", 5);
    expect(hits.length).toBeGreaterThan(0);
    expect(hits[0].obsId).toBe("mem_moy3u6ua_8c6962b668e7");
  });

  it("keeps the full payload shape by default and drops only facts when slim", () => {
    const memory = makeMemory();
    delete process.env[SLIM];
    const fat = memoryAsIndexable(memory);
    expect(fat.facts).toEqual([memory.content]);
    expect(fat.narrative).toBe(memory.content);
    expect(fat.sessionId).toBe(MEMORY_SESSION);

    process.env[SLIM] = "true";
    const slim = memoryAsIndexable(memory);
    expect(slim.facts).toEqual([]);
    // Everything still searchable — content survives in narrative, and
    // the title is only a prefix of it, so nothing is lost lexically.
    expect(slim.narrative).toBe(memory.content);
    expect(slim.title).toBe(memory.title);
    expect(slim.concepts).toEqual(memory.concepts);
  });

  // MEASURED, not assumed. The design note behind this flag claimed the
  // triplicated content inflates docLen ~2.2x and that b=0.75 length
  // normalisation "roughly halves the score", so dropping facts[0] would
  // raise it. That is wrong: duplicating content scales tf AND docLen by
  // the same factor, and BM25 saturates in tf — so the fat doc scores
  // HIGHER, and the gap widens on a corpus where every memory is fat
  // (avgDocLen falls when they all shrink, re-inflating every length
  // ratio). Probed on a synthetic 22.6k-doc corpus: memory-vs-echo score
  // ratio 0.258 fat, 0.205 slim.
  //
  // The flag stays because it is a legitimate knob and defaults to off.
  // This test exists so nobody enables it in production expecting a win.
  it("slim scores no better than fat — the flag is not a ranking win", () => {
    const memory = makeMemory({
      id: "mem_slim_001",
      title: "Teleport app routing",
      content:
        "The mainpc-rdp TCP application is reached through Teleport Connect VNet and needs a residentKey passkey",
      concepts: ["teleport", "vnet"],
    });
    // A short, unrelated document is what sets avgDocLen; the memory's
    // length penalty is measured against it.
    const noise = {
      id: "obs_noise",
      sessionId: "ses_1",
      timestamp: new Date().toISOString(),
      type: "other" as const,
      title: "unrelated",
      facts: [],
      narrative: "nothing to see",
      concepts: [],
      files: [],
      importance: 5,
    };

    delete process.env[SLIM];
    const fatIdx = new SearchIndex();
    fatIdx.add(memoryAsIndexable(memory));
    fatIdx.add(noise);
    const fatScore = fatIdx.search("mainpc-rdp Teleport Connect VNet residentKey", 5)[0];

    process.env[SLIM] = "true";
    const slimIdx = new SearchIndex();
    slimIdx.add(memoryAsIndexable(memory));
    slimIdx.add(noise);
    const slimScore = slimIdx.search("mainpc-rdp Teleport Connect VNet residentKey", 5)[0];

    // Still findable either way — the flag is safe, just not a win.
    expect(fatScore.obsId).toBe("mem_slim_001");
    expect(slimScore.obsId).toBe("mem_slim_001");
    expect(slimScore.score).toBeLessThanOrEqual(fatScore.score);
  });

  it("matches concepts as well as title and content", () => {
    const idx = new SearchIndex();
    idx.add(memoryAsIndexable(makeMemory({
      id: "mem_concept_001",
      title: "Generic title",
      content: "Generic content",
      concepts: ["unique-concept-marker"],
    })));

    const hits = idx.search("unique-concept-marker", 5);
    expect(hits.length).toBeGreaterThan(0);
    expect(hits[0].obsId).toBe("mem_concept_001");
  });
});
