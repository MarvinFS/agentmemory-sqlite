import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { VectorIndex, parentIdOf } from "../src/state/vector-index.js";

const v = (...xs: number[]) => new Float32Array(xs);

describe("VectorIndex chunk handling", () => {
  const saved = process.env.AGENTMEMORY_MEMORY_CHUNKING;

  beforeEach(() => {
    delete process.env.AGENTMEMORY_MEMORY_CHUNKING;
  });
  afterEach(() => {
    if (saved === undefined) delete process.env.AGENTMEMORY_MEMORY_CHUNKING;
    else process.env.AGENTMEMORY_MEMORY_CHUNKING = saved;
  });

  it("parentIdOf strips the chunk suffix and leaves bare ids alone", () => {
    expect(parentIdOf("mem_abc_123#7")).toBe("mem_abc_123");
    expect(parentIdOf("mem_abc_123")).toBe("mem_abc_123");
    expect(parentIdOf("obs_x#0")).toBe("obs_x");
  });

  it("collapses chunks to one row per parent carrying the best chunk score", () => {
    const idx = new VectorIndex();
    // Chunk 1 is the on-topic section; chunks 0 and 2 are off-topic.
    idx.add("mem_a#0", "memory", v(0, 1));
    idx.add("mem_a#1", "memory", v(1, 0));
    idx.add("mem_a#2", "memory", v(0, 1));
    idx.add("obs_b", "ses_1", v(0.7071, 0.7071));

    const hits = idx.search(v(1, 0), 10);

    expect(hits.map((h) => h.obsId)).toEqual(["mem_a", "obs_b"]);
    // Best chunk wins, not the average and not the last one seen.
    expect(hits[0].score).toBeCloseTo(1, 6);
  });

  it("remove(parent) drops every parent#* chunk", () => {
    const idx = new VectorIndex();
    idx.add("mem_a#0", "memory", v(1, 0));
    idx.add("mem_a#1", "memory", v(1, 0));
    idx.add("mem_b", "memory", v(1, 0));
    expect(idx.size).toBe(3);

    idx.remove("mem_a");

    expect(idx.size).toBe(1);
    expect(idx.search(v(1, 0), 10).map((h) => h.obsId)).toEqual(["mem_b"]);
  });

  it("a chunked index searched with chunking OFF still returns parent ids", () => {
    // The L1 rollback contract. Removing the drop-in turns chunk WRITING
    // off, but the on-disk index is still chunked — if the collapse were
    // flag-gated too, this would return unresolvable `mem_a#1` rows and
    // silently gut the vector leg.
    expect(process.env.AGENTMEMORY_MEMORY_CHUNKING).toBeUndefined();

    const idx = new VectorIndex();
    idx.add("mem_a#0", "memory", v(0, 1));
    idx.add("mem_a#1", "memory", v(1, 0));

    const hits = idx.search(v(1, 0), 5);

    expect(hits).toHaveLength(1);
    expect(hits[0].obsId).toBe("mem_a");
    expect(hits[0].obsId).not.toContain("#");
  });

  it("re-adding a parent as a single vector drops its stale chunks", () => {
    const idx = new VectorIndex();
    idx.add("mem_a#0", "memory", v(1, 0));
    idx.add("mem_a#1", "memory", v(1, 0));
    expect(idx.size).toBe(2);

    idx.add("mem_a", "memory", v(0, 1)); // now a single-chunk document

    expect(idx.size).toBe(1);
    const hits = idx.search(v(1, 0), 5);
    // The stale chunks pointed at (1,0); if they survived they would win.
    expect(hits).toHaveLength(1);
    expect(hits[0].score).toBeCloseTo(0, 6);
  });

  it("survives a serialize / deserialize round trip with chunks intact", () => {
    const idx = new VectorIndex();
    idx.add("mem_a#0", "memory", v(0, 1));
    idx.add("mem_a#1", "memory", v(1, 0));

    const restored = VectorIndex.deserialize(idx.serialize());

    expect(restored.search(v(1, 0), 5).map((h) => h.obsId)).toEqual(["mem_a"]);
    restored.remove("mem_a");
    expect(restored.size).toBe(0);
  });

  it("does not over-fetch or collapse when the index holds no chunks", () => {
    const idx = new VectorIndex();
    for (let i = 0; i < 10; i++) idx.add(`obs_${i}`, "ses_1", v(1, i / 100));

    const hits = idx.search(v(1, 0), 3);

    expect(hits).toHaveLength(3);
    expect(hits.every((h) => !h.obsId.includes("#"))).toBe(true);
  });
});
