import { describe, it, expect, afterEach } from "vitest";
import { SearchIndex } from "../src/state/search-index.js";
import type { CompressedObservation } from "../src/types.js";

function doc(id: string, narrative: string): CompressedObservation {
  return {
    id,
    sessionId: "ses_1",
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

// Filler so idf and avgDocLen are not degenerate — with a 2-document corpus
// every idf collapses and the effect under test is invisible.
function corpus(): SearchIndex {
  const idx = new SearchIndex();
  for (let i = 0; i < 200; i++) {
    idx.add(
      doc(
        `obs_filler_${i}`,
        `Filler observation ${i} concerning servers, disks, networks and backups.`,
      ),
    );
  }
  return idx;
}

const scoreOf = (idx: SearchIndex, id: string, q: string) =>
  idx.search(q, 50).find((h) => h.obsId === id)?.score ?? 0;

const tagList = (n: number) =>
  `available tags: ${Array.from({ length: n }, (_, i) => `v1.${i}.0`).join(" ")}`;

describe("SearchIndex prefix expansion is bounded", () => {
  afterEach(() => {
    delete process.env.AGENTMEMORY_PREFIX_MATCH;
  });

  it("does not grow without bound as distinct prefix siblings are added", () => {
    // The defect: each distinct term sharing the query prefix ("v1" matching
    // v1.0.0, v1.1.0, ...) added its own full BM25 contribution, so a tag
    // list outscored every exact match. 80 siblings measured 110.01 against
    // a ~22 single-term ceiling of idf_max * (k1 + 1).
    const few = corpus();
    few.add(doc("obs_tags", tagList(1)));
    const many = corpus();
    many.add(doc("obs_tags", tagList(80)));

    const oneSibling = scoreOf(few, "obs_tags", "v1");
    const manySiblings = scoreOf(many, "obs_tags", "v1");

    expect(manySiblings).toBeLessThanOrEqual(oneSibling);
    // Hard ceiling: idf is bounded by log(N + 1) and the prefix leg carries a
    // deliberate 0.5 damping factor, so one query term can never exceed this.
    const ceiling = Math.log(201 + 1) * 0.5 * (1.2 + 1);
    expect(manySiblings).toBeLessThan(ceiling);
  });

  it("keeps a prefix-only match below a genuine exact match", () => {
    const idx = corpus();
    idx.add(doc("obs_tags", tagList(40)));
    idx.add(doc("obs_exact", "This document is about v1 and nothing else."));

    expect(scoreOf(idx, "obs_exact", "v1")).toBeGreaterThan(
      scoreOf(idx, "obs_tags", "v1"),
    );
  });

  it("leaves exact-term tf saturation untouched", () => {
    // Guards the blast radius: the fix must change prefix handling only. An
    // exact term repeated N times still saturates in tf exactly as before.
    const idx = corpus();
    idx.add(doc("obs_rep", `tags: ${Array(80).fill("v1").join(" ")}`));

    const score = scoreOf(idx, "obs_rep", "v1");
    expect(score).toBeGreaterThan(9);
    expect(score).toBeLessThan(12);
  });

  it("AGENTMEMORY_PREFIX_MATCH=sum restores the additive behaviour", () => {
    process.env.AGENTMEMORY_PREFIX_MATCH = "sum";
    const few = corpus();
    few.add(doc("obs_tags", tagList(1)));
    const many = corpus();
    many.add(doc("obs_tags", tagList(80)));

    expect(scoreOf(many, "obs_tags", "v1")).toBeGreaterThan(
      scoreOf(few, "obs_tags", "v1") * 10,
    );
  });
});
