import { describe, expect, it } from "vitest";
import { HybridSearch } from "../src/state/hybrid-search.js";
import type { QueryExpansion } from "../src/types.js";

// The expansion path is worth exactly two assertions, and both are about the
// seam rather than about relevance:
//
//  1. an empty expansion must not perturb ordering, because a provider outage
//     would otherwise silently reshuffle every page of every search;
//  2. a non-empty expansion must actually issue the extra queries, because the
//     whole point is reaching documents the original query never retrieves.
//
// Both are checked against a stub index, so nothing here needs a provider, an
// embedding key, or a live daemon.

type Row = { obsId: string; sessionId: string; score: number };

function stubIndexes(perQuery: Record<string, Row[]>) {
  const seen: string[] = [];
  const bm25 = {
    search(query: string, _depth: number): Row[] {
      seen.push(query);
      return perQuery[query] ?? [];
    },
  };
  return { bm25, seen };
}

function hybrid(bm25: unknown) {
  // No vector index and no embedding provider: BM25-only mode, which is a
  // supported configuration and keeps the fusion arithmetic single-legged and
  // therefore predictable.
  return new HybridSearch(
    bm25 as never,
    null as never,
    null as never,
    {
      // Every indexed row resolves to a minimal observation. `resolveRow`
      // tries KV.observations first, so answering here keeps the stub to one
      // branch and never reaches the KV.memories fallback.
      get: async (_scope: string, id: string) => ({
        id,
        sessionId: "s1",
        content: id,
        timestamp: new Date().toISOString(),
      }),
      list: async () => [],
      set: async () => undefined,
    } as never,
    0.4,
    0.6,
    0,
  );
}

const EMPTY: QueryExpansion = {
  original: "alpha",
  reformulations: [],
  temporalConcretizations: [],
  entityExtractions: [],
};

describe("searchWithExpansion", () => {
  it("issues only the original query when the expansion is empty", async () => {
    const { bm25, seen } = stubIndexes({
      alpha: [{ obsId: "obs_a", sessionId: "s1", score: 9 }],
    });
    const results = await hybrid(bm25).searchWithExpansion("alpha", 5, EMPTY);

    expect(seen).toEqual(["alpha"]);
    expect(results.map((r) => r.observation.id)).toEqual(["obs_a"]);
  });

  it("returns identical output to a plain search on an empty expansion", async () => {
    const rows = {
      alpha: [
        { obsId: "obs_a", sessionId: "s1", score: 9 },
        { obsId: "obs_b", sessionId: "s1", score: 4 },
        { obsId: "obs_c", sessionId: "s2", score: 1 },
      ],
    };
    const plain = await hybrid(stubIndexes(rows).bm25).search("alpha", 5);
    const expanded = await hybrid(stubIndexes(rows).bm25).searchWithExpansion(
      "alpha",
      5,
      EMPTY,
    );

    expect(expanded.map((r) => r.observation.id)).toEqual(
      plain.map((r) => r.observation.id),
    );
  });

  it("reaches a document only a reformulation retrieves", async () => {
    const { bm25, seen } = stubIndexes({
      "job hunting folder": [{ obsId: "obs_noise", sessionId: "s1", score: 2 }],
      "authoritative repository for the job search project": [
        { obsId: "obs_target", sessionId: "s2", score: 8 },
      ],
    });
    const results = await hybrid(bm25).searchWithExpansion(
      "job hunting folder",
      5,
      {
        original: "job hunting folder",
        reformulations: ["authoritative repository for the job search project"],
        temporalConcretizations: [],
        entityExtractions: [],
      },
    );

    expect(seen).toContain("authoritative repository for the job search project");
    expect(results.map((r) => r.observation.id)).toContain("obs_target");
  });
});
