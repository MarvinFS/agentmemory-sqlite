import { describe, it, expect } from "vitest";
import { SearchIndex } from "../src/state/search-index.js";
import type { CompressedObservation } from "../src/types.js";

function obs(
  id: string,
  narrative: string,
  overrides: Partial<CompressedObservation> = {},
): CompressedObservation {
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
    ...overrides,
  };
}

describe("SearchIndex re-add does not corrupt avgDocLen", () => {
  // The corruption is INVISIBLE when you look only at the re-added
  // document: `entries.set` overwrites its termCount correctly. What
  // breaks is `totalDocLength`, which is shared — so the damage shows up
  // on an unrelated third document's score. Hence the control doc.
  it("leaves an unrelated document's score unchanged after a re-add", () => {
    const long = obs("obs_long", "alpha ".repeat(200).trim());
    const control = obs("obs_control", "beta gamma delta");
    const filler = obs("obs_filler", "epsilon zeta eta theta iota kappa");

    const clean = new SearchIndex();
    clean.add(long);
    clean.add(control);
    clean.add(filler);
    const expected = clean.search("beta gamma", 5).find((r) => r.obsId === "obs_control")!;

    const dirty = new SearchIndex();
    dirty.add(long);
    dirty.add(long); // same id, second time
    dirty.add(long); // and a third
    dirty.add(control);
    dirty.add(filler);
    const actual = dirty.search("beta gamma", 5).find((r) => r.obsId === "obs_control")!;

    expect(dirty.size).toBe(clean.size);
    expect(actual.score).toBe(expected.score);
  });

  it("a re-add replaces rather than duplicates the document's own postings", () => {
    const idx = new SearchIndex();
    idx.add(obs("obs_1", "original narrative about postgres"));
    idx.add(obs("obs_1", "replacement narrative about redis"));

    expect(idx.size).toBe(1);
    expect(idx.search("postgres", 5)).toHaveLength(0);
    expect(idx.search("redis", 5).map((r) => r.obsId)).toEqual(["obs_1"]);
  });
});
