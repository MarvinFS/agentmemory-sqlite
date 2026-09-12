import { describe, it, expect } from "vitest";
import {
  datesIn,
  refersToDifferentDates,
} from "../src/state/memory-utils.js";
import { jaccardSimilarity } from "../src/state/schema.js";

// The two real failures this guard exists to stop, reconstructed from the
// step 5 census. Both are consecutive runs of the same daily report, and both
// scored above the threshold that hides one memory behind another.
const DIGEST_07_14 =
  "2026-07-14 programming-humor-html-digest heartbeat run completed. " +
  "Reddit top JSON remained blocked with 403, while old Reddit plus RSS worked. " +
  "Parse summary: 328 raw posts, 250 unique posts, 110 eligible candidates. " +
  "Updated programming-humor-reported-posts.json from 261 to 281 entries.";
const DIGEST_07_15 =
  "2026-07-15 programming-humor-html-digest heartbeat run completed. " +
  "Reddit top JSON remained blocked with 403, while old Reddit plus RSS worked. " +
  "Parse summary: 321 raw posts, 244 unique posts, 105 eligible candidates. " +
  "Updated programming-humor-reported-posts.json from 281 to 301 entries.";

describe("distinct-date guard", () => {
  it("the failing pair really is above both thresholds (guard is load-bearing)", () => {
    const sim = jaccardSimilarity(
      DIGEST_07_14.toLowerCase(),
      DIGEST_07_15.toLowerCase(),
    );
    // > 0.7 supersedes in remember; > 0.9 demotes in auto-forget. If this
    // assertion ever fails the fixture stopped reproducing the bug.
    expect(sim).toBeGreaterThan(0.7);
  });

  it("blocks two different days", () => {
    expect(refersToDifferentDates(DIGEST_07_14, DIGEST_07_15)).toBe(true);
  });

  it("allows a same-day correction, which is what supersession is FOR", () => {
    const corrected = DIGEST_07_14.replace("328 raw posts", "329 raw posts");
    expect(refersToDifferentDates(DIGEST_07_14, corrected)).toBe(false);
  });

  it("does not fire when only one side is dated", () => {
    expect(refersToDifferentDates(DIGEST_07_14, "an undated note")).toBe(false);
    expect(refersToDifferentDates("an undated note", DIGEST_07_14)).toBe(false);
  });

  it("does not fire when the rows share any date", () => {
    expect(
      refersToDifferentDates(
        "covers 2026-07-14 and 2026-07-15",
        "covers 2026-07-15 only",
      ),
    ).toBe(false);
  });

  it("ignores non-dates that look numeric", () => {
    expect(datesIn("version 1.2.3 build 2026-13-45")).toEqual(new Set());
    expect(datesIn("shipped 2026-08-10 ok")).toEqual(new Set(["2026-08-10"]));
  });
});
