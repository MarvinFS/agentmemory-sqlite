import { describe, it, expect } from "vitest";
import { stem } from "../src/state/stemmer.js";

describe("stem() memo", () => {
  it("returns the same stem on repeated calls and leaves short words alone", () => {
    const words = ["running", "connections", "happiness", "relational", "caresses", "ok", "agentmemory"];
    const first = words.map(stem);
    const second = words.map(stem);
    expect(second).toEqual(first);
    expect(stem("running")).toBe("run");
    // The detached key copy must not alter code units: a lone surrogate comes back as is.
    expect(stem("\uD800")).toBe("\uD800");
    expect(stem("abc\uD800def")).toBe("abc\uD800def");
    expect(stem("ok")).toBe("ok");
  });
});
