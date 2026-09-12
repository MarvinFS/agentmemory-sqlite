import { describe, it, expect } from "vitest";
import { chunkText } from "../src/state/chunker.js";

const norm = (s: string) => s.replace(/\s+/g, " ").trim();

// Strips the overlap each chunk carried in from its predecessor and
// rejoins. If the chunker ever drops or duplicates content, this is what
// catches it.
function reassemble(chunks: string[], overlapChars: number): string {
  let out = chunks[0];
  for (let i = 1; i < chunks.length; i++) {
    const prev = chunks[i - 1];
    // The carry is word-aligned, so its length is <= overlapChars but not
    // exactly it. Find the longest suffix of prev that opens chunks[i].
    let carry = 0;
    for (let len = Math.min(overlapChars, prev.length); len > 0; len--) {
      if (chunks[i].startsWith(prev.slice(prev.length - len))) {
        carry = len;
        break;
      }
    }
    expect(carry).toBeGreaterThan(0);
    out += "\n\n" + chunks[i].slice(carry);
  }
  return out;
}

describe("chunkText", () => {
  it("returns short text unchanged, as a single chunk", () => {
    const text = "One short memory about a single topic.";
    expect(chunkText(text)).toEqual([text]);
    // Exactly at the boundary is still "fits".
    const exact = "x".repeat(1200);
    expect(chunkText(exact, 1200)).toEqual([exact]);
  });

  it("forces a boundary at an ALL-CAPS heading and keeps it glued to its body", () => {
    const intro = "intro ".repeat(50).trim(); // 299 chars, packs first
    const body = "The daemon returns 500 on the smart-search graph read path.";
    const text = [intro, "SYMPTOM\n" + body, "trailing paragraph here"].join(
      "\n\n",
    );

    const chunks = chunkText(text, 200, 20);

    const withHeading = chunks.filter((c) => c.includes("SYMPTOM"));
    expect(withHeading).toHaveLength(1);
    const chunk = withHeading[0];
    // The heading opens the chunk (only the carried overlap precedes it)…
    expect(chunk.indexOf("SYMPTOM")).toBeLessThanOrEqual(20 + 2);
    // …and its body did not get pushed into the next chunk.
    expect(chunk).toContain(body);
  });

  it("hard-splits an oversized single paragraph at whitespace, never mid-word", () => {
    const words = Array.from({ length: 800 }, (_, i) => `word${i}`);
    const paragraph = words.join(" "); // ~5.5k chars, one block, no newlines
    expect(paragraph.length).toBeGreaterThan(5000);

    const chunks = chunkText(paragraph, 1200, 150);

    expect(chunks.length).toBeGreaterThan(1);
    for (const c of chunks) expect(c.length).toBeLessThanOrEqual(1200);

    // Every token in every chunk is a whole source word.
    const source = new Set(words);
    for (const c of chunks) {
      for (const token of c.split(/\s+/)) {
        expect(source.has(token)).toBe(true);
      }
    }
  });

  it("loses no content: concatenation minus overlap reproduces the source", () => {
    const sections = [
      "CONTEXT",
      "The retrieval path fuses BM25 and vector legs with reciprocal rank fusion.",
      "ROOT CAUSE",
      "RRF discards relevance magnitude entirely; only rank position survives.",
      "FIX",
      "Normalise each leg and fuse on magnitude instead. " + "detail ".repeat(200),
      "VALIDATION",
      "Measured against a live corpus with an explicit acceptance gate.",
    ];
    const text = sections.join("\n\n");

    const chunks = chunkText(text, 400, 60);

    expect(chunks.length).toBeGreaterThan(1);
    expect(norm(reassemble(chunks, 60))).toBe(norm(text));
  });

  it("is idempotent — re-chunking a chunk yields that chunk", () => {
    const text = ("paragraph body text. ".repeat(30) + "\n\n").repeat(12);
    const chunks = chunkText(text, 1200, 150);
    expect(chunks.length).toBeGreaterThan(1);
    for (const c of chunks) {
      expect(c.length).toBeLessThanOrEqual(1200);
      expect(chunkText(c, 1200, 150)).toEqual([c]);
    }
  });
});
