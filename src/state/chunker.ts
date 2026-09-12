// Pure text chunker for the vector leg. No imports on purpose — it is
// arithmetic over a string and nothing else.
//
// WHY: a saved memory is embedded as ONE vector today. A 7.8k-char memory
// spanning a dozen unrelated topics averages out to a point near the
// corpus centroid and matches no query strongly. Splitting it into
// section-sized pieces gives each idea its own vector.
//
// The binding constraint on chunk size is semantic dilution, not model
// context — 1200 chars is roughly 300 tokens, about one section of a
// written memory, which is the granularity at which one idea is still
// one vector. (EMBED_MAX_CHARS is 16k; it never fires on a chunk.)

// An ALL-CAPS-ish line on its own is a section label (SYMPTOM, ROOT
// CAUSE, FIX/VALIDATION). Bounded at 60 chars so a shouted sentence
// doesn't qualify.
const HEADING = /^[A-Z0-9][A-Z0-9 _/&().:-]{2,60}$/;

// Cuts `block` into pieces of at most `max` chars, preferring a
// paragraph-internal newline, then a sentence end, then any whitespace.
// Falls back to a hard cut only when a single run has no break at all.
function hardSplit(block: string, max: number): string[] {
  const out: string[] = [];
  let rest = block;
  while (rest.length > max) {
    const window = rest.slice(0, max);
    const floor = max * 0.5; // refuse a break so early it wastes the chunk
    let cut = window.lastIndexOf("\n");
    if (cut < floor) {
      const dot = window.lastIndexOf(". ");
      if (dot >= floor) cut = dot + 1; // keep the period on the left piece
    }
    if (cut < floor) cut = window.lastIndexOf(" ");
    if (cut <= 0) cut = max;
    out.push(rest.slice(0, cut).trim());
    rest = rest.slice(cut).trim();
  }
  if (rest) out.push(rest);
  return out;
}

/**
 * Split `text` into overlapping chunks of at most `maxChars`.
 *
 * Returns `[text]` unchanged when the input already fits, so the common
 * path (short memories, every observation) is byte-identical to not
 * chunking at all. Idempotent: every returned chunk is <= maxChars, so
 * re-chunking a chunk yields that chunk.
 */
export function chunkText(
  text: string,
  maxChars = 1200,
  overlapChars = 150,
): string[] {
  if (text.length <= maxChars) return [text];

  // Every unit is sized so that unit + carried overlap + separator still
  // fits in maxChars. That bound is what makes the function idempotent.
  const unitMax = Math.max(1, maxChars - overlapChars - 2);

  const units: Array<{ text: string; forceBoundary: boolean }> = [];
  for (const raw of text.split(/\n\s*\n/)) {
    const block = raw.trim();
    if (!block) continue;
    // A section label forces a boundary BEFORE its block and stays glued
    // to the body that follows it — otherwise "SYMPTOM" trails the end of
    // the previous section and its symptom text starts a new chunk.
    const forceBoundary = HEADING.test(block.split("\n", 1)[0].trim());
    const pieces = block.length <= unitMax ? [block] : hardSplit(block, unitMax);
    pieces.forEach((p, i) =>
      units.push({ text: p, forceBoundary: forceBoundary && i === 0 }),
    );
  }

  const chunks: string[] = [];
  let cur = "";
  const flush = () => {
    if (cur) chunks.push(cur);
    cur = "";
  };
  // Trailing slice of the previous chunk, carried into the next one so a
  // sentence spanning a boundary is still embedded intact somewhere.
  const carry = (): string => {
    const prev = chunks[chunks.length - 1];
    if (!prev || overlapChars <= 0) return "";
    if (prev.length <= overlapChars) return prev;
    let start = prev.length - overlapChars;
    // Snap forward to a word boundary. A raw character slice opens the
    // carried text with half a word, which is pure noise to an embedder.
    const space = prev.indexOf(" ", start);
    const newline = prev.indexOf("\n", start);
    const boundary = Math.min(
      space === -1 ? Infinity : space,
      newline === -1 ? Infinity : newline,
    );
    if (boundary < prev.length - 1) start = boundary + 1;
    return prev.slice(start);
  };

  for (const unit of units) {
    if (
      cur &&
      (unit.forceBoundary || cur.length + 2 + unit.text.length > maxChars)
    ) {
      flush();
    }
    if (!cur && chunks.length > 0) cur = carry();
    cur = cur ? cur + "\n\n" + unit.text : unit.text;
  }
  flush();

  return chunks.length > 0 ? chunks : [text];
}
