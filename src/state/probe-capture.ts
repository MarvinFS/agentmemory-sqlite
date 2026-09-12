import { appendFileSync } from "node:fs";
import { logger } from "../logger.js";

// Probe 0 instrumentation (plan step 1).
//
// The question the probe answers is whether the binding constraint on search
// quality is RETRIEVAL (the target never enters the candidate pool), SELECTION
// (it enters the pool but is lost to truncation, diversification or
// enrichment before a reranker would see it), or RANKING (it reaches the
// rerank window but sits too low). Only the third is something a cross-encoder
// can fix, so measuring this before building a reranker service is what stops
// us from deploying an instrument for the wrong problem.
//
// Neither capture point is reachable through the public API: raising `limit`
// also raises retrievalDepth, which changes diversification, so a caller
// cannot reconstruct the same pool from outside.
//
// Output goes to a FILE rather than the logger on purpose - `combined` runs to
// a couple of hundred ids per query and journald truncates long messages,
// which would silently corrupt exactly the tail of the ranking the probe cares
// about.
//
// Entirely inert unless AGENTMEMORY_PROBE_CAPTURE is set to a writable path.
// Failures are swallowed after one warning: a diagnostic must never be able to
// break a live search.

let warned = false;

export function probeCapture(
  query: string,
  point: "combined" | "head",
  ids: string[],
): void {
  const path = process.env.AGENTMEMORY_PROBE_CAPTURE;
  if (!path) return;
  try {
    appendFileSync(
      path,
      JSON.stringify({
        at: new Date().toISOString(),
        query,
        point,
        count: ids.length,
        ids,
      }) + "\n",
      "utf8",
    );
  } catch (err) {
    if (!warned) {
      warned = true;
      logger.warn("probe capture: write failed, continuing without it", {
        path,
        error: err instanceof Error ? err.message : String(err),
      });
    }
  }
}
