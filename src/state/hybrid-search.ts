import { SearchIndex } from "./search-index.js";
import { VectorIndex } from "./vector-index.js";
import type {
  EmbeddingProvider,
  HybridSearchResult,
  CompressedObservation,
  Memory,
  QueryExpansion,
} from "../types.js";
import {
  memoryToObservation,
  isLatestEligible,
  MEMORY_SESSION,
} from "./memory-utils.js";
import { isNonLatestFilterEnabled } from "../config.js";
import { probeCapture } from "./probe-capture.js";
import { getCounters } from "../telemetry/setup.js";
import type { StateKV } from "./kv.js";
import { KV } from "./schema.js";
import {
  GraphRetrieval,
  type GraphRetrievalResult,
} from "../functions/graph-retrieval.js";
import { extractEntitiesFromQuery } from "../functions/query-expansion.js";
import { rerank } from "./reranker.js";

const RRF_K = 60;
const MAX_PER_SESSION = 3;
// A1: how much deeper than the page to ask diversification for when the
// non-latest filter is on, so rejected rows can be refilled from
// lower-ranked eligible candidates. Only the rows the page actually needs
// are ever resolved, so this bounds the candidate LIST, not the KV reads.
const CANDIDATE_OVERFETCH = 3;
// The fixed head both the reranker and the Probe 0 `head` capture see. Shared
// by the plain and the expanded path so a probe run compares two heads of the
// same size.
const RERANK_WINDOW = 20;

function envNumber(key: string, fallback: number): number {
  const raw = process.env[key];
  if (!raw) return fallback;
  const n = Number(raw);
  return Number.isFinite(n) && n > 0 ? n : fallback;
}

export class HybridSearch {
  private graphRetrieval: GraphRetrieval;

  constructor(
    private bm25: SearchIndex,
    private vector: VectorIndex | null,
    private embeddingProvider: EmbeddingProvider | null,
    private kv: StateKV,
    private bm25Weight = 0.4,
    private vectorWeight = 0.6,
    private graphWeight = 0.3,
    private rerankEnabled = process.env.RERANK_ENABLED === "true",
    // RRF's rank constant. Larger K flattens the gap between adjacent
    // ranks further; smaller K sharpens it. Exposed so the flattening
    // can be tuned without a rebuild.
    private rrfK = envNumber("AGENTMEMORY_RRF_K", RRF_K),
    // Per-session cap in diversifyBySession. Every saved memory shares
    // the synthetic sessionId "memory", so on a memory-heavy corpus this
    // cap is what bounds how many memories a single page can contain.
    private maxPerSession = envNumber(
      "AGENTMEMORY_MAX_PER_SESSION",
      MAX_PER_SESSION,
    ),
  ) {
    this.graphRetrieval = new GraphRetrieval(kv);
  }

  async search(query: string, limit = 20): Promise<HybridSearchResult[]> {
    return this.tripleStreamSearch(query, limit);
  }

  async searchWithExpansion(
    query: string,
    limit: number,
    expansion: QueryExpansion,
  ): Promise<HybridSearchResult[]> {
    const allQueries = [
      query,
      ...expansion.reformulations,
      ...expansion.temporalConcretizations,
    ];

    const allEntities = [
      ...expansion.entityExtractions,
      ...extractEntitiesFromQuery(query),
    ];

    // An expansion that produced nothing must not change the result ORDER.
    // The merge below re-sorts by combinedScore, which discards the
    // session-diversified ordering `search()` returns; taking the plain path
    // keeps a failed or empty expansion byte-identical to no expansion at all
    // rather than silently reshuffling every page.
    if (allQueries.length === 1 && expansion.entityExtractions.length === 0) {
      return this.tripleStreamSearch(query, limit);
    }

    const resultSets = await Promise.all(
      allQueries.map((q) => this.tripleStreamSearch(q, limit, allEntities)),
    );

    const merged = new Map<string, HybridSearchResult>();
    for (const results of resultSets) {
      for (const r of results) {
        const existing = merged.get(r.observation.id);
        if (!existing || r.combinedScore > existing.combinedScore) {
          merged.set(r.observation.id, r);
        }
      }
    }

    const ranked = Array.from(merged.values()).sort(
      (a, b) => b.combinedScore - a.combinedScore,
    );

    // Probe 0 capture, expansion arm. The two capture points inside
    // `tripleStreamSearch` fire once per sub-query and describe individual
    // legs of the merge, so without these lines the probe would score the
    // ORIGINAL query's unexpanded pipeline and report no change no matter
    // what expansion did. Emitted last on purpose: the scorer keeps the last
    // record per (query, point), so the merged view wins over the leg view.
    //
    // `head` is the top RERANK_WINDOW rather than the top `limit`, matching
    // the unexpanded arm exactly - otherwise a probe run at limit=10 would
    // compare a 10-row head against a 20-row one and the difference would be
    // the instrument, not the change.
    probeCapture(query, "combined", ranked.map((r) => r.observation.id));
    probeCapture(
      query,
      "head",
      ranked.slice(0, RERANK_WINDOW).map((r) => r.observation.id),
    );

    return ranked.slice(0, limit);
  }

  private async tripleStreamSearch(
    query: string,
    limit: number,
    entityHints?: string[],
  ): Promise<HybridSearchResult[]> {
    // Each leg must be fetched to a DEPTH FLOOR, not to a multiple of the
    // caller's page size. With `limit * 2`, a document that is rank ~10 in
    // BM25 but rank ~70 in the vector leg falls outside the vector fetch at
    // small limits and forfeits its vector term entirely. Under RRF that is
    // unrecoverable: the vector term is worth up to 0.6/61 = 0.0098 while the
    // whole BM25 term tops out at 0.4/61 = 0.0066, so ANY doc present in both
    // legs outranks ANY doc present in one. Curated memories are exactly the
    // long, semantically diffuse documents that sit deep in the vector leg, so
    // they were systematically excluded at default limits — and results were
    // non-monotonic in `limit` (absent at limit 15, rank 5 at limit 60).
    //
    // Raising the floor is nearly free: VectorIndex.search scans every vector
    // regardless of `limit` (the argument only sizes the top-K heap), and
    // SearchIndex.search already walks every matching posting. Only the final
    // sort grows.
    //
    // This is a CANDIDATE POOL size, not a page size — the caller still gets
    // exactly `limit` rows. Set AGENTMEMORY_RETRIEVAL_DEPTH=0 to disable the
    // floor and restore the old `limit * 2` fetch. Not envNumber(), which
    // rejects 0 and would silently fall back to the default.
    const rawDepth = process.env.AGENTMEMORY_RETRIEVAL_DEPTH;
    const parsedDepth = rawDepth === undefined ? NaN : Number(rawDepth);
    const depthFloor =
      Number.isFinite(parsedDepth) && parsedDepth >= 0 ? parsedDepth : 100;
    const depth = Math.max(limit * 2, depthFloor);
    const bm25Results = this.bm25.search(query, depth);

    let vectorResults: Array<{
      obsId: string;
      sessionId: string;
      score: number;
    }> = [];
    let queryEmbedding: Float32Array | null = null;

    if (this.vector && this.embeddingProvider && this.vector.size > 0) {
      try {
        queryEmbedding = await this.embeddingProvider.embed(query);
        vectorResults = this.vector.search(queryEmbedding, depth);
      } catch {
        // fall through to BM25-only
      }
    }

    const entities =
      entityHints && entityHints.length > 0
        ? entityHints
        : extractEntitiesFromQuery(query);
    let graphResults: GraphRetrievalResult[] = [];
    if (entities.length > 0) {
      try {
        graphResults = await this.graphRetrieval.searchByEntities(
          entities,
          2,
          limit,
        );
      } catch {
        // graph search is best-effort
      }
    }

    const topVectorObs = vectorResults.slice(0, 5).map((r) => r.obsId);
    if (topVectorObs.length > 0) {
      try {
        const expansionResults =
          await this.graphRetrieval.expandFromChunks(topVectorObs, 1, 5);
        graphResults = [...graphResults, ...expansionResults];
      } catch {
        // expansion is best-effort
      }
    }

    const scores = new Map<
      string,
      {
        bm25Rank: number;
        vectorRank: number;
        graphRank: number;
        sessionId: string;
        bm25Score: number;
        vectorScore: number;
        graphScore: number;
        graphContext?: string;
      }
    >();

    bm25Results.forEach((r, i) => {
      scores.set(r.obsId, {
        bm25Rank: i + 1,
        vectorRank: Infinity,
        graphRank: Infinity,
        sessionId: r.sessionId,
        bm25Score: r.score,
        vectorScore: 0,
        graphScore: 0,
      });
    });

    vectorResults.forEach((r, i) => {
      const existing = scores.get(r.obsId);
      if (existing) {
        existing.vectorRank = i + 1;
        existing.vectorScore = r.score;
      } else {
        scores.set(r.obsId, {
          bm25Rank: Infinity,
          vectorRank: i + 1,
          graphRank: Infinity,
          sessionId: r.sessionId,
          bm25Score: 0,
          vectorScore: r.score,
          graphScore: 0,
        });
      }
    });

    graphResults.forEach((r, i) => {
      const existing = scores.get(r.obsId);
      if (existing) {
        existing.graphRank = Math.min(existing.graphRank, i + 1);
        existing.graphScore = Math.max(existing.graphScore, r.score);
        if (r.graphContext && !existing.graphContext) {
          existing.graphContext = r.graphContext;
        }
      } else {
        scores.set(r.obsId, {
          bm25Rank: Infinity,
          vectorRank: Infinity,
          graphRank: i + 1,
          sessionId: r.sessionId,
          bm25Score: 0,
          vectorScore: 0,
          graphScore: r.score,
          graphContext: r.graphContext,
        });
      }
    });

    const hasVector = vectorResults.length > 0;
    const hasGraph = graphResults.length > 0;

    let effectiveBm25W = this.bm25Weight;
    let effectiveVectorW = hasVector ? this.vectorWeight : 0;
    let effectiveGraphW = hasGraph ? this.graphWeight : 0;

    const totalW = effectiveBm25W + effectiveVectorW + effectiveGraphW;
    if (totalW > 0) {
      effectiveBm25W /= totalW;
      effectiveVectorW /= totalW;
      effectiveGraphW /= totalW;
    }

    // Fusion mode. `rrf` (default) fuses on rank alone — scale-free, but
    // it discards relevance magnitude entirely: a BM25 score of 74.9 and
    // one of 42.2 differ only by their rank positions once fused. `norm`
    // fuses on normalised magnitude instead. The asymmetry is deliberate:
    // BM25 has no absolute scale so it needs min-max over the result set,
    // while cosine already lives in [-1, 1] and maps absolutely — min-max
    // there would promote the best of a uniformly terrible vector set to
    // 1.0 and let the vector leg dominate off-topic queries.
    const normFusion = process.env.AGENTMEMORY_FUSION === "norm";
    let bm25Min = 0;
    let bm25Range = 1;
    if (normFusion && bm25Results.length > 0) {
      let lo = Infinity;
      let hi = -Infinity;
      for (const r of bm25Results) {
        if (r.score < lo) lo = r.score;
        if (r.score > hi) hi = r.score;
      }
      bm25Min = lo;
      bm25Range = hi - lo || 1;
    }

    // Multiplicative, so it needs no recalibration when RRF_K, the leg
    // weights, or the fusion mode change. The id prefix is the primary
    // predicate: it needs no KV read and keeps working if sessionIds
    // later gets populated by consolidate/flow-compress.
    const layerBoost = envNumber("AGENTMEMORY_MEMORY_LAYER_BOOST", 1.0);
    const boostFor = (obsId: string, sessionId: string): number =>
      layerBoost !== 1 &&
      (obsId.startsWith("mem_") || sessionId === MEMORY_SESSION)
        ? layerBoost
        : 1;

    const combined = Array.from(scores.entries()).map(([obsId, s]) => {
      // The graph leg has no absolute scale of its own, so it keeps its
      // rank-reciprocal contribution in both modes. effectiveGraphW is 0
      // whenever the leg returns nothing (and it is frozen off in this
      // deployment), so in practice this term only exists in A-mode.
      const graphTerm = effectiveGraphW * (1 / (this.rrfK + s.graphRank));
      const fused = normFusion
        ? effectiveBm25W *
            (s.bm25Rank === Infinity ? 0 : (s.bm25Score - bm25Min) / bm25Range) +
          effectiveVectorW *
            (s.vectorRank === Infinity ? 0 : (s.vectorScore + 1) / 2) +
          graphTerm
        : effectiveBm25W * (1 / (this.rrfK + s.bm25Rank)) +
          effectiveVectorW * (1 / (this.rrfK + s.vectorRank)) +
          graphTerm;
      return {
        obsId,
        sessionId: s.sessionId,
        bm25Score: s.bm25Score,
        vectorScore: s.vectorScore,
        graphScore: s.graphScore,
        graphContext: s.graphContext,
        combinedScore: fused * boostFor(obsId, s.sessionId),
      };
    });

    combined.sort((a, b) => b.combinedScore - a.combinedScore);
    // Probe 0 capture point 1: the fused pool in rank order. Must be taken
    // AFTER this sort - the unsorted array carries no meaningful ranks - and
    // it is not reachable from the public API, because raising `limit` also
    // raises retrievalDepth below and therefore changes diversification, so a
    // caller cannot reconstruct this pool.
    probeCapture(query, "combined", combined.map((c) => c.obsId));

    const retrievalDepth = Math.max(limit, 20);
    const rerankWindow = RERANK_WINDOW;
    // A1: with the filter on, ask diversification for headroom so
    // rejections refill from lower-ranked eligible candidates instead of
    // silently underfilling the page. Enlarging the limit returns a
    // SUPERSET whose top-`retrievalDepth` is identical: loop 1 extends the
    // same cap-eligible prefix, loop 2 fills in score order so the extra
    // rows are strictly lower-scored, and the trailing sort puts them
    // below. With the filter off the argument is unchanged, so the
    // diversification output is byte-identical to pre-A1.
    const filterNonLatest = isNonLatestFilterEnabled();
    const diversified = this.diversifyBySession(
      combined,
      filterNonLatest ? retrievalDepth * CANDIDATE_OVERFETCH : retrievalDepth,
    );
    const enriched = await this.enrichResults(
      diversified,
      retrievalDepth,
      filterNonLatest,
    );

    // Probe 0 capture point 2: what a reranker would actually receive. Taken
    // unconditionally rather than inside the rerankEnabled branch below,
    // because the probe's whole question is whether turning reranking ON is
    // worth it - capturing only when it is already on would answer nothing.
    probeCapture(
      query,
      "head",
      enriched.slice(0, rerankWindow).map((e) => e.observation.id),
    );

    if (this.rerankEnabled && enriched.length > 1) {
      try {
        const head = enriched.slice(0, rerankWindow);
        const tail = enriched.slice(rerankWindow);
        const reranked = await rerank(query, head, rerankWindow);
        return reranked.concat(tail).slice(0, limit);
      } catch {
        return enriched.slice(0, limit);
      }
    }

    return enriched.slice(0, limit);
  }

  private diversifyBySession(
    results: Array<{
      obsId: string;
      sessionId: string;
      bm25Score: number;
      vectorScore: number;
      graphScore: number;
      combinedScore: number;
      graphContext?: string;
    }>,
    limit: number,
    maxPerSession = this.maxPerSession,
  ): typeof results {
    // Saved memories all share the synthetic sessionId "memory", so the
    // per-session cap treats an entire curated corpus as one chatty
    // session and hands it `maxPerSession` slots total. Exempting it
    // restores per-document competition for memories while leaving the
    // cap doing its real job (bounding one noisy coding session).
    const exemptMemorySession =
      process.env.AGENTMEMORY_DIVERSITY_EXEMPT_MEMORY_SESSION === "true";

    const selected: typeof results = [];
    const sessionCounts = new Map<string, number>();

    for (const r of results) {
      if (!(exemptMemorySession && r.sessionId === MEMORY_SESSION)) {
        const count = sessionCounts.get(r.sessionId) || 0;
        if (count >= maxPerSession) continue;
        sessionCounts.set(r.sessionId, count + 1);
      }
      selected.push(r);
      if (selected.length >= limit) break;
    }

    if (selected.length < limit) {
      for (const r of results) {
        if (selected.length >= limit) break;
        if (!selected.some(s => s.obsId === r.obsId)) {
          selected.push(r);
        }
      }
    }

    // The fallback fill appends cap-skipped rows AFTER lower-scored ones
    // that made the first pass, so `selected` is not monotonic in
    // combinedScore without this. Restores the descending-order contract
    // the callers (and test/hybrid-search.test.ts) assume.
    selected.sort((a, b) => b.combinedScore - a.combinedScore);

    return selected;
  }

  private async enrichResults(
    results: Array<{
      obsId: string;
      sessionId: string;
      bm25Score: number;
      vectorScore: number;
      graphScore: number;
      combinedScore: number;
      graphContext?: string;
    }>,
    limit: number,
    filterNonLatest = false,
  ): Promise<HybridSearchResult[]> {
    const enriched: HybridSearchResult[] = [];
    let cursor = 0;

    // A1: resolve in rank order, only as far as the page needs. The first
    // batch is exactly `limit` rows - the same Promise.all shape and the
    // same number of KV reads as pre-A1 - and with the filter off the loop
    // breaks after it, so the disabled path costs nothing extra. Later
    // batches are sized to the shortfall and only occur when rows were
    // actually rejected.
    while (cursor < results.length && enriched.length < limit) {
      const batch = results.slice(cursor, cursor + (limit - enriched.length));
      // Advance past REJECTED rows too, not just accepted ones, or a pool
      // of entirely non-latest candidates re-resolves its own head forever.
      cursor += batch.length;

      const observations = await Promise.all(
        batch.map((r) => this.resolveRow(r, filterNonLatest)),
      );

      for (let i = 0; i < batch.length; i++) {
        const obs = observations[i];
        if (!obs) continue;
        enriched.push({
          observation: obs,
          bm25Score: batch[i].bm25Score,
          vectorScore: batch[i].vectorScore,
          graphScore: batch[i].graphScore,
          combinedScore: batch[i].combinedScore,
          sessionId: batch[i].sessionId,
          graphContext: batch[i].graphContext,
        });
        if (enriched.length >= limit) break;
      }

      // Pre-A1 behaviour resolved exactly `limit` rows and simply dropped
      // any that failed to resolve, returning a short page. Preserve that
      // exactly when the filter is off - refill is an A1 behaviour.
      if (!filterNonLatest) break;
    }

    return enriched;
  }

  // Resolve one ranked row to the record the caller will see, applying the
  // A1 eligibility gate. This is the TERMINAL gate: nothing downstream can
  // re-admit a row, which is why the exclusion is safe here even though
  // `diversifyBySession`'s fill loop re-admits cap-skipped rows upstream.
  private async resolveRow(
    r: { obsId: string; sessionId: string },
    filterNonLatest: boolean,
  ): Promise<CompressedObservation | null> {
    const obs = await this.kv
      .get<CompressedObservation>(KV.observations(r.sessionId), r.obsId)
      .catch(() => null);
    if (obs) return obs;

    // Fallback: indexed entry may originate from mem::remember, which
    // writes to KV.memories with a synthetic sessionId ("memory" or the
    // memory's first associated session). Coerce the Memory record into
    // a CompressedObservation so search/recall surface saved memories.
    const mem = await this.kv
      .get<Memory>(KV.memories, r.obsId)
      .catch(() => null);
    if (!mem) return null;

    // The eligibility check MUST happen here, on the Memory record.
    // memoryToObservation does not carry isLatest through, so a check
    // placed after coercion has nothing left to test and passes everything.
    if (!isLatestEligible(mem)) {
      if (filterNonLatest) {
        getCounters().nonlatestFiltered.add(1);
        return null;
      }
      // Filter off: measure the live incidence of the bug instead. Goes
      // structurally to zero once A1 is enabled.
      getCounters().nonlatestLeaked.add(1);
    }
    return memoryToObservation(mem);
  }
}
