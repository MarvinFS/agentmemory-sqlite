import { registerWorker, TriggerAction } from "iii-sdk";
import {
  hydrateProcessEnvFromFile,
  loadConfig,
  getEnvVar,
  loadEmbeddingConfig,
  loadFallbackConfig,
  loadClaudeBridgeConfig,
  loadTeamConfig,
  loadSnapshotConfig,
  isGraphExtractionEnabled,
  isAutoCompressEnabled,
  isConsolidationEnabled,
  isContextInjectionEnabled,
  isDropStaleIndexEnabled,
  isInprocEngine,
  getSqlitePath,
} from "./config.js";
import { createInprocSdk, type InprocSdk } from "./engine/inproc/sdk.js";
import { SqliteVectorStore } from "./engine/inproc/vectors.js";
import { createIndexFill, registerIndexFillFunction } from "./functions/index-fill.js";
import { registerMaintenanceFunctions } from "./functions/maintenance.js";
import {
  createProvider,
  createFallbackProvider,
  createEmbeddingProvider,
  createImageEmbeddingProvider,
} from "./providers/index.js";
import { StateKV } from "./state/kv.js";
import { KV } from "./state/schema.js";
import {
  GRAPH_INDEX_NODE_CEILING,
  backfillGraphIndexes,
  graphIndexesReady,
  graphLegDisabled,
} from "./state/graph-indexes.js";
import { VectorIndex } from "./state/vector-index.js";
import { HybridSearch } from "./state/hybrid-search.js";
import { IndexPersistence } from "./state/index-persistence.js";
import { registerPrivacyFunction } from "./functions/privacy.js";
import { registerObserveFunction } from "./functions/observe.js";
import { registerImageQuotaCleanup } from "./functions/image-quota-cleanup.js";
import { registerVisionSearchFunctions } from "./functions/vision-search.js";
import { registerSlotsFunctions, isSlotsEnabled, isReflectEnabled } from "./functions/slots.js";
import { registerDiskSizeManager } from "./functions/disk-size-manager.js";
import { registerCompressFunction } from "./functions/compress.js";
import {
  registerSearchFunction,
  rebuildIndex,
  getSearchIndex,
  setVectorIndex,
  setEmbeddingProvider,
  setIndexPersistence,
  setInprocStores,
  rebuildBm25FromContent,
  pendingRebuildToken,
  markRebuildTokenDone,
} from "./functions/search.js";
import {
  reconcileIndexes,
  pendingReconcileToken,
  markReconcileTokenDone,
  isReconcilePublishEnabled,
} from "./functions/reconcile.js";
import { memoryToIndexDoc } from "./state/memory-utils.js";
import { registerContextFunction } from "./functions/context.js";
import { registerSummarizeFunction } from "./functions/summarize.js";
import { registerMigrateFunction } from "./functions/migrate.js";
import { registerFileIndexFunction } from "./functions/file-index.js";
import { registerConsolidateFunction } from "./functions/consolidate.js";
import { registerPatternsFunction } from "./functions/patterns.js";
import { registerRememberFunction } from "./functions/remember.js";
import { registerEvictFunction } from "./functions/evict.js";
import { registerRelationsFunction } from "./functions/relations.js";
import { registerTimelineFunction } from "./functions/timeline.js";
import { registerSmartSearchFunction } from "./functions/smart-search.js";
import { registerRecentSearchesSweepFunction } from "./functions/recent-searches-sweep.js";
import { registerProfileFunction } from "./functions/profile.js";
import { registerAutoForgetFunction } from "./functions/auto-forget.js";
import { registerExportImportFunction } from "./functions/export-import.js";
import { registerEnrichFunction } from "./functions/enrich.js";
import { registerClaudeBridgeFunction } from "./functions/claude-bridge.js";
import { registerGraphFunction } from "./functions/graph.js";
import { registerGraphImportFunction } from "./functions/graph-import.js";
import { registerConsolidationPipelineFunction } from "./functions/consolidation-pipeline.js";
import { registerTeamFunction } from "./functions/team.js";
import { registerGovernanceFunction } from "./functions/governance.js";
import { registerSnapshotFunction } from "./functions/snapshot.js";
import { registerActionsFunction } from "./functions/actions.js";
import { registerFrontierFunction } from "./functions/frontier.js";
import { registerLeasesFunction } from "./functions/leases.js";
import { registerRoutinesFunction } from "./functions/routines.js";
import { registerSignalsFunction } from "./functions/signals.js";
import { registerCheckpointsFunction } from "./functions/checkpoints.js";
import { registerFlowCompressFunction } from "./functions/flow-compress.js";
import { registerMeshFunction } from "./functions/mesh.js";
import { registerBranchAwareFunction } from "./functions/branch-aware.js";
import { registerSentinelsFunction } from "./functions/sentinels.js";
import { registerSketchesFunction } from "./functions/sketches.js";
import { registerCrystallizeFunction } from "./functions/crystallize.js";
import { registerDiagnosticsFunction } from "./functions/diagnostics.js";
import { registerFacetsFunction } from "./functions/facets.js";
import { registerVerifyFunction } from "./functions/verify.js";
import { registerCascadeFunction } from "./functions/cascade.js";
import { registerLessonsFunctions } from "./functions/lessons.js";
import { registerObsidianExportFunction } from "./functions/obsidian-export.js";
import { registerReflectFunctions } from "./functions/reflect.js";
import { registerWorkingMemoryFunctions } from "./functions/working-memory.js";
import { registerSkillExtractFunctions } from "./functions/skill-extract.js";
import { registerSlidingWindowFunction } from "./functions/sliding-window.js";
import {
  expandQuery,
  registerQueryExpansionFunction,
} from "./functions/query-expansion.js";
import { registerTemporalGraphFunctions } from "./functions/temporal-graph.js";
import { registerRetentionFunctions } from "./functions/retention.js";
import { registerCompressFileFunction } from "./functions/compress-file.js";
import { registerReplayFunctions } from "./functions/replay.js";
import { registerApiTriggers } from "./triggers/api.js";
import { registerEventTriggers } from "./triggers/events.js";
import { registerMcpEndpoints } from "./mcp/server.js";
import { getAllTools } from "./mcp/tools-registry.js";
import { startViewerServer } from "./viewer/server.js";
import { MetricsStore } from "./eval/metrics-store.js";
import { DedupMap } from "./functions/dedup.js";
import { registerHealthMonitor } from "./health/monitor.js";
import { initMetrics, OTEL_CONFIG } from "./telemetry/setup.js";
import { VERSION } from "./version.js";
import { bootLog } from "./logger.js";
import { mkdirSync, writeFileSync, unlinkSync } from "node:fs";
import { dirname, join } from "node:path";
import { homedir } from "node:os";

// #640 + #474: the worker process (this file) is spawned by iii-exec
// inside the engine. When `agentmemory stop` kills only the engine pid,
// this worker can survive (detached spawn, signal not propagated, or a
// wrapper script keeps it running) and reconnects to the next engine as
// a duplicate worker. Write the worker pid alongside iii.pid so
// `agentmemory stop` can reap us too.
function workerPidfilePath(): string {
  return join(homedir(), ".agentmemory", "worker.pid");
}
function writeWorkerPidfile(): void {
  try {
    const p = workerPidfilePath();
    mkdirSync(dirname(p), { recursive: true });
    writeFileSync(p, `${process.pid}\n`, { encoding: "utf-8" });
  } catch {
    // best-effort; stop still has the engine pidfile + port scan fallback
  }
}
function clearWorkerPidfile(): void {
  try {
    unlinkSync(workerPidfilePath());
  } catch {}
}

function hasGetMeter(
  sdk: unknown,
): sdk is { getMeter: (name: string) => unknown } {
  return (
    typeof sdk === "object" &&
    sdk !== null &&
    "getMeter" in sdk &&
    typeof (sdk as { getMeter?: unknown }).getMeter === "function"
  );
}

// Top-level safety net for iii-engine invocation timeouts (issue #204).
// Under sustained write load (e.g. Claude Code hooks across many
// projects) `state::set` can occasionally exceed the SDK's 30s timeout.
// We don't want one such timeout to terminate the long-lived memory
// service — the rejection is surfaced to the relevant call site via
// .catch() where it matters; everything else is logged-and-continued.
// Throttle logs to avoid spamming on bursts.
let lastUnhandledLogAt = 0;
process.on("unhandledRejection", (reason) => {
  const now = Date.now();
  if (now - lastUnhandledLogAt < 60_000) return;
  lastUnhandledLogAt = now;
  const r = reason as { code?: string; function_id?: string; message?: string };
  console.warn(
    `[agentmemory] unhandledRejection (suppressed):`,
    r?.code ? `${r.code} ${r.function_id ?? ""} ${r.message ?? ""}`.trim() : reason,
  );
});

async function main() {
  // Fold ~/.agentmemory/.env into process.env before anything reads config
  // or raw process.env. Only-if-unset, so real process.env still wins.
  hydrateProcessEnvFromFile();

  const config = loadConfig();
  const embeddingConfig = loadEmbeddingConfig();
  const fallbackConfig = loadFallbackConfig();

  const provider =
    fallbackConfig.providers.length > 0
      ? createFallbackProvider(config.provider, fallbackConfig)
      : createProvider(config.provider);

  const embeddingProvider = createEmbeddingProvider();
  const imageEmbeddingProvider = createImageEmbeddingProvider();

  const inproc = isInprocEngine();
  const sqlitePath = getSqlitePath();

  bootLog(`Starting worker v${VERSION}...`);
  bootLog(inproc ? `Engine: in-process (${sqlitePath})` : `Engine: ${config.engineUrl}`);
  bootLog(
    `Provider: ${config.provider.provider} (${config.provider.model})`,
  );
  if (embeddingProvider) {
    bootLog(
      `Embedding provider: ${embeddingProvider.name} (${embeddingProvider.dimensions} dims)`,
    );
  } else {
    bootLog(`Embedding provider: none (BM25-only mode)`);
  }
  if (imageEmbeddingProvider) {
    bootLog(
      `Image embedding provider: ${imageEmbeddingProvider.name} (${imageEmbeddingProvider.dimensions} dims) — vision-search active`,
    );
  }
  bootLog(
    `REST API: http://localhost:${config.restPort}/agentmemory/*`,
  );
  bootLog(`Streams: ws://localhost:${config.streamsPort}`);

  // In-process runtime: no engine, no worker bus. The shim binds the REST and
  // stream ports itself, so a bind failure has to be fatal here rather than
  // leaving a daemon up with no listeners.
  const inprocSdk: InprocSdk | null = inproc
    ? createInprocSdk({
        restPort: config.restPort,
        streamsPort: config.streamsPort,
        sqlitePath,
        maxBodyBytes: parseInt(getEnvVar("AGENTMEMORY_MAX_BODY_BYTES") || "", 10) || undefined,
      })
    : null;
  if (inprocSdk) await inprocSdk.listening();

  const sdk = inprocSdk ?? registerWorker(config.engineUrl, {
    workerName: "agentmemory",
    invocationTimeoutMs: 180000,
    otel: {
      serviceName: OTEL_CONFIG.serviceName,
      serviceVersion: OTEL_CONFIG.serviceVersion,
      metricsExportIntervalMs: OTEL_CONFIG.metricsExportIntervalMs,
    },
    // Explicit worker telemetry metadata. iii-sdk falls back to
    // auto-detection (cwd / package.json name / hostname) when this
    // is omitted, which produces inconsistent values per host —
    // `agentmemory`, `node`, `npm`, occasionally the user's home
    // directory basename. Pinning the value here gives every install
    // the same stable project identifier for downstream attribution
    // and grouping in the engine's metrics + traces output.
    telemetry: {
      project_name: "agentmemory",
      language: "node",
      framework: "iii-sdk",
    },
  });

  writeWorkerPidfile();

  const kv = new StateKV(sdk);
  const secret = getEnvVar("AGENTMEMORY_SECRET");
  const metricsStore = new MetricsStore(kv);
  const dedupMap = new DedupMap();

  const vectorIndex = embeddingProvider ? new VectorIndex() : null;

  // Inproc: vectors are rows in the same SQLite file as the state. Every map
  // mutation writes its rows first and every embedding completion commits
  // through the store's content-revalidating transaction (src/engine/inproc/vectors.ts).
  // The store exists even without an embedding provider so content deletes
  // still take their persisted vector rows with them.
  const vectorStore = inprocSdk ? new SqliteVectorStore(inprocSdk.store) : null;
  vectorIndex?.attachStore(vectorStore);
  setInprocStores(inprocSdk?.store ?? null, vectorStore);

  setVectorIndex(vectorIndex);
  setEmbeddingProvider(embeddingProvider);

  const meterAccessor = hasGetMeter(sdk)
    ? (sdk.getMeter.bind(sdk) as (name: string) => unknown)
    : undefined;

  initMetrics(meterAccessor as ((name: string) => import("@opentelemetry/api").Meter) | undefined);

  registerPrivacyFunction(sdk);
  registerObserveFunction(sdk, kv, dedupMap, config.maxObservationsPerSession);
  registerImageQuotaCleanup(sdk, kv);
  registerVisionSearchFunctions(sdk, kv, imageEmbeddingProvider);
  if (isSlotsEnabled()) {
    registerSlotsFunctions(sdk, kv);
  }
  registerDiskSizeManager(sdk, kv);
  registerCompressFunction(sdk, kv, provider, metricsStore);
  registerSearchFunction(sdk, kv);
  registerContextFunction(sdk, kv, config.tokenBudget);
  registerSummarizeFunction(sdk, kv, provider, metricsStore);
  registerMigrateFunction(sdk, kv);
  registerFileIndexFunction(sdk, kv);
  registerConsolidateFunction(sdk, kv, provider);
  registerPatternsFunction(sdk, kv);
  registerRememberFunction(sdk, kv);
  registerEvictFunction(sdk, kv);

  registerRelationsFunction(sdk, kv);
  registerTimelineFunction(sdk, kv);
  registerProfileFunction(sdk, kv);
  registerAutoForgetFunction(sdk, kv);
  registerExportImportFunction(sdk, kv);
  registerEnrichFunction(sdk, kv);

  const claudeBridgeConfig = loadClaudeBridgeConfig();
  if (claudeBridgeConfig.enabled) {
    registerClaudeBridgeFunction(sdk, kv, claudeBridgeConfig);
    bootLog(
      `Claude bridge: syncing to ${claudeBridgeConfig.memoryFilePath}`,
    );
  }

  registerGraphFunction(sdk, kv, provider);
  registerGraphImportFunction(sdk, kv);
  bootLog(
    `Knowledge graph: structural extraction on (LLM relations ${isGraphExtractionEnabled() ? "enabled" : "off"})`,
  );

  registerConsolidationPipelineFunction(sdk, kv, provider);
  bootLog(`Consolidation pipeline: registered (CONSOLIDATION_ENABLED=${isConsolidationEnabled() ? "true" : "false"})`);

  if (isAutoCompressEnabled()) {
    bootLog(
      `WARNING: AGENTMEMORY_AUTO_COMPRESS=true — every PostToolUse observation will be sent to your LLM provider for compression. This spends API tokens proportional to your session tool-use frequency. Set AGENTMEMORY_AUTO_COMPRESS=false to disable.`,
    );
  } else {
    bootLog(
      `Auto-compress: OFF (default) — observations indexed via zero-LLM synthetic compression. Set AGENTMEMORY_AUTO_COMPRESS=true to opt-in to LLM-powered summaries (uses your API key).`,
    );
  }

  if (isContextInjectionEnabled()) {
    bootLog(
      `WARNING: AGENTMEMORY_INJECT_CONTEXT=true — the PreToolUse and SessionStart hooks will inject up to ~4000 chars of memory context into every tool turn. On Claude Pro this burns session tokens proportional to your tool-call frequency. Set AGENTMEMORY_INJECT_CONTEXT=false to disable.`,
    );
  } else {
    bootLog(
      `Context injection: OFF (default) — hooks capture observations but do not inject context into Claude Code's conversation. Set AGENTMEMORY_INJECT_CONTEXT=true to opt-in (warning: expect your Claude Pro allocation to drain faster).`,
    );
  }

  const teamConfig = loadTeamConfig();
  if (teamConfig) {
    registerTeamFunction(sdk, kv, teamConfig);
    bootLog(
      `Team memory: ${teamConfig.teamId} (${teamConfig.mode})`,
    );
  }

  registerGovernanceFunction(sdk, kv);

  registerActionsFunction(sdk, kv);
  registerFrontierFunction(sdk, kv);
  registerLeasesFunction(sdk, kv);
  registerRoutinesFunction(sdk, kv);
  registerSignalsFunction(sdk, kv);
  registerCheckpointsFunction(sdk, kv);
  registerMeshFunction(sdk, kv, secret);
  registerBranchAwareFunction(sdk, kv);
  registerFlowCompressFunction(sdk, kv, provider);
  registerSentinelsFunction(sdk, kv);
  registerSketchesFunction(sdk, kv);
  registerCrystallizeFunction(sdk, kv, provider);
  registerDiagnosticsFunction(sdk, kv);
  registerFacetsFunction(sdk, kv);
  registerVerifyFunction(sdk, kv);
  registerLessonsFunctions(sdk, kv);
  registerObsidianExportFunction(sdk, kv);
  registerReflectFunctions(sdk, kv, provider);
  registerWorkingMemoryFunctions(sdk, kv, config.tokenBudget);
  registerSkillExtractFunctions(sdk, kv, provider);
  registerCascadeFunction(sdk, kv);

  registerSlidingWindowFunction(sdk, kv, provider);
  registerQueryExpansionFunction(sdk, provider);
  registerTemporalGraphFunctions(sdk, kv, provider);
  registerRetentionFunctions(sdk, kv);
  registerCompressFileFunction(sdk, kv, provider);
  registerReplayFunctions(sdk, kv);
  bootLog(
    `v0.6 advanced retrieval: sliding-window, query-expansion, temporal-graph, retention-scoring`,
  );
  bootLog(
    `Orchestration layer: actions, frontier, leases, routines, signals, checkpoints, flow-compress, mesh, branch-aware, sentinels, sketches, crystallize, diagnostics, facets`,
  );
  if (isSlotsEnabled()) {
    bootLog(
      `Slots: enabled (pinned editable memory). Reflect on Stop hook: ${isReflectEnabled() ? "on" : "off"}`,
    );
  }

  const snapshotConfig = loadSnapshotConfig();
  if (snapshotConfig.enabled) {
    registerSnapshotFunction(sdk, kv, snapshotConfig.dir);
    // The boot line promised "every <interval>s" but nothing ever fired
    // mem::snapshot-create. Drive it on a periodic timer (unref'd so it
    // never keeps the process alive), mirroring the auto-forget timer.
    // mem::snapshot-create serializes overlapping runs internally (git-lock
    // safety), so the timer can stay a simple fire-and-forget tick.
    const snapshotTimer = setInterval(() => {
      sdk
        .trigger({
          function_id: "mem::snapshot-create",
          payload: {},
          action: TriggerAction.Void(),
        })
        .catch(() => {});
    }, snapshotConfig.interval * 1000);
    snapshotTimer.unref();
    bootLog(
      `Git snapshots: ${snapshotConfig.dir} (every ${snapshotConfig.interval}s)`,
    );
  }

  const bm25Index = getSearchIndex();
  const graphWeight = parseFloat(getEnvVar("AGENTMEMORY_GRAPH_WEIGHT") || "0.3");
  const hybridSearch = new HybridSearch(
    bm25Index,
    vectorIndex,
    embeddingProvider,
    kv,
    embeddingConfig.bm25Weight,
    embeddingConfig.vectorWeight,
    graphWeight,
  );

  // `searchWithExpansion` existed as dead code since the fork point. Probe 0
  // measured why it is worth waking up: the two frozen-fixture targets that
  // never reached the candidate pool sit at combined rank 422 and 261 when the
  // pool is opened to depth 4000, so BOTH retrieval legs rank them past 100 and
  // neither a deeper pool nor a weight change would seat them in a 20-slot
  // head. The gap is vocabulary — the query and the memory share almost no
  // surface — which is the one thing reformulation addresses.
  //
  // OFF by default: it costs one LLM round-trip plus N extra searches per
  // query. Affordable here (single-user, a handful of searches a day) but not a
  // cost to impose on every deployment silently.
  const queryExpansionEnabled =
    getEnvVar("AGENTMEMORY_QUERY_EXPANSION") === "true";
  registerSmartSearchFunction(sdk, kv, async (query, limit) =>
    queryExpansionEnabled
      ? hybridSearch.searchWithExpansion(
          query,
          limit,
          await expandQuery(provider, query),
        )
      : hybridSearch.search(query, limit),
  );
  registerRecentSearchesSweepFunction(sdk, kv);

  registerApiTriggers(sdk, kv, secret, metricsStore, provider);
  registerEventTriggers(sdk, kv);
  registerMcpEndpoints(sdk, kv, secret, metricsStore);

  const healthMonitor = registerHealthMonitor(sdk, kv);

  // Inproc mode has no giant-string index persistence at all: vectors hydrate
  // from their rows below and BM25 is rebuilt from content. The
  // scheduleSave/markDirty/flushIndexSave hooks stay no-ops there.
  const indexPersistence = inproc
    ? null
    : new IndexPersistence(kv, bm25Index, vectorIndex);
  // Wire the persistence hook so delete paths can flush BM25/vector
  // index mutations to disk. Without this, an in-memory remove can be
  // lost across a hard process exit and the persisted snapshot
  // restores the deleted entry at next boot.
  setIndexPersistence(indexPersistence);

  const loaded = indexPersistence
    ? await indexPersistence.load().catch((err) => {
        console.warn(`[agentmemory] Failed to load persisted index:`, err);
        return null;
      })
    : null;
  if (loaded?.bm25 && loaded.bm25.size > 0) {
    bm25Index.restoreFrom(loaded.bm25);
    bootLog(
      `Loaded persisted BM25 index (${bm25Index.size} docs)`,
    );
  }
  if (vectorStore && vectorIndex) {
    const hydrated = vectorStore.hydrate(vectorIndex);
    if (hydrated > 0) bootLog(`Hydrated ${hydrated} vectors from SQLite`);
  }
  // The dimension guard below runs over whichever source supplied vectors:
  // the rows just hydrated (inproc) or the deserialised snapshot (iii).
  const vectorSource = vectorStore ? vectorIndex : loaded?.vector ?? null;
  if (vectorSource && vectorIndex && vectorSource.size > 0) {
    // Persisted vectors carry whatever dimension the provider had when
    // they were written. If the active provider declares a different
    // dimension — or if the on-disk index contains a mix of dimensions
    // (legacy indexes written before the live-API guard in this PR) —
    // restoring would silently corrupt search: cosineSimilarity returns
    // 0 on cross-dim pairs, so affected observations stop matching
    // anything and recall degrades without an error. Walk every stored
    // vector instead of trusting the first; refuse to load if anything
    // is off.
    const activeDim = embeddingProvider?.dimensions ?? 0;
    const { mismatches, seenDimensions } =
      activeDim > 0
        ? vectorSource.validateDimensions(activeDim)
        : { mismatches: [], seenDimensions: new Set<number>() };

    if (mismatches.length > 0) {
      const sample = mismatches
        .slice(0, 5)
        .map((m) => `${m.obsId} (dim=${m.dim})`)
        .join(", ");
      const distinct = Array.from(seenDimensions).sort((a, b) => a - b).join(", ");
      const dropStale = isDropStaleIndexEnabled();
      if (dropStale) {
        console.warn(
          `[agentmemory] Persisted vector index has ${mismatches.length} of ` +
            `${vectorSource.size} vectors with the wrong dimension. Active ` +
            `provider (${embeddingProvider?.name}) declares ${activeDim}; ` +
            `dimensions seen on disk: ${distinct}. ` +
            `AGENTMEMORY_DROP_STALE_INDEX=true is set — discarding the persisted ` +
            `vectors. Live observations will rebuild the index over time.`,
        );
        // Inproc: the rows ARE the index, so discarding means deleting them.
        if (vectorStore) vectorIndex.clear();
      } else {
        throw new Error(
          `[agentmemory] Refusing to start: persisted vector index has ` +
            `${mismatches.length} of ${vectorSource.size} vectors with the ` +
            `wrong dimension. Active provider (${embeddingProvider?.name}) ` +
            `declares ${activeDim}; dimensions seen on disk: ${distinct}. ` +
            `First mismatched obsIds: ${sample}. Loading would silently corrupt ` +
            `search (cross-dimension cosine returns 0). Choose one:\n` +
            `  - Re-embed the existing index against the new provider, then start.\n` +
            `  - Set AGENTMEMORY_DROP_STALE_INDEX=true to discard the persisted ` +
            `vectors and rebuild from live observations.\n` +
            `  - Switch the embedding provider back to the one that wrote the index.`,
        );
      }
    } else if (!vectorStore) {
      vectorIndex.restoreFrom(vectorSource);
      bootLog(
        `Loaded persisted vector index (${vectorIndex.size} vectors)`,
      );
    }
  }

  // Inproc: BM25 is not persisted, so it is rebuilt from the content rows on
  // every boot - one ordered walk, vectors untouched (they hydrated above).
  // rebuildIndex / reconcileIndexes and their env tokens are unreachable in
  // this mode: both would clear or re-serialise the vector index.
  if (inprocSdk) {
    const t0 = performance.now();
    const r = await rebuildBm25FromContent(inprocSdk.store.db);
    bootLog(
      `BM25 rebuilt from content: ${r.docs} docs of ${r.rows} rows in ` +
        `${Math.round(performance.now() - t0)} ms (read ${Math.round(r.readMs)} ms, ` +
        `index ${Math.round(r.indexMs)} ms${r.skipped ? `, ${r.skipped} rows skipped` : ""})`,
    );
  }

  // Inproc maintenance routes (plan steps 9.7, 10): mem::backup, and
  // mem::index-debug-legs when AGENTMEMORY_INDEX_DEBUG=1. Registered before
  // the gate opens so they exist the moment readyz answers 200.
  if (inprocSdk) registerMaintenanceFunctions(sdk, inprocSdk.store);

  // Inproc readiness (plan step 6): DB opened, a write probe round-tripped,
  // vectors hydrated, BM25 rebuilt -> readyz 200 and the 503 gate opens. The
  // probe uses the same key the health monitor writes. A failure here is
  // fatal on purpose: systemd restarts a daemon that cannot write its store.
  if (inprocSdk) {
    const stamp = Date.now();
    inprocSdk.store.set(KV.health, "_probe", { ts: stamp });
    const back = inprocSdk.store.get(KV.health, "_probe") as { ts?: number } | null;
    if (back?.ts !== stamp) throw new Error("readiness write probe did not round-trip");
    inprocSdk.setReady();
    bootLog("Ready: /agentmemory/readyz -> 200, routes open");
  }

  // Inproc: the vector fill/repair pass (plan step 5) closes the gap between
  // the content rows and the vectors table - missing or stale rows are
  // embedded, orphans pruned. Runs once at boot after readiness (not awaited:
  // a large backlog after an import must not hold up the listener) and hourly.
  if (inprocSdk && vectorStore && vectorIndex) {
    const fill = createIndexFill(inprocSdk.store, vectorStore, vectorIndex);
    registerIndexFillFunction(sdk, fill);
    void fill.run();
    setInterval(() => void fill.run(), 3_600_000).unref();
    bootLog("Vector fill pass: at boot and hourly (mem::index-fill-missing)");
  }

  // An explicit rebuild token in the environment that the store has not
  // recorded yet. Idempotent: markRebuildTokenDone below records it, so
  // the flag can live in a systemd drop-in permanently.
  const rebuildToken = inproc ? null : await pendingRebuildToken(kv).catch(() => null);

  // bm25Index.size === 0 alone misses the asymmetric case: a boot that
  // restored BM25 from disk but lost (or never had) the vector index
  // degrades to BM25-only PERMANENTLY, because nothing ever re-triggers
  // the rebuild. If a provider exists, an empty vector index is a
  // rebuild trigger in its own right.
  const vectorEmpty =
    vectorIndex !== null &&
    embeddingProvider !== null &&
    vectorIndex.size === 0;
  const needsRebuild =
    !inproc && (bm25Index.size === 0 || vectorEmpty || rebuildToken !== null);

  // A2 baseline reconcile takes precedence over the ordinary rebuild path:
  // it does the same repopulation, but offline, fail-closed, and published
  // as one retained-predecessor checkpoint. Running rebuildIndex as well
  // would clear the live singletons the reconcile is about to replace.
  // Inproc ignores the reconcile token: reconcileIndexes serialises and
  // checkpoints through IndexPersistence, which does not exist here (plan
  // step 4).
  const reconcileToken = indexPersistence
    ? await pendingReconcileToken(kv).catch(() => null)
    : null;
  if (reconcileToken && indexPersistence) {
    const publish = isReconcilePublishEnabled();
    bootLog(
      `Index reconcile requested by token "${reconcileToken}" (${publish ? "PUBLISH" : "dry run"})`,
    );
    void reconcileIndexes(kv, indexPersistence, { publish })
      .then(async (report) => {
        bootLog(
          `Index reconcile ${publish ? "published" : "dry run"}: ` +
            `bm25 ${report.bm25.before} -> ${report.bm25.after} ` +
            `(+${report.bm25.added} / -${report.bm25.dropped}), ` +
            `vectors ${report.vectors.before} -> ${report.vectors.after}, ` +
            `${Math.round(report.durationMs / 1000)}s`,
        );
        if (report.published) await markReconcileTokenDone(kv, reconcileToken);
      })
      .catch((err) => {
        console.warn(
          `[agentmemory] Index reconcile FAILED - nothing published:`,
          err,
        );
      });
  } else if (needsRebuild && !inproc) {
    if (rebuildToken) {
      bootLog(`Index rebuild requested by token "${rebuildToken}"`);
    } else if (vectorEmpty && bm25Index.size > 0) {
      bootLog(
        `Vector index empty with an active embedding provider (${embeddingProvider?.name}) — rebuilding`,
      );
    }
    // Fire-and-forget. rebuildIndex iterates every observation across
    // every session and AWAITS an embedding-provider call per record.
    // On a large corpus + rate-limited embedding endpoint that can
    // take HOURS; awaiting it here blocks every subsequent boot step
    // (including startViewerServer below, leaving the viewer port
    // unbound for the duration). The index lazily fills in over time
    // and search degrades gracefully — partial coverage > no viewer
    // for hours. Errors still surface via the inner .catch.
    void rebuildIndex(kv)
      .then(async (indexCount) => {
        if (indexCount > 0) {
          bootLog(`Search index rebuilt: ${indexCount} entries`);
          indexPersistence?.scheduleSave();
        }
        // Only after the rebuild actually resolves — a crashed or
        // aborted rebuild must run again on the next boot.
        if (rebuildToken) await markRebuildTokenDone(kv, rebuildToken);
      })
      .catch((err) => {
        console.warn(`[agentmemory] Failed to rebuild search index:`, err);
      });
  } else if (!inproc) {
    // Backfill memories into BM25 for users upgrading from <0.9.5: prior
    // versions of mem::remember never indexed memories, so the persisted
    // BM25 covers observations only and `memory_smart_search` returns
    // empty for everything saved via memory_save (#257). Walk KV.memories
    // and add the ones missing from the restored index. Idempotent on
    // re-runs because SearchIndex.has() short-circuits already-indexed
    // ids.
    try {
      const memories = await kv.list<import("./types.js").Memory>(KV.memories);
      let backfilled = 0;
      for (const memory of memories) {
        if (memory.isLatest === false) continue;
        if (!memory.title || !memory.content) continue;
        if (bm25Index.has(memory.id)) continue;
        bm25Index.add(memoryToIndexDoc(memory));
        backfilled++;
      }
      if (backfilled > 0) {
        bootLog(
          `Backfilled ${backfilled} memories into BM25 (legacy index gap)`,
        );
        indexPersistence?.scheduleSave();
      }
    } catch (err) {
      console.warn(
        `[agentmemory] Failed to backfill memories into BM25:`,
        err,
      );
    }
  }

  // Backfill the graph read side-indexes for corpora that predate them.
  // Mirrors the BM25 memories backfill above: one-time, gated on the
  // snapshot's recorded node count so we never enumerate a corpus large
  // enough to starve the worker heartbeat.
  //
  // graph-read-fix local delta (B-mode boot-skip): when the leg is off
  // this block is skipped ENTIRELY — it makes the only graph-scope reads
  // at boot (the kv.list arming enumeration), and B-mode must issue zero.
  // Skipping is safe: with the marker absent the readers fail-closed
  // (graphReadable === false) rather than enumerating.
  if (!graphLegDisabled()) {
    try {
      if (!(await graphIndexesReady(kv))) {
        const graphSnap = await kv.get<import("./types.js").GraphSnapshot>(
          KV.graphSnapshot,
          "current",
        );
        const totalNodes = graphSnap?.stats?.totalNodes ?? 0;
        if (graphSnap && totalNodes > 0 && totalNodes <= GRAPH_INDEX_NODE_CEILING) {
          const [graphNodes, graphEdges] = await Promise.all([
            kv.list<import("./types.js").GraphNode>(KV.graphNodes),
            kv.list<import("./types.js").GraphEdge>(KV.graphEdges),
          ]);
          await backfillGraphIndexes(
            kv,
            graphNodes.filter((n) => !n.stale),
            graphEdges.filter((e) => !e.stale),
          );
          bootLog(`Backfilled graph read indexes (${totalNodes} nodes)`);
        }
      }
    } catch (err) {
      console.warn(`[agentmemory] Failed to backfill graph indexes:`, err);
    }
  }

  // Ready / Endpoints lines are emitted via `bootLog` so they're
  // buffered in quiet mode and printed verbatim under --verbose. The
  // CLI surfaces a compact summary when it sees the worker reach
  // ready state.
  bootLog(
    `Ready. ${embeddingProvider ? "Triple-stream (BM25+Vector+Graph)" : "BM25+Graph"} search active.`,
  );
  bootLog(
    `REST API: 131 endpoints at http://localhost:${config.restPort}/agentmemory/*`,
  );
  bootLog(
    `MCP surface (opt-in via \`npx @agentmemory/mcp\`): ${getAllTools().length} tools · 6 resources · 3 prompts`,
  );

  const viewerPort = config.restPort + 2;
  const viewerServer = startViewerServer(
    viewerPort,
    kv,
    sdk,
    secret,
    config.restPort,
  );

  const autoForgetIntervalMs = parseInt(process.env.AUTO_FORGET_INTERVAL_MS || "3600000", 10);
  const consolidationIntervalMs = parseInt(process.env.CONSOLIDATION_INTERVAL_MS || "7200000", 10);

  if (process.env.AUTO_FORGET_ENABLED !== "false") {
    const autoForgetTimer = setInterval(async () => {
      try {
        await sdk.trigger({ function_id: "mem::auto-forget", payload: { dryRun: false } });
      } catch {}
    }, autoForgetIntervalMs);
    autoForgetTimer.unref();
    bootLog(`Auto-forget: enabled (every ${autoForgetIntervalMs / 60000}m)`);
  }

  if (process.env.LESSON_DECAY_ENABLED !== "false") {
    const lessonDecayTimer = setInterval(async () => {
      try {
        await sdk.trigger({ function_id: "mem::lesson-decay-sweep", payload: {} });
      } catch {}
    }, 86400000);
    lessonDecayTimer.unref();
    bootLog(`Lesson decay sweep: enabled (every 24h)`);
  }

  if (process.env.INSIGHT_DECAY_ENABLED !== "false") {
    const insightDecayTimer = setInterval(async () => {
      try {
        await sdk.trigger({ function_id: "mem::insight-decay-sweep", payload: {} });
      } catch {}
    }, 86400000);
    insightDecayTimer.unref();
  }

  // #771: hourly TTL sweep for the followup-rate diagnostic. The
  // recent-searches scope only needs the last entry per session;
  // sweeping anything older than the retention window keeps the scope
  // from growing unbounded across long-lived deployments.
  const recentSearchesSweepTimer = setInterval(async () => {
    try {
      await sdk.trigger({
        function_id: "mem::diagnostic::recent-searches-sweep",
        payload: {},
      });
    } catch {}
  }, 60 * 60 * 1000);
  recentSearchesSweepTimer.unref();

  if (isConsolidationEnabled()) {
    const consolidationTimer = setInterval(async () => {
      try {
        await sdk.trigger({ function_id: "mem::consolidate-pipeline", payload: {} });
      } catch {}
    }, consolidationIntervalMs);
    consolidationTimer.unref();
    bootLog(`Auto-consolidation: enabled (every ${consolidationIntervalMs / 60000}m)`);
  }

  const shutdown = async () => {
    console.log(`\n[agentmemory] Shutting down...`);
    healthMonitor.stop();
    dedupMap.stop();
    indexPersistence?.stop();
    await new Promise<void>((resolve) => viewerServer.close(() => resolve()));
    await indexPersistence?.save().catch((err) => {
      console.warn(`[agentmemory] Failed to save index on shutdown:`, err);
    });
    await sdk.shutdown();
    clearWorkerPidfile();
    process.exit(0);
  };
  process.on("SIGINT", shutdown);
  process.on("SIGTERM", shutdown);
}

main().catch((err) => {
  console.error(`[agentmemory] Fatal:`, err);
  process.exit(1);
});
