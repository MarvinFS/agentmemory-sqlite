import type { ISdk } from "iii-sdk";
import type {
  SemanticMemory,
  ProceduralMemory,
  SessionSummary,
  Memory,
  MemoryProvider,
} from "../types.js";
import { KV, generateId } from "../state/schema.js";
import type { StateKV } from "../state/kv.js";
import {
  SEMANTIC_MERGE_SYSTEM,
  buildSemanticMergePrompt,
  PROCEDURAL_EXTRACTION_SYSTEM,
  buildProceduralExtractionPrompt,
} from "../prompts/consolidation.js";
import { recordAudit } from "./audit.js";
import { getConsolidationDecayDays, isConsolidationEnabled } from "../config.js";
import { logger } from "../logger.js";

// Returns the rows whose strength changed, so the caller writes only those. A
// row inside its decay window or already at the 0.1 floor is left alone: the
// old "rewrite every row" tier was 45k fsyncs per run (day-0 soak finding).
function applyDecay<
  T extends { strength: number; lastAccessedAt?: string; updatedAt: string },
>(items: T[], decayDays: number): T[] {
  if (decayDays <= 0 || !Number.isFinite(decayDays)) return [];
  const changed: T[] = [];
  const now = Date.now();
  for (const item of items) {
    const lastAccess = item.lastAccessedAt || item.updatedAt;
    const daysSince =
      (now - new Date(lastAccess).getTime()) / (1000 * 60 * 60 * 24);
    if (daysSince > decayDays) {
      const decayPeriods = Math.floor(daysSince / decayDays);
      const next = Math.max(0.1, item.strength * Math.pow(0.9, decayPeriods));
      if (next !== item.strength) {
        item.strength = next;
        changed.push(item);
      }
    }
  }
  return changed;
}

export function registerConsolidationPipelineFunction(
  sdk: ISdk,
  kv: StateKV,
  provider: MemoryProvider,
): void {
  // The 5 min session-end cooldown and the 120 min timer are schedules, not a
  // lock: a run that outlived them was joined by a second one over the same
  // rows (day-0 soak finding). One run at a time; a caller that finds one in
  // flight is told so and skipped, not queued.
  let inflight: Promise<unknown> | null = null;
  sdk.registerFunction("mem::consolidate-pipeline",
    async (data?: { tier?: string; force?: boolean; project?: string }) => {
      if (!data?.force && !isConsolidationEnabled()) {
        return { success: false, skipped: true, reason: "Consolidation disabled: set CONSOLIDATION_ENABLED=true or configure an LLM provider (ANTHROPIC_API_KEY / OPENAI_API_KEY / OPENROUTER_API_KEY / GEMINI_API_KEY / GOOGLE_API_KEY / MINIMAX_API_KEY / OPENAI_BASE_URL / AGENTMEMORY_PROVIDER=agent-sdk)" };
      }
      if (inflight) {
        return { success: false, skipped: true, reason: "Consolidation pipeline already running" };
      }
      inflight = run(data);
      try {
        return await inflight;
      } finally {
        inflight = null;
      }
    },
  );

  async function run(data?: { tier?: string; project?: string }) {
    const tier = data?.tier || "all";
    const decayDays = getConsolidationDecayDays();
    const results: Record<string, unknown> = {};

    if (tier === "all" || tier === "semantic") {
      const summaries = await kv.list<SessionSummary>(KV.summaries);
      const existingSemantic = await kv.list<SemanticMemory>(KV.semantic);

      if (summaries.length >= 5) {
        const recentSummaries = summaries
          .sort(
            (a, b) =>
              new Date(b.createdAt).getTime() -
              new Date(a.createdAt).getTime(),
          )
          .slice(0, 20);

        const prompt = buildSemanticMergePrompt(
          recentSummaries.map((s) => ({
            title: s.title,
            narrative: s.narrative,
            concepts: s.concepts,
          })),
        );

        try {
          const response = await provider.summarize(
            SEMANTIC_MERGE_SYSTEM,
            prompt,
          );

          const factRegex = /<fact\s+confidence="([^"]+)">([^<]+)<\/fact>/g;
          let match;
          let newFacts = 0;
          const now = new Date().toISOString();

          while ((match = factRegex.exec(response)) !== null) {
            const parsedConf = parseFloat(match[1]);
            const confidence = Number.isNaN(parsedConf) ? 0.5 : parsedConf;
            const fact = match[2].trim();

            const existing = existingSemantic.find(
              (s) => s.fact.toLowerCase() === fact.toLowerCase(),
            );
            if (existing) {
              existing.accessCount++;
              existing.lastAccessedAt = now;
              existing.updatedAt = now;
              existing.confidence = Math.max(existing.confidence, confidence);
              await kv.set(KV.semantic, existing.id, existing);
            } else {
              const sem: SemanticMemory = {
                id: generateId("sem"),
                fact,
                confidence,
                sourceSessionIds: recentSummaries.map((s) => s.sessionId),
                sourceMemoryIds: [],
                accessCount: 1,
                lastAccessedAt: now,
                strength: confidence,
                createdAt: now,
                updatedAt: now,
              };
              await kv.set(KV.semantic, sem.id, sem);
              newFacts++;
            }
          }
          results.semantic = { newFacts, totalSummaries: summaries.length };
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          logger.error("Semantic consolidation failed", { error: msg });
          results.semantic = { error: msg };
        }
      } else {
        results.semantic = {
          skipped: true,
          reason: "fewer than 5 summaries",
        };
      }
    }

    if (tier === "all" || tier === "reflect") {
      try {
        const reflectResult = await sdk.trigger({ function_id: "mem::reflect", payload: {
          maxClusters: 10,
          project: data?.project,
        } });
        results.reflect = reflectResult;
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        logger.warn("Reflect tier failed", { error: msg });
        results.reflect = { error: msg };
      }
    }

    if (tier === "all" || tier === "procedural") {
      const memories = await kv.list<Memory>(KV.memories);
      const patterns = memories
        .filter((m) => m.isLatest && m.type === "pattern")
        .map((m) => ({
          content: m.content,
          frequency: m.sessionIds.length || 1,
        }))
        .filter((p) => p.frequency >= 2);

      if (patterns.length >= 2) {
        const prompt = buildProceduralExtractionPrompt(patterns);

        try {
          const response = await provider.summarize(
            PROCEDURAL_EXTRACTION_SYSTEM,
            prompt,
          );

          const procRegex =
            /<procedure\s+name="([^"]+)"\s+trigger="([^"]+)">([\s\S]*?)<\/procedure>/g;
          let match;
          let newProcs = 0;
          const now = new Date().toISOString();
          const existingProcs = await kv.list<ProceduralMemory>(
            KV.procedural,
          );

          while ((match = procRegex.exec(response)) !== null) {
            const name = match[1];
            const trigger = match[2];
            const stepsBlock = match[3];
            const steps: string[] = [];

            const stepRegex = /<step>([^<]+)<\/step>/g;
            let stepMatch;
            while ((stepMatch = stepRegex.exec(stepsBlock)) !== null) {
              steps.push(stepMatch[1].trim());
            }

            const existing = existingProcs.find(
              (p) => p.name.toLowerCase() === name.toLowerCase(),
            );
            if (existing) {
              existing.frequency++;
              existing.updatedAt = now;
              existing.strength = Math.min(1, existing.strength + 0.1);
              await kv.set(KV.procedural, existing.id, existing);
            } else {
              const proc: ProceduralMemory = {
                id: generateId("proc"),
                name,
                steps,
                triggerCondition: trigger,
                frequency: 1,
                sourceSessionIds: [],
                strength: 0.5,
                createdAt: now,
                updatedAt: now,
              };
              await kv.set(KV.procedural, proc.id, proc);
              newProcs++;
            }
          }
          results.procedural = {
            newProcedures: newProcs,
            patternsAnalyzed: patterns.length,
          };
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          logger.error("Procedural extraction failed", { error: msg });
          results.procedural = { error: msg };
        }
      } else {
        results.procedural = {
          skipped: true,
          reason: "fewer than 2 recurring patterns",
        };
      }
    }

    if (tier === "all" || tier === "decay") {
      const semantic = await kv.list<SemanticMemory>(KV.semantic);
      const semanticChanged = applyDecay(semantic, decayDays);
      await kv.setMany(KV.semantic, semanticChanged.map((s) => ({ key: s.id, value: s })));

      const procedural = await kv.list<ProceduralMemory>(KV.procedural);
      const proceduralChanged = applyDecay(procedural, decayDays);
      await kv.setMany(KV.procedural, proceduralChanged.map((p) => ({ key: p.id, value: p })));

      results.decay = {
        semantic: semantic.length,
        procedural: procedural.length,
        semanticWritten: semanticChanged.length,
        proceduralWritten: proceduralChanged.length,
      };
    }

    if (process.env["OBSIDIAN_AUTO_EXPORT"] === "true") {
      try {
        await sdk.trigger({ function_id: "mem::obsidian-export", payload: {} });
        results.obsidianExport = { success: true };
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        logger.warn("Obsidian auto-export failed", { error: msg });
        results.obsidianExport = { success: false, error: msg };
      }
    }

    await recordAudit(kv, "consolidate", "mem::consolidate-pipeline", [], {
      tier,
      results,
    });

    logger.info("Consolidation pipeline complete", { tier, results });
    return { success: true, results };
  }
}
