import { TriggerAction, type ISdk } from "iii-sdk";
import { readFileSync } from "node:fs";
import { isManagedImagePath } from "../utils/image-store.js";
import type {
  RawObservation,
  CompressedObservation,
  ObservationType,
  MemoryProvider,
} from "../types.js";
import { KV, STREAM } from "../state/schema.js";
import { StateKV } from "../state/kv.js";
import {
  COMPRESSION_SYSTEM,
  buildCompressionPrompt,
} from "../prompts/compression.js";
import { VISION_DESCRIPTION_PROMPT } from "../prompts/vision.js";
import { getXmlTag, getXmlChildren } from "../prompts/xml.js";
import { getSearchIndex, vectorIndexAddGuarded, isIndexExcluded, markIndexDirty } from "./search.js";
import { buildSyntheticCompression } from "./compress-synthetic.js";
import { CompressOutputSchema } from "../eval/schemas.js";
import { validateOutput } from "../eval/validator.js";
import { scoreCompression } from "../eval/quality.js";
import { compressWithRetry } from "../eval/self-correct.js";
import type { MetricsStore } from "../eval/metrics-store.js";
import { logger } from "../logger.js";

const VALID_TYPES = new Set<string>([
  "file_read",
  "file_write",
  "file_edit",
  "command_run",
  "search",
  "web_fetch",
  "conversation",
  "error",
  "decision",
  "discovery",
  "subagent",
  "notification",
  "task",
  "image",
  "other",
]);

function parseCompressionXml(
  xml: string,
): Omit<CompressedObservation, "id" | "sessionId" | "timestamp"> | null {
  const rawType = getXmlTag(xml, "type");
  const title = getXmlTag(xml, "title");
  if (!rawType || !title) return null;
  const type = VALID_TYPES.has(rawType) ? rawType : "other";

  return {
    type: type as ObservationType,
    title,
    subtitle: getXmlTag(xml, "subtitle") || undefined,
    facts: getXmlChildren(xml, "facts", "fact"),
    narrative: getXmlTag(xml, "narrative"),
    concepts: getXmlChildren(xml, "concepts", "concept"),
    files: getXmlChildren(xml, "files", "file"),
    importance: Math.max(
      1,
      Math.min(10, parseInt(getXmlTag(xml, "importance") || "5", 10) || 5),
    ),
  };
}

export function registerCompressFunction(
  sdk: ISdk,
  kv: StateKV,
  provider: MemoryProvider,
  metricsStore?: MetricsStore,
): void {
  sdk.registerFunction("mem::compress", 
    async (data: {
      observationId: string;
      sessionId: string;
      raw: RawObservation;
    }) => {
      const startMs = Date.now();

      // Compression is the ONLY writer of KV.observations for this id -
      // observe.ts hands us the raw record and stores nothing itself, and
      // nothing re-compresses a failure later. So every early return that
      // skips the kv.set below leaves the observation present on disk as raw
      // bytes but absent from BOTH search legs: unfindable forever, from one
      // transient provider 5xx. Degrade to the same zero-LLM synthetic record
      // observe.ts writes when auto-compress is off. confidence stays 0.3,
      // which is what marks these for a later re-compression sweep.
      // This is a degradation, not a silent fallback: the caller still gets
      // success:false and the error log still fires, so the alert still pages.
      const storeDegraded = async (reason: string): Promise<void> => {
        try {
          // Park the full raw BEFORE overwriting it, so the sweep that
          // re-compresses this later works from the original and not from
          // the synthetic's truncated narrative. attempts counts how many
          // times this record has failed compression - the sweep uses it to
          // quarantine records that fail deterministically (e.g. content the
          // model refuses to summarize) instead of cycling on them forever.
          const prior = await kv.get<{ attempts?: number }>(
            KV.compressPending,
            data.observationId,
          );
          await kv.set(KV.compressPending, data.observationId, {
            observationId: data.observationId,
            sessionId: data.sessionId,
            raw: data.raw,
            reason,
            failedAt: new Date().toISOString(),
            attempts: (prior?.attempts ?? 0) + 1,
          });

          const synthetic = buildSyntheticCompression(data.raw);
          synthetic.id = data.observationId;
          synthetic.sessionId = data.sessionId;
          await kv.set(
            KV.observations(data.sessionId),
            data.observationId,
            synthetic,
          );
          if (!isIndexExcluded(synthetic)) {
            getSearchIndex().add(synthetic);
            await vectorIndexAddGuarded(
              synthetic.id,
              synthetic.sessionId,
              synthetic.title + " " + (synthetic.narrative || ""),
              { kind: "synthetic", logId: synthetic.id },
            );
            markIndexDirty();
          }
          logger.warn("Stored degraded synthetic observation after compression failure", {
            obsId: data.observationId,
            sessionId: data.sessionId,
            reason,
          });
        } catch (err) {
          // Nothing left to fall back to. Log loudly: this is the case where
          // an observation really is lost to retrieval.
          logger.error("Degraded store FAILED - observation is unretrievable", {
            obsId: data.observationId,
            sessionId: data.sessionId,
            reason,
            error: err instanceof Error ? err.message : String(err),
          });
        }
      };

      let imageDescription: string | undefined;
      const hasImage = data.raw.modality === "image" || data.raw.modality === "mixed";

      if (hasImage && data.raw.imageData && provider.describeImage) {
        try {
          let base64Data = data.raw.imageData;
          let mimeType = "image/png";

          if (!data.raw.imageData.startsWith("/9j/") && !data.raw.imageData.startsWith("iVBOR")) {
            if (!isManagedImagePath(data.raw.imageData)) {
              throw new Error(`Refusing to read image outside managed store: ${data.raw.imageData}`);
            }
            const fileBuffer = readFileSync(data.raw.imageData);
            base64Data = fileBuffer.toString("base64");
            if (data.raw.imageData.endsWith(".jpg") || data.raw.imageData.endsWith(".jpeg")) mimeType = "image/jpeg";
            else if (data.raw.imageData.endsWith(".webp")) mimeType = "image/webp";
            else if (data.raw.imageData.endsWith(".gif")) mimeType = "image/gif";
          }

          imageDescription = await provider.describeImage(base64Data, mimeType, VISION_DESCRIPTION_PROMPT);
          logger.info("Image described by vision model", { obsId: data.observationId });
        } catch (err) {
          const msg = err instanceof Error ? err.message : String(err);
          logger.warn("Vision model call failed, falling back to text-only compression", {
            obsId: data.observationId,
            error: msg,
          });
        }
      }

      const promptArgs = {
        hookType: data.raw.hookType,
        toolName: data.raw.toolName,
        toolInput: data.raw.toolInput,
        toolOutput: imageDescription
          ? `[Image Description]: ${imageDescription}\n\n${data.raw.toolOutput ?? ""}`
          : data.raw.toolOutput,
        userPrompt: data.raw.userPrompt,
        timestamp: data.raw.timestamp,
      };
      const prompt = buildCompressionPrompt(promptArgs);

      try {
        const validator = (response: string) => {
          const parsed = parseCompressionXml(response);
          if (!parsed) return { valid: false, errors: ["xml_parse_failed"] };
          const result = validateOutput(
            CompressOutputSchema,
            parsed,
            "mem::compress",
          );
          return result.valid
            ? { valid: true }
            : { valid: false, errors: result.result.errors };
        };

        const { response, retried } = await compressWithRetry(
          provider,
          COMPRESSION_SYSTEM,
          prompt,
          validator,
          1,
          buildCompressionPrompt(promptArgs, { neutralize: true }),
        );

        const parsed = parseCompressionXml(response);
        if (!parsed) {
          const latencyMs = Date.now() - startMs;
          if (metricsStore) {
            await metricsStore.record("mem::compress", latencyMs, false);
          }
          logger.warn("Failed to parse compression XML", {
            obsId: data.observationId,
            retried,
          });
          await storeDegraded("parse_failed");
          return { success: false, error: "parse_failed" };
        }

        const qualityScore = scoreCompression(parsed);

        const compressed: CompressedObservation = {
          id: data.observationId,
          sessionId: data.sessionId,
          timestamp: data.raw.timestamp,
          ...parsed,
          confidence: qualityScore / 100,
          ...(hasImage ? { modality: data.raw.modality } : {}),
          ...(imageDescription ? { imageDescription } : {}),
          ...(data.raw.imageData ? { imageRef: data.raw.imageData } : {}),
          ...(data.raw.agentId ? { agentId: data.raw.agentId } : {}),
          ...(data.raw.origin ? { origin: data.raw.origin } : {}),
          // Persisted so the index-write sites can recognise the
          // daemon's own retrieval calls. Compression overwrites the raw
          // KV record, so without this the tool name is gone from disk.
          ...(data.raw.toolName ? { toolName: data.raw.toolName } : {}),
          // Full prompt text survives compression verbatim - the summary
          // above is derived from it, it does not replace it.
          ...(data.raw.userPrompt ? { userPrompt: data.raw.userPrompt } : {}),
        };

        await kv.set(
          KV.observations(data.sessionId),
          data.observationId,
          compressed,
        );

        // Re-compressing a previously-failed observation clears its recovery
        // entry. A no-op for the overwhelming majority that never failed.
        await kv.delete(KV.compressPending, data.observationId);

        // Stored above unconditionally; only the INDEX writes are
        // skipped for excluded tools (retrieval echoes).
        if (!isIndexExcluded(compressed)) {
          try {
            getSearchIndex().add(compressed);
          } catch (err) {
            logger.warn("Failed to index compressed observation into BM25", {
              obsId: compressed.id,
              sessionId: compressed.sessionId,
              title: compressed.title,
              error: err instanceof Error ? err.message : String(err),
            });
          }

          await vectorIndexAddGuarded(
            compressed.id,
            compressed.sessionId,
            compressed.title + " " + (compressed.narrative || ""),
            { kind: "observation", logId: compressed.id },
          );
          markIndexDirty();
        }

        const streamResults = await Promise.allSettled([
          sdk.trigger({
            function_id: "stream::set",
            payload: {
              stream_name: STREAM.name,
              group_id: STREAM.group(data.sessionId),
              item_id: data.observationId,
              data: { type: "compressed", observation: compressed },
            },
          }),
          sdk.trigger({
            function_id: "stream::send",
            payload: {
              stream_name: STREAM.name,
              group_id: STREAM.viewerGroup,
              id: `compressed-${data.observationId}`,
              type: "compressed_observation",
              data: {
                type: "compressed",
                observation: compressed,
                sessionId: data.sessionId,
              },
            },
            action: TriggerAction.Void(),
          }),
        ]);
        for (const result of streamResults) {
          if (result.status === "rejected") {
            logger.warn("Non-fatal stream publish failure after compress", {
              sessionId: data.sessionId,
              observationId: data.observationId,
              error:
                result.reason instanceof Error
                  ? result.reason.message
                  : String(result.reason),
            });
          }
        }

        const latencyMs = Date.now() - startMs;
        if (metricsStore) {
          await metricsStore.record(
            "mem::compress",
            latencyMs,
            true,
            qualityScore,
          );
        }

        logger.info("Observation compressed", {
          obsId: data.observationId,
          type: compressed.type,
          importance: compressed.importance,
          qualityScore,
          retried,
        });

        return { success: true, compressed, qualityScore };
      } catch (err) {
        const msg = err instanceof Error ? err.message : String(err);
        const latencyMs = Date.now() - startMs;
        if (metricsStore) {
          await metricsStore.record("mem::compress", latencyMs, false);
        }
        logger.error("Compression failed", {
          obsId: data.observationId,
          error: msg,
        });
        await storeDegraded("compression_failed");
        return { success: false, error: "compression_failed" };
      }
    },
  );
}
