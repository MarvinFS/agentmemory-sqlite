import { SearchIndex } from "./search-index.js";
import { VectorIndex } from "./vector-index.js";
import type { StateKV } from "./kv.js";
import { KV, generateId } from "./schema.js";
import { logger } from "../logger.js";
import { safeAudit } from "../functions/audit.js";

const DEBOUNCE_MS = 5000;
const FAILURE_LOG_THROTTLE_MS = 60_000;
const INDEX_PERSISTENCE_FUNCTION_ID = "mem::index-persistence";
const BM25_KEY = "data";
const BM25_MANIFEST_KEY = "data:manifest";
const BM25_SHARD_SCOPE_PREFIX = `${KV.bm25Index}:bm25:`;
const VECTOR_KEY = "vectors";
const VECTOR_MANIFEST_KEY = "vectors:manifest";
const VECTOR_SHARD_SCOPE_PREFIX = `${KV.bm25Index}:vectors:`;
const INDEX_SHARD_KEY = "data";
// The A2 rollback artifact: the manifest pair that was live before the
// reconcile published. Their shards are deliberately never cleaned up.
const CHECKPOINT_ROLLBACK_KEY = "checkpoint:rollback";
const DEFAULT_INDEX_SHARD_CHARS = 2_000_000;

type IndexShardManifest = {
  v: 1;
  generation?: string;
  shards: Array<{ scope: string; key: string; chars: number }>;
  chars: number;
};

export type IndexCheckpoint = {
  generation: string;
  previous: {
    bm25: IndexShardManifest | null;
    vector: IndexShardManifest | null;
  };
};

type IndexPersistenceOptions = {
  shardChars?: number;
  createGeneration?: () => string;
};

function shardChars(options: IndexPersistenceOptions): number {
  const configured = options.shardChars;
  if (typeof configured !== "number" || !Number.isFinite(configured)) {
    return DEFAULT_INDEX_SHARD_CHARS;
  }
  const wholeChars = Math.floor(configured);
  return wholeChars >= 1 ? wholeChars : DEFAULT_INDEX_SHARD_CHARS;
}

function createIndexGeneration(): string {
  return generateId("idx");
}

function statePath(scope: string, key: string): string {
  return `${scope}/${key}`;
}

function errorMessage(err: unknown): string {
  return err instanceof Error ? err.message : String(err);
}

function isValidShardDescriptor(
  shard: unknown,
): shard is IndexShardManifest["shards"][number] {
  if (!shard || typeof shard !== "object") return false;
  const candidate = shard as { scope?: unknown; key?: unknown; chars?: unknown };
  return (
    typeof candidate.scope === "string" &&
    candidate.scope.length > 0 &&
    typeof candidate.key === "string" &&
    candidate.key.length > 0 &&
    Number.isInteger(candidate.chars) &&
    candidate.chars >= 0
  );
}

// How often a dirty index is flushed to KV. Serializing the whole index is
// expensive (hundreds of MB on a large corpus), so this is deliberately NOT
// the 5s add-debounce — it is a periodic checkpoint. The exposure window is
// one interval's worth of live-indexed entries on an unclean exit.
const PERIODIC_SAVE_MS = 300_000;

export class IndexPersistence {
  private timer: ReturnType<typeof setTimeout> | null = null;
  private periodic: ReturnType<typeof setInterval> | null = null;
  private dirty = false;
  private lastFailureLogAt = 0;

  constructor(
    private kv: StateKV,
    private bm25: SearchIndex,
    private vector: VectorIndex | null,
    private options: IndexPersistenceOptions = {},
  ) {}

  // Every observation and memory indexed while the daemon runs is added to
  // the IN-MEMORY index only. Nothing was flushing that to disk: scheduleSave
  // is called at boot (after a rebuild or backfill) and save() on deletes, so
  // the on-disk snapshot froze at the last rebuild and every entry indexed
  // since was lost on restart. Measured on production: a 24,027-document
  // corpus restored 17,625 documents, silently missing everything indexed
  // since the previous rebuild.
  //
  // markDirty() is the cheap call the live add sites make; the flush itself
  // is amortised onto a timer.
  markDirty(): void {
    this.dirty = true;
    if (this.periodic) return;
    const ms = Number(process.env.AGENTMEMORY_INDEX_SAVE_INTERVAL_MS) || PERIODIC_SAVE_MS;
    this.periodic = setInterval(() => {
      if (!this.dirty) return;
      this.dirty = false;
      this.save().catch((err) => {
        this.dirty = true; // retry on the next tick
        this.logFailure(err);
      });
    }, ms);
    // Never hold the event loop open for this alone.
    this.periodic.unref?.();
  }

  scheduleSave(): void {
    if (this.timer) clearTimeout(this.timer);
    // setTimeout discards the returned promise, so any rejection inside
    // save() would surface as unhandledRejection and crash the process
    // under sustained iii-engine write timeouts (issue #204). Funnel
    // rejections through logFailure() instead.
    this.timer = setTimeout(() => {
      this.save().catch((err) => this.logFailure(err));
    }, DEBOUNCE_MS);
  }

  async save(): Promise<void> {
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
    try {
      await this.saveBm25Index(this.bm25.serialize());
      if (this.vector) {
        await this.saveVectorIndex(this.vector.serialize());
      }
    } catch (err) {
      this.logFailure(err);
    }
  }

  async load(): Promise<{
    bm25: SearchIndex | null;
    vector: VectorIndex | null;
  }> {
    let bm25: SearchIndex | null = null;
    let vector: VectorIndex | null = null;

    const bm25Data = await this.loadBm25Data();
    if (bm25Data && typeof bm25Data === "string") {
      bm25 = SearchIndex.deserialize(bm25Data);
    }

    const vecData = await this.loadVectorData();
    if (vecData && typeof vecData === "string") {
      vector = VectorIndex.deserialize(vecData);
    }

    return { bm25, vector };
  }

  stop(): void {
    if (this.periodic) {
      clearInterval(this.periodic);
      this.periodic = null;
    }
    if (this.timer) {
      clearTimeout(this.timer);
      this.timer = null;
    }
  }

  private logFailure(err: unknown): void {
    const now = Date.now();
    // Throttle: persistence failures under load arrive in bursts
    // (iii-engine queue pressure). Logging every debounce flush adds
    // noise without information.
    if (now - this.lastFailureLogAt < FAILURE_LOG_THROTTLE_MS) return;
    this.lastFailureLogAt = now;
    const code = (err as { code?: string })?.code;
    const message = err instanceof Error ? err.message : String(err);
    logger.warn("index persistence: failed to save BM25/vector index", {
      code,
      message,
      hint:
        code === "TIMEOUT"
          ? "iii-engine state::set timed out; recent index updates remain in memory and will retry on the next debounce flush"
          : undefined,
    });
  }

  private async saveBm25Index(serialized: string): Promise<void> {
    await this.saveShardedIndex(
      serialized,
      BM25_MANIFEST_KEY,
      BM25_KEY,
      BM25_SHARD_SCOPE_PREFIX,
    );
  }

  private async saveVectorIndex(serialized: string): Promise<void> {
    await this.saveShardedIndex(
      serialized,
      VECTOR_MANIFEST_KEY,
      VECTOR_KEY,
      VECTOR_SHARD_SCOPE_PREFIX,
    );
  }

  private async saveShardedIndex(
    serialized: string,
    manifestKey: string,
    legacyKey: string,
    scopePrefix: string,
  ): Promise<void> {
    const previous = await this.kv
      .get<IndexShardManifest>(KV.bm25Index, manifestKey)
      .catch(() => null);
    const generation =
      this.options.createGeneration?.() ?? createIndexGeneration();
    const chunkChars = shardChars(this.options);
    const shards: IndexShardManifest["shards"] = [];
    const chunks: string[] = [];

    for (let offset = 0; offset < serialized.length; offset += chunkChars) {
      const shardIndex = shards.length;
      const scope = `${scopePrefix}${generation}:${String(shardIndex).padStart(
        5,
        "0",
      )}`;
      const chunk = serialized.slice(offset, offset + chunkChars);
      shards.push({ scope, key: INDEX_SHARD_KEY, chars: chunk.length });
      chunks.push(chunk);
    }

    const writeResults = await Promise.allSettled(
      shards.map(async (shard, index) => {
        const chunk = chunks[index] ?? "";
        await this.kv.set(shard.scope, shard.key, chunk);
        // Successful per-shard writes are deliberately NOT audited. A large
        // index is hundreds of shards and the periodic flush republishes it
        // every five minutes, so these rows dominated the audit log (42 MB)
        // while carrying nothing the retained manifest_publish row does not
        // already state - same generation, same shard count, same total
        // chars. Suppression rather than aggregation for that reason.
        // Failures are untouched: a rejected write still rolls back through
        // deleteShards, which audits every delete.
      }),
    );
    const failedWrite = writeResults.find(
      (result): result is PromiseRejectedResult => result.status === "rejected",
    );
    if (failedWrite) {
      await this.deleteShards(shards, "shard_write_rollback");
      throw failedWrite.reason;
    }

    const nextManifest: IndexShardManifest = {
      v: 1,
      generation,
      shards,
      chars: serialized.length,
    };
    try {
      await this.kv.set<IndexShardManifest>(
        KV.bm25Index,
        manifestKey,
        nextManifest,
      );
      await this.auditIndexPersistence("manifest_publish", [
        statePath(KV.bm25Index, manifestKey),
      ], {
        manifestKey,
        generation,
        chars: serialized.length,
        shards: shards.length,
        result: "committed",
      });
    } catch (err) {
      if (await this.isManifestPublished(manifestKey, nextManifest)) {
        await this.auditIndexPersistence("manifest_publish", [
          statePath(KV.bm25Index, manifestKey),
        ], {
          manifestKey,
          generation,
          chars: serialized.length,
          shards: shards.length,
          result: "committed_after_error",
          error: errorMessage(err),
        });
      } else {
        await this.deleteShards(shards, "manifest_publish_rollback");
      }
      throw err;
    }

    await this.deleteKey(KV.bm25Index, legacyKey, "legacy_cleanup");
    if (previous?.v === 1 && Array.isArray(previous.shards)) {
      const currentShardIds = new Set(
        shards.map((shard) => `${shard.scope}\0${shard.key}`),
      );
      for (const shard of previous.shards) {
        if (currentShardIds.has(`${shard.scope}\0${shard.key}`)) continue;
        await this.deleteShards([shard], "previous_generation_cleanup");
      }
    }
  }

  // --- A2 reconcile checkpoint -------------------------------------------
  //
  // save() publishes BM25 and vector as two INDEPENDENT operations, each
  // minting its own generation and each deleting its predecessor's shards
  // on success. That is fine for incremental checkpoints and wrong for a
  // baseline reconcile, which needs the opposite of both properties: one
  // generation covering both indexes, and a predecessor that SURVIVES so
  // there is something to roll back to.
  //
  // Atomicity here is publish-both-then-compensate, not a true two-key
  // commit — the store has no multi-key transaction and the load path
  // reads two manifests. If the second publish fails the first is rolled
  // back to its previous value, and the reconcile runs with the daemon
  // quiesced, so no reader observes the window. Making the combined
  // checkpoint the canonical publisher belongs to 4B.
  async saveCheckpoint(
    bm25Serialized: string,
    vectorSerialized: string | null,
  ): Promise<IndexCheckpoint> {
    const generation =
      this.options.createGeneration?.() ?? createIndexGeneration();
    const previousBm25 = await this.readManifest(BM25_MANIFEST_KEY);
    const previousVector = await this.readManifest(VECTOR_MANIFEST_KEY);

    // Recorded BEFORE anything is written. Rollback restores these two
    // manifest objects; their shards are never deleted by this method, and
    // ordinary saves afterwards only ever delete the generation they
    // themselves superseded — so the artifact stays intact.
    const rollback: IndexCheckpoint = {
      generation,
      previous: {
        bm25: previousBm25 ?? null,
        vector: previousVector ?? null,
      },
    };
    await this.kv.set(KV.bm25Index, CHECKPOINT_ROLLBACK_KEY, rollback);

    const bm25Shards = await this.writeGeneration(
      bm25Serialized,
      BM25_SHARD_SCOPE_PREFIX,
      generation,
    );
    let vectorShards: IndexShardManifest["shards"] | null = null;
    if (vectorSerialized !== null) {
      try {
        vectorShards = await this.writeGeneration(
          vectorSerialized,
          VECTOR_SHARD_SCOPE_PREFIX,
          generation,
        );
      } catch (err) {
        await this.deleteShards(bm25Shards, "checkpoint_rollback");
        throw err;
      }
    }

    const bm25Manifest: IndexShardManifest = {
      v: 1,
      generation,
      shards: bm25Shards,
      chars: bm25Serialized.length,
    };
    await this.kv.set(KV.bm25Index, BM25_MANIFEST_KEY, bm25Manifest);

    if (vectorShards && vectorSerialized !== null) {
      const vectorManifest: IndexShardManifest = {
        v: 1,
        generation,
        shards: vectorShards,
        chars: vectorSerialized.length,
      };
      try {
        await this.kv.set(KV.bm25Index, VECTOR_MANIFEST_KEY, vectorManifest);
      } catch (err) {
        // Compensate: put BM25 back where it was, so the pair stays
        // consistent. A half-published checkpoint is the one outcome
        // worse than not publishing at all.
        if (previousBm25) {
          await this.kv
            .set(KV.bm25Index, BM25_MANIFEST_KEY, previousBm25)
            .catch(() => undefined);
        }
        await this.deleteShards(bm25Shards, "checkpoint_rollback");
        await this.deleteShards(vectorShards, "checkpoint_rollback");
        throw err;
      }
    }

    await this.auditIndexPersistence(
      "checkpoint_publish",
      [
        statePath(KV.bm25Index, BM25_MANIFEST_KEY),
        statePath(KV.bm25Index, VECTOR_MANIFEST_KEY),
      ],
      {
        generation,
        bm25Chars: bm25Serialized.length,
        vectorChars: vectorSerialized?.length ?? 0,
        previousBm25Generation: previousBm25?.generation ?? null,
        previousVectorGeneration: previousVector?.generation ?? null,
        cleanup: "skipped_predecessor_retained",
      },
    );
    return rollback;
  }

  async readCheckpointRollback(): Promise<IndexCheckpoint | null> {
    return (
      (await this.kv
        .get<IndexCheckpoint>(KV.bm25Index, CHECKPOINT_ROLLBACK_KEY)
        .catch(() => null)) ?? null
    );
  }

  private async readManifest(
    manifestKey: string,
  ): Promise<IndexShardManifest | null> {
    const manifest = await this.kv
      .get<IndexShardManifest>(KV.bm25Index, manifestKey)
      .catch(() => null);
    return manifest?.v === 1 && Array.isArray(manifest.shards)
      ? manifest
      : null;
  }

  private async writeGeneration(
    serialized: string,
    scopePrefix: string,
    generation: string,
  ): Promise<IndexShardManifest["shards"]> {
    const chunkChars = shardChars(this.options);
    const shards: IndexShardManifest["shards"] = [];
    const chunks: string[] = [];
    for (let offset = 0; offset < serialized.length; offset += chunkChars) {
      const scope = `${scopePrefix}${generation}:${String(
        shards.length,
      ).padStart(5, "0")}`;
      const chunk = serialized.slice(offset, offset + chunkChars);
      shards.push({ scope, key: INDEX_SHARD_KEY, chars: chunk.length });
      chunks.push(chunk);
    }
    const writes = await Promise.allSettled(
      shards.map(async (shard, index) => {
        // Same suppression as the ordinary save path: the checkpoint's own
        // audit row records generation, both manifests and their sizes.
        await this.kv.set(shard.scope, shard.key, chunks[index] ?? "");
      }),
    );
    const failed = writes.find(
      (r): r is PromiseRejectedResult => r.status === "rejected",
    );
    if (failed) {
      await this.deleteShards(shards, "checkpoint_shard_write_rollback");
      throw failed.reason;
    }
    return shards;
  }

  private async auditIndexPersistence(
    action: string,
    targetIds: string[],
    details: Record<string, unknown>,
  ): Promise<void> {
    await safeAudit(
      this.kv,
      "index_persist",
      INDEX_PERSISTENCE_FUNCTION_ID,
      targetIds,
      { action, ...details },
    );
  }

  private async deleteKey(
    scope: string,
    key: string,
    reason: string,
  ): Promise<void> {
    let result = "deleted";
    let error: string | undefined;
    try {
      await this.kv.delete(scope, key);
    } catch (err) {
      result = "failed";
      error = errorMessage(err);
    }
    await this.auditIndexPersistence("delete", [statePath(scope, key)], {
      scope,
      key,
      reason,
      result,
      error,
    });
  }

  private async deleteShards(
    shards: IndexShardManifest["shards"],
    reason: string,
  ): Promise<void> {
    for (const shard of shards) {
      await this.deleteKey(shard.scope, shard.key, reason);
    }
  }

  private async isManifestPublished(
    manifestKey: string,
    expected: IndexShardManifest,
  ): Promise<boolean> {
    const published = await this.kv
      .get<IndexShardManifest>(KV.bm25Index, manifestKey)
      .catch(() => null);
    if (
      published?.v !== 1 ||
      published.generation !== expected.generation ||
      published.chars !== expected.chars ||
      !Array.isArray(published.shards) ||
      published.shards.length !== expected.shards.length
    ) {
      return false;
    }
    return published.shards.every((shard, index) => {
      const expectedShard = expected.shards[index];
      if (!expectedShard) return false;
      return (
        shard.scope === expectedShard.scope &&
        shard.key === expectedShard.key &&
        shard.chars === expectedShard.chars
      );
    });
  }

  private async loadBm25Data(): Promise<string | null> {
    return this.loadShardedData(BM25_KEY, BM25_MANIFEST_KEY, "BM25");
  }

  private async loadVectorData(): Promise<string | null> {
    return this.loadShardedData(VECTOR_KEY, VECTOR_MANIFEST_KEY, "vector");
  }

  private async loadShardedData(
    legacyKey: string,
    manifestKey: string,
    label: string,
  ): Promise<string | null> {
    const manifest = await this.readIndexValue<IndexShardManifest>(
      KV.bm25Index,
      manifestKey,
      label,
      "manifest",
    );
    if (!manifest.ok) return null;
    // #797: some iii-state adapters return `undefined` (not `null`) for
    // a missing key. The previous `value !== null` check passed
    // undefined through to loadManifestData, which then crashed on
    // `manifest.v` with TypeError. Treat both null and undefined as
    // "no manifest" and fall through to the legacy path. The shape
    // check stays so a malformed-but-present row still fails closed.
    if (
      manifest.value != null &&
      typeof manifest.value === "object"
    ) {
      return this.loadManifestData(manifest.value, label);
    }

    const legacy = await this.readIndexValue<string>(
      KV.bm25Index,
      legacyKey,
      label,
      "legacy",
    );
    if (!legacy.ok) return null;
    if (legacy.value && typeof legacy.value === "string") return legacy.value;
    return null;
  }

  private async readIndexValue<T>(
    scope: string,
    key: string,
    label: string,
    source: "manifest" | "legacy",
  ): Promise<{ ok: true; value: T | null } | { ok: false }> {
    try {
      return { ok: true, value: await this.kv.get<T>(scope, key) };
    } catch (err) {
      logger.warn(`index persistence: ${label} ${source} read failed`, {
        scope,
        key,
        message: errorMessage(err),
      });
      return { ok: false };
    }
  }

  private async loadManifestData(
    manifest: IndexShardManifest,
    label: string,
  ): Promise<string | null> {
    if (
      manifest.v !== 1 ||
      !Array.isArray(manifest.shards) ||
      manifest.shards.length === 0 ||
      !Number.isInteger(manifest.chars) ||
      manifest.chars < 0
    ) {
      logger.warn(`index persistence: ${label} shard manifest invalid`);
      return null;
    }
    for (const shard of manifest.shards) {
      if (!isValidShardDescriptor(shard)) {
        logger.warn(`index persistence: ${label} shard manifest invalid`);
        return null;
      }
    }
    const loadedShards = await Promise.all(
      manifest.shards.map(async (shard) => ({
        shard,
        chunk: await this.kv.get<string>(shard.scope, shard.key).catch(() => null),
      })),
    );
    const chunks: string[] = [];
    let chars = 0;
    for (const { shard, chunk } of loadedShards) {
      if (typeof chunk !== "string") {
        logger.warn(`index persistence: ${label} shard missing`, {
          scope: shard.scope,
          key: shard.key,
        });
        return null;
      }
      if (chunk.length !== shard.chars) {
        logger.warn(`index persistence: ${label} shard length mismatch`, {
          scope: shard.scope,
          key: shard.key,
          expected: shard.chars,
          actual: chunk.length,
        });
        return null;
      }
      chunks.push(chunk);
      chars += chunk.length;
    }
    if (chars !== manifest.chars) {
      logger.warn(`index persistence: ${label} total length mismatch`, {
        expected: manifest.chars,
        actual: chars,
      });
      return null;
    }
    return chunks.join("");
  }
}
