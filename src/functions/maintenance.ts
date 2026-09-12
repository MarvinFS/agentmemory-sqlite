// Maintenance functions for the inproc engine (plan steps 9.7 and 10).
//
// `mem::backup` (POST /agentmemory/backup): a `VACUUM INTO` snapshot of the
// live SQLite file written under a unique name, opened read-only and checked
// (integrity_check, row counts against the live DB), then renamed over
// `agentmemory-snapshot.sqlite`. Any failure leaves no partial file and
// answers 500, so the backup job that calls it fails loudly instead of
// tarring yesterday's copy.
//
// `mem::index-debug-legs` (POST /agentmemory/index-debug-legs): the raw BM25
// and vector legs for a query, full scored lists, for the acceptance
// harness. Registered only with AGENTMEMORY_INDEX_DEBUG=1 and meant to be
// switched off again after acceptance.
//
// Both routes run behind the daemon's `middleware::api-auth`.
import { existsSync, renameSync, statSync, unlinkSync } from "node:fs";
import { dirname, join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import type { ApiRequest, ISdk } from "iii-sdk";

import { getSqlitePath } from "../config.js";
import type { SqliteState } from "../engine/inproc/state.js";
import { logger } from "../logger.js";
import { getEmbeddingProvider, getSearchIndex, getVectorIndex } from "./search.js";

export type SnapshotResult = {
  path: string;
  bytes: number;
  ms: number;
  kvRows: number;
  vectorRows: number;
  integrity: string;
};

const SNAPSHOT_NAME = "agentmemory-snapshot.sqlite";

const count = (db: DatabaseSync, table: string): number =>
  (db.prepare(`SELECT count(*) AS n FROM ${table}`).get() as { n: number }).n;

export function snapshotDatabase(db: DatabaseSync, dir: string): SnapshotResult {
  const t0 = performance.now();
  const final = join(dir, SNAPSHOT_NAME);
  // VACUUM INTO refuses an existing non-empty file, hence the unique name.
  const tmp = join(dir, `agentmemory-snapshot.${Date.now()}.${process.pid}.sqlite`);
  const sidecars = [`${tmp}-wal`, `${tmp}-shm`];
  try {
    db.prepare("VACUUM INTO ?").run(tmp);
    // Nothing else runs on this thread between the VACUUM and these counts,
    // so the snapshot must hold exactly what the live DB holds.
    const liveKv = count(db, "kv");
    const liveVectors = count(db, "vectors");
    const ro = new DatabaseSync(tmp, { readOnly: true });
    let integrity: string;
    let kvRows: number;
    let vectorRows: number;
    try {
      const rows = ro.prepare("PRAGMA integrity_check").all() as Array<{ integrity_check: string }>;
      integrity = rows.map((r) => r.integrity_check).join("; ");
      kvRows = count(ro, "kv");
      vectorRows = count(ro, "vectors");
    } finally {
      ro.close();
    }
    for (const f of sidecars) if (existsSync(f)) unlinkSync(f);
    if (integrity !== "ok") throw new Error(`snapshot integrity_check: ${integrity}`);
    if (kvRows !== liveKv || vectorRows !== liveVectors) {
      throw new Error(`snapshot holds ${kvRows} kv / ${vectorRows} vector rows, live DB ${liveKv} / ${liveVectors}`);
    }
    renameSync(tmp, final);
    return { path: final, bytes: statSync(final).size, ms: Math.round(performance.now() - t0), kvRows, vectorRows, integrity };
  } catch (err) {
    for (const f of [tmp, ...sidecars]) if (existsSync(f)) unlinkSync(f);
    throw err;
  }
}

export type DebugLegs = {
  bm25Docs: number;
  vectorRows: number;
  bm25: Array<{ obsId: string; sessionId: string; score: number }>;
  vector: Array<{ obsId: string; sessionId: string; score: number }> | null;
  // The query embedding when this call computed it (so a caller can reuse
  // it against another index); null when the caller supplied one.
  embedding: number[] | null;
};

export async function indexDebugLegs(input: { query?: unknown; depth?: unknown; embedding?: unknown }): Promise<DebugLegs> {
  if (typeof input.query !== "string" || !input.query.trim()) throw new Error("query must be a non-empty string");
  const depth = Number.isInteger(input.depth) && (input.depth as number) > 0 ? Math.min(input.depth as number, 100_000) : 200;
  const bm25 = getSearchIndex();
  const vi = getVectorIndex();
  let embedding: Float32Array | null = null;
  let computed = false;
  if (input.embedding !== undefined) {
    if (!Array.isArray(input.embedding) || !input.embedding.every((n) => typeof n === "number" && Number.isFinite(n))) {
      throw new Error("embedding must be an array of finite numbers");
    }
    // cosineSimilarity returns 0 on a length mismatch: a wrong-sized query
    // vector would produce an all-zero leg that compares equal to another.
    const dims = vi?.dims ?? null;
    if (dims !== null && input.embedding.length !== dims) {
      throw new Error(`embedding has ${input.embedding.length} dimensions, the index has ${dims}`);
    }
    embedding = Float32Array.from(input.embedding as number[]);
  } else if (vi && vi.size > 0) {
    const ep = getEmbeddingProvider();
    if (ep) {
      embedding = await ep.embed(input.query);
      computed = true;
    }
  }
  return {
    bm25Docs: bm25.size,
    vectorRows: vi?.size ?? 0,
    bm25: bm25.search(input.query, depth),
    vector: vi && embedding ? vi.search(embedding, depth) : null,
    embedding: computed && embedding ? Array.from(embedding) : null,
  };
}

type Response = { status_code: number; body: unknown };
const message = (err: unknown): string => (err instanceof Error ? err.message : String(err));

export function registerMaintenanceFunctions(sdk: ISdk, state: SqliteState): void {
  const route = (id: string, path: string, handler: (req: ApiRequest<any>) => Promise<Response>) => {
    sdk.registerFunction(id, handler);
    sdk.registerTrigger({
      type: "http",
      function_id: id,
      config: { api_path: path, http_method: "POST", middleware_function_ids: ["middleware::api-auth"] },
    });
  };

  route("mem::backup", "/agentmemory/backup", async (req: ApiRequest<{ dir?: string }>) => {
    const dir = typeof req.body?.dir === "string" && req.body.dir ? req.body.dir : dirname(getSqlitePath());
    try {
      const r = snapshotDatabase(state.db, dir);
      logger.info("backup snapshot written", r);
      return { status_code: 200, body: r };
    } catch (err) {
      logger.error("backup snapshot FAILED", { dir, error: message(err) });
      return { status_code: 500, body: { error: message(err) } };
    }
  });

  if (process.env["AGENTMEMORY_INDEX_DEBUG"] === "1") {
    route("mem::index-debug-legs", "/agentmemory/index-debug-legs", async (req) => {
      try {
        return { status_code: 200, body: await indexDebugLegs((req.body ?? {}) as Record<string, unknown>) };
      } catch (err) {
        return { status_code: 400, body: { error: message(err) } };
      }
    });
    logger.warn("mem::index-debug-legs enabled (AGENTMEMORY_INDEX_DEBUG=1) - remove after acceptance");
  }
}
