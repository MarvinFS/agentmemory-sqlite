import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { existsSync, mkdtempSync, readdirSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";

import { createInprocSdk, type InprocSdk } from "../src/engine/inproc/sdk.js";
import { SqliteVectorStore } from "../src/engine/inproc/vectors.js";
import { VectorIndex } from "../src/state/vector-index.js";
import { KV } from "../src/state/schema.js";
import { registerMaintenanceFunctions, snapshotDatabase } from "../src/functions/maintenance.js";
import { getSearchIndex, setEmbeddingProvider, setVectorIndex } from "../src/functions/search.js";
import type { CompressedObservation, EmbeddingProvider } from "../src/types.js";

const SECRET = "s3cret";
const v = (...xs: number[]) => new Float32Array(xs);

function observation(id: string, sessionId: string, narrative: string): CompressedObservation {
  return {
    id,
    sessionId,
    timestamp: "2026-09-07T00:00:00.000Z",
    type: "decision",
    title: "title " + id,
    facts: [],
    narrative,
    concepts: [],
    files: [],
    importance: 5,
  };
}

describe("inproc maintenance functions", () => {
  let dir: string;
  let sdk: InprocSdk;
  let base: string;
  let vi: VectorIndex;

  const call = (path: string, body: unknown, token: string | null = SECRET) =>
    fetch(base + path, {
      method: "POST",
      headers: { "content-type": "application/json", ...(token ? { authorization: `Bearer ${token}` } : {}) },
      body: JSON.stringify(body),
    });

  beforeEach(async () => {
    dir = mkdtempSync(join(tmpdir(), "am-maint-"));
    process.env.AGENTMEMORY_INDEX_DEBUG = "1";
    sdk = createInprocSdk({ restPort: 0, streamsPort: 0, sqlitePath: join(dir, "state.sqlite") });
    await sdk.listening();
    base = `http://127.0.0.1:${sdk.ports().rest}`;
    sdk.registerFunction("middleware::api-auth", async (input: any) => {
      const auth = input?.request?.headers?.["authorization"];
      return auth === `Bearer ${SECRET}` ? { action: "continue" } : { action: "respond", response: { status_code: 401, body: { error: "unauthorized" } } };
    });
    registerMaintenanceFunctions(sdk, sdk.store);
    sdk.setReady();

    const vectors = new SqliteVectorStore(sdk.store);
    vi = new VectorIndex();
    vi.attachStore(vectors);
    setVectorIndex(vi);
    const provider: EmbeddingProvider = {
      name: "fake",
      dimensions: 2,
      async embed() {
        return v(1, 0);
      },
      async embedBatch(ts: string[]) {
        return ts.map(() => v(1, 0));
      },
    };
    setEmbeddingProvider(provider);
    getSearchIndex().clear();
    for (const [id, text, emb] of [
      ["obs_1", "zebras graze at dawn", v(1, 0)],
      ["obs_2", "zebras and lions", v(0.9, 0.1)],
      ["obs_3", "quarterly budget", v(0, 1)],
    ] as Array<[string, string, Float32Array]>) {
      const obs = observation(id, "s1", text);
      sdk.store.set(KV.observations("s1"), id, obs);
      getSearchIndex().add(obs);
      vi.add(id, "s1", emb);
    }
  });

  afterEach(async () => {
    delete process.env.AGENTMEMORY_INDEX_DEBUG;
    setVectorIndex(null);
    setEmbeddingProvider(null);
    getSearchIndex().clear();
    await sdk.shutdown();
    rmSync(dir, { recursive: true, force: true });
  });

  it("mem::backup writes a verified snapshot that opens with the live counts, atomically renamed", async () => {
    const res = await call("/agentmemory/backup", { dir });
    expect(res.status).toBe(200);
    const r = (await res.json()) as { path: string; kvRows: number; vectorRows: number; integrity: string; bytes: number };
    expect(r).toMatchObject({ path: join(dir, "agentmemory-snapshot.sqlite"), kvRows: 3, vectorRows: 3, integrity: "ok" });
    expect(r.bytes).toBeGreaterThan(0);
    const snap = new DatabaseSync(r.path, { readOnly: true });
    try {
      expect((snap.prepare("SELECT count(*) AS n FROM kv WHERE scope = ?").get(KV.observations("s1")) as { n: number }).n).toBe(3);
      expect((snap.prepare("SELECT count(*) AS n FROM vectors").get() as { n: number }).n).toBe(3);
    } finally {
      snap.close();
    }
    // Only the snapshot (and the live store) are left behind: no unique-name temp, no sidecars of it.
    expect(readdirSync(dir).filter((f) => f.startsWith("agentmemory-snapshot.") && f !== "agentmemory-snapshot.sqlite" && !f.endsWith("-wal") && !f.endsWith("-shm"))).toEqual([]);
    // A second run replaces the first in place.
    sdk.store.set(KV.observations("s1"), "obs_4", observation("obs_4", "s1", "later"));
    const again = (await (await call("/agentmemory/backup", { dir })).json()) as { kvRows: number };
    expect(again.kvRows).toBe(4);
  });

  it("mem::backup fails loudly and leaves nothing behind when the snapshot cannot be written", async () => {
    const bad = join(dir, "missing", "deeper");
    const res = await call("/agentmemory/backup", { dir: bad });
    expect(res.status).toBe(500);
    expect(((await res.json()) as { error: string }).error).toMatch(/./);
    expect(existsSync(bad)).toBe(false);
    expect(() => snapshotDatabase(sdk.store.db, bad)).toThrow();
  });

  it("both routes are behind the auth middleware", async () => {
    expect((await call("/agentmemory/backup", { dir }, null)).status).toBe(401);
    expect((await call("/agentmemory/backup", { dir }, "wrong")).status).toBe(401);
    expect((await call("/agentmemory/index-debug-legs", { query: "zebras" }, null)).status).toBe(401);
  });

  it("mem::index-debug-legs returns the raw legs exactly as the indexes compute them", async () => {
    const res = await call("/agentmemory/index-debug-legs", { query: "zebras", depth: 10, embedding: [1, 0] });
    expect(res.status).toBe(200);
    const legs = (await res.json()) as { bm25Docs: number; vectorRows: number; bm25: unknown[]; vector: unknown[]; embedding: number[] | null };
    expect(legs.bm25Docs).toBe(3);
    expect(legs.vectorRows).toBe(3);
    expect(legs.bm25).toEqual(getSearchIndex().search("zebras", 10));
    expect(legs.vector).toEqual(vi.search(v(1, 0), 10));
    expect(legs.vector.map((r: any) => r.obsId)).toEqual(["obs_1", "obs_2", "obs_3"]);
    expect(legs.embedding).toBeNull();
    // Without a supplied embedding the provider embeds and the vector is returned for reuse.
    const own = (await (await call("/agentmemory/index-debug-legs", { query: "zebras" })).json()) as { embedding: number[] | null };
    expect(own.embedding).toEqual([1, 0]);
    expect((await call("/agentmemory/index-debug-legs", { query: "" })).status).toBe(400);
    expect((await call("/agentmemory/index-debug-legs", { query: "x", embedding: ["a"] })).status).toBe(400);
    // A query vector of another length scores 0 against every row (a zero
    // leg that would compare equal): refused, as is an empty one.
    expect((await call("/agentmemory/index-debug-legs", { query: "x", embedding: [] })).status).toBe(400);
    expect((await call("/agentmemory/index-debug-legs", { query: "x", embedding: [1, 0, 0] })).status).toBe(400);
  });

  it("mem::index-debug-legs is not registered without AGENTMEMORY_INDEX_DEBUG=1", async () => {
    await sdk.shutdown();
    delete process.env.AGENTMEMORY_INDEX_DEBUG;
    sdk = createInprocSdk({ restPort: 0, streamsPort: 0, sqlitePath: join(dir, "state2.sqlite") });
    await sdk.listening();
    base = `http://127.0.0.1:${sdk.ports().rest}`;
    sdk.registerFunction("middleware::api-auth", async () => ({ action: "continue" }));
    new SqliteVectorStore(sdk.store); // the daemon always creates the vectors table
    registerMaintenanceFunctions(sdk, sdk.store);
    sdk.setReady();
    expect((await call("/agentmemory/index-debug-legs", { query: "zebras" })).status).toBe(404);
    expect((await call("/agentmemory/backup", { dir })).status).toBe(200);
  });
});
