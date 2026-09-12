import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { SqliteState } from "../src/engine/inproc/state.js";
import { SqliteVectorStore } from "../src/engine/inproc/vectors.js";
import { VectorIndex } from "../src/state/vector-index.js";
import { KV } from "../src/state/schema.js";
import { enumerateIndexCorpus } from "../src/state/index-corpus.js";
import {
  getSearchIndex,
  isMemoryIndexReady,
  rebuildBm25FromContent,
  setInprocStores,
  setVectorIndex,
} from "../src/functions/search.js";
import type { Memory, CompressedObservation } from "../src/types.js";

const v = (...xs: number[]) => new Float32Array(xs);

function memory(id: string, content: string, extra: Partial<Memory> = {}): Memory {
  return {
    id,
    createdAt: "2026-09-07T00:00:00.000Z",
    updatedAt: "2026-09-07T00:00:00.000Z",
    type: "fact",
    title: content.slice(0, 20),
    content,
    concepts: [],
    files: [],
    sessionIds: [],
    strength: 7,
    version: 1,
    isLatest: true,
    ...extra,
  };
}

function observation(id: string, sessionId: string, narrative: string, extra: Partial<CompressedObservation> = {}): CompressedObservation {
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
    ...extra,
  };
}

describe("inproc BM25 rebuild from content", () => {
  let dir: string;
  let state: SqliteState;
  const savedEnv = { ...process.env };

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "am-inproc-bm25-"));
    state = new SqliteState(join(dir, "state.sqlite"));
    process.env.AGENTMEMORY_INDEX_EXCLUDE_TOOL_PREFIXES = "mcp__plugin_agentmemory";
  });

  afterEach(() => {
    getSearchIndex().clear();
    setInprocStores(null, null);
    setVectorIndex(null);
    state.close();
    rmSync(dir, { recursive: true, force: true });
    process.env = { ...savedEnv };
  });

  function seed() {
    // Interleaved across scopes so seq order is not scope order.
    state.set(KV.observations("ses_live"), "obs_1", observation("obs_1", "ses_live", "kept, session exists"));
    state.set(KV.memories, "mem_1", memory("mem_1", "latest memory about zebras"));
    state.set(KV.observations("ses_gone"), "obs_2", observation("obs_2", "ses_gone", "kept, session record evicted"));
    state.set(KV.memories, "mem_old", memory("mem_old", "superseded memory", { isLatest: false }));
    state.set(KV.memories, "mem_blank", memory("mem_blank", "", { title: "" }));
    state.set(KV.observations("ses_live"), "obs_echo", observation("obs_echo", "ses_live", "zebras", { toolName: "mcp__plugin_agentmemory_agentmemory__memory_recall" }));
    state.set(KV.observations("ses_live"), "obs_raw", observation("obs_raw", "ses_live", ""));
    state.set(KV.sessions, "ses_live", { id: "ses_live", project: "p", cwd: "/", startedAt: "", status: "active", observationCount: 2 });
    state.set(KV.compressPending, "obs_9", { observationId: "obs_9", raw: { title: "zebras pending" } });
    state.set(KV.summaries, "ses_live", { title: "zebras summary" });
    state.set(KV.memories, "mem_2", memory("mem_2", "second latest memory about lions"));
  }

  it("enumerates memories and every mem:obs:* scope in seq order with one eligibility rule", () => {
    seed();
    const items = Array.from(enumerateIndexCorpus(state.db));
    expect(items.map((i) => i.key)).toEqual(["obs_1", "mem_1", "obs_2", "mem_old", "mem_blank", "obs_echo", "obs_raw", "mem_2"]);
    expect(items.filter((i) => i.doc).map((i) => i.key)).toEqual(["obs_1", "mem_1", "obs_2", "mem_2"]);
    for (const i of items) expect(i.jobs.length > 0).toBe(i.doc !== null);
    const mem1 = items.find((i) => i.key === "mem_1")!;
    expect(mem1.kind).toBe("memory");
    expect(mem1.doc!.sessionId).toBe("memory");
    expect(mem1.jobs[0]).toMatchObject({ id: "mem_1", sessionId: "memory", kind: "memory" });
  });

  it("rebuilds BM25 from content and never touches the vector index", async () => {
    seed();
    const vectors = new SqliteVectorStore(state);
    const vi = new VectorIndex();
    vi.attachStore(vectors);
    vi.add("mem_1", "memory", v(1, 0), "h");
    vi.add("stale_orphan", "x", v(0, 1), "h");
    setVectorIndex(vi);
    setInprocStores(state, vectors);

    const idx = getSearchIndex();
    idx.add(observation("leftover", "s", "from a previous life"));
    const r = await rebuildBm25FromContent(state.db);

    expect(r.rows).toBe(8);
    expect(r.docs).toBe(4);
    expect(idx.size).toBe(4);
    expect(["obs_1", "mem_1", "obs_2", "mem_2"].every((id) => idx.has(id))).toBe(true);
    expect(["leftover", "mem_old", "mem_blank", "obs_echo", "obs_raw", "obs_9"].some((id) => idx.has(id))).toBe(false);
    expect(idx.search("zebras", 10).map((h) => h.obsId)).toEqual(["mem_1"]);
    expect(isMemoryIndexReady()).toBe(true);
    // Vectors: same two entries, same rows, nothing re-embedded or cleared.
    expect(vi.size).toBe(2);
    expect(vectors.count()).toBe(2);
    expect(r.readMs).toBeGreaterThanOrEqual(0);
    expect(r.indexMs).toBeGreaterThanOrEqual(0);
  });

  it("indexes an observation that lacks the array fields instead of failing the boot", async () => {
    seed();
    const bare = { id: "obs_bare", sessionId: "s", timestamp: "", type: "decision", title: "bare zebras", narrative: "no arrays here" };
    state.set(KV.observations("s"), "obs_bare", bare);
    state.set(KV.observations("s"), "obs_odd", { ...bare, id: "obs_odd", files: 5, facts: null });
    // Non-string fields where strings are expected: not indexable, not fatal.
    state.set(KV.observations("s"), "obs_objtitle", { ...bare, id: "obs_objtitle", title: { toString: 0 } });
    state.set(KV.observations("s"), "obs_numtool", { ...bare, id: "obs_numtool", toolName: 7 });
    state.set(KV.memories, "mem_objtitle", memory("mem_objtitle", "content", { title: { toString: 0 } as never }));
    const items = Array.from(enumerateIndexCorpus(state.db));
    expect(items.find((i) => i.key === "obs_objtitle")).toMatchObject({ doc: null, jobs: [] });
    expect(items.find((i) => i.key === "mem_objtitle")).toMatchObject({ doc: null, jobs: [] });
    expect(items.find((i) => i.key === "obs_numtool")!.doc).not.toBeNull();
    const r = await rebuildBm25FromContent(state.db);
    expect(r).toMatchObject({ docs: 7, rows: 13, skipped: 0 });
    expect(getSearchIndex().has("obs_objtitle")).toBe(false);
    expect(getSearchIndex().has("mem_objtitle")).toBe(false);
    expect(getSearchIndex().has("obs_bare")).toBe(true);
    expect(getSearchIndex().has("obs_odd")).toBe(true);
    expect(getSearchIndex().search("zebras", 10).map((h) => h.obsId).sort()).toEqual(["mem_1", "obs_bare", "obs_numtool", "obs_odd"]);
  });

  it("is idempotent and yields an empty index on an empty store", async () => {
    const r = await rebuildBm25FromContent(state.db);
    expect(r).toMatchObject({ docs: 0, rows: 0 });
    expect(getSearchIndex().size).toBe(0);
    seed();
    await rebuildBm25FromContent(state.db);
    await rebuildBm25FromContent(state.db);
    expect(getSearchIndex().size).toBe(4);
  });
});
