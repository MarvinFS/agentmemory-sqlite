import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { SqliteState } from "../src/engine/inproc/state.js";
import { SqliteVectorStore } from "../src/engine/inproc/vectors.js";
import { VectorIndex } from "../src/state/vector-index.js";
import { SearchIndex } from "../src/state/search-index.js";
import { KV } from "../src/state/schema.js";
import { embedInputHash, memoryEmbedJobs } from "../src/state/index-corpus.js";
import {
  deleteIndexed,
  getSearchIndex,
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

function observation(id: string, sessionId: string, narrative: string): CompressedObservation {
  return {
    id,
    sessionId,
    timestamp: "2026-09-07T00:00:00.000Z",
    type: "decision",
    title: "t",
    facts: [],
    narrative,
    concepts: [],
    files: [],
    importance: 5,
  };
}

describe("inproc vector store", () => {
  let dir: string;
  let state: SqliteState;
  let store: SqliteVectorStore;
  let vi: VectorIndex;
  const savedEnv = { ...process.env };

  const rows = () =>
    state.db
      .prepare("SELECT id, session_id, dims, length(embedding) AS bytes, input_hash, hash_state FROM vectors ORDER BY seq")
      .all() as Array<{ id: string; session_id: string; dims: number; bytes: number; input_hash: string; hash_state: string }>;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "am-inproc-vec-"));
    state = new SqliteState(join(dir, "state.sqlite"));
    store = new SqliteVectorStore(state);
    vi = new VectorIndex();
    vi.attachStore(store);
    delete process.env.AGENTMEMORY_MEMORY_CHUNKING;
    delete process.env.AGENTMEMORY_CHUNK_MAX_CHARS;
  });

  afterEach(() => {
    setInprocStores(null, null);
    setVectorIndex(null);
    state.close();
    rmSync(dir, { recursive: true, force: true });
    process.env = { ...savedEnv };
  });

  it("commits an embedding only when the content row still calls for that text", () => {
    const mem = memory("mem_a", "alpha content");
    state.set(KV.memories, "mem_a", mem);
    const text = mem.title + " " + mem.content;

    expect(store.commitEmbedding(vi, { id: "mem_a", sessionId: "memory", text, kind: "memory" }, v(1, 0))).toBe(true);
    expect(vi.size).toBe(1);
    expect(rows()).toEqual([
      { id: "mem_a", session_id: "memory", dims: 2, bytes: 8, input_hash: embedInputHash(text), hash_state: "verified" },
    ]);

    // Content rewritten after the embed was requested: the old completion is stale.
    state.set(KV.memories, "mem_a", memory("mem_a", "alpha content, revised"));
    expect(store.commitEmbedding(vi, { id: "mem_a", sessionId: "memory", text, kind: "memory" }, v(0, 1))).toBe(false);
    expect(rows()[0].input_hash).toBe(embedInputHash(text));
    expect(vi.search(v(1, 0), 1)[0].score).toBeCloseTo(1, 6);

    // Superseded or deleted: nothing is written.
    state.set(KV.memories, "mem_a", memory("mem_a", "alpha content, revised", { isLatest: false }));
    const revised = "alpha content, revis alpha content, revised";
    expect(store.commitEmbedding(vi, { id: "mem_a", sessionId: "memory", text: revised, kind: "memory" }, v(0, 1))).toBe(false);
    state.delete(KV.memories, "mem_a");
    expect(store.commitEmbedding(vi, { id: "mem_a", sessionId: "memory", text: revised, kind: "memory" }, v(0, 1))).toBe(false);
    expect(rows()).toHaveLength(1);
  });

  it("resolves observations and synthetics through the session scope", () => {
    const obs = observation("obs_1", "ses_1", "narrative one");
    state.set(KV.observations("ses_1"), "obs_1", obs);
    const text = obs.title + " " + obs.narrative;
    expect(store.commitEmbedding(vi, { id: "obs_1", sessionId: "ses_1", text, kind: "synthetic" }, v(1, 1))).toBe(true);
    // Wrong session: the row is not where the job says it is.
    expect(store.commitEmbedding(vi, { id: "obs_1", sessionId: "ses_2", text, kind: "observation" }, v(1, 1))).toBe(false);
    // Excluded tool: stored, never indexed.
    process.env.AGENTMEMORY_INDEX_EXCLUDE_TOOL_PREFIXES = "mcp__plugin_agentmemory";
    state.set(KV.observations("ses_1"), "obs_2", { ...observation("obs_2", "ses_1", "echo"), toolName: "mcp__plugin_agentmemory_agentmemory__memory_recall" });
    expect(store.commitEmbedding(vi, { id: "obs_2", sessionId: "ses_1", text: "t echo", kind: "observation" }, v(1, 1))).toBe(false);
    expect(rows().map((r) => r.id)).toEqual(["obs_1"]);
  });

  it("updates the in-memory map only after the enclosing transaction commits, never after a rollback", () => {
    state.transaction(() => {
      vi.add("mem_x", "memory", v(1, 0), "h");
      expect(vi.size).toBe(0);
    });
    expect(vi.size).toBe(1);

    expect(() =>
      state.transaction(() => {
        vi.add("mem_y", "memory", v(0, 1), "h");
        vi.remove("mem_x");
        throw new Error("boom");
      }),
    ).toThrow("boom");
    expect(vi.size).toBe(1);
    expect(vi.search(v(1, 0), 5).map((r) => r.obsId)).toEqual(["mem_x"]);
    expect(rows().map((r) => r.id)).toEqual(["mem_x"]);
  });

  it("persists chunk rows, hydrates parent-level search and parent deletion after a restart", () => {
    process.env.AGENTMEMORY_MEMORY_CHUNKING = "true";
    process.env.AGENTMEMORY_CHUNK_MAX_CHARS = "40";
    process.env.AGENTMEMORY_CHUNK_OVERLAP_CHARS = "0";
    const mem = memory("mem_c", "first sentence about apples and pears. second sentence about ships and sails. third about mountains.");
    state.set(KV.memories, "mem_c", mem);
    const jobs = memoryEmbedJobs(mem);
    expect(jobs.length).toBeGreaterThan(1);
    jobs.forEach((job, i) => {
      expect(store.commitEmbedding(vi, job, i === 1 ? v(1, 0) : v(0, 1))).toBe(true);
    });
    vi.add("obs_solo", "ses_1", v(0.7, 0.7), "h");
    expect(rows().map((r) => r.id)).toEqual([...jobs.map((j) => j.id), "obs_solo"]);

    // Restart: a fresh index filled from rows, in seq order, without writes.
    const restarted = new VectorIndex();
    restarted.attachStore(store);
    expect(store.hydrate(restarted)).toBe(jobs.length + 1);
    expect(restarted.size).toBe(jobs.length + 1);
    const hits = restarted.search(v(1, 0), 10);
    expect(hits.map((h) => h.obsId)).toEqual(["mem_c", "obs_solo"]);
    expect(hits[0].score).toBeCloseTo(1, 6);

    // Parent delete after restart drops every chunk row and map entry.
    restarted.remove("mem_c");
    expect(restarted.size).toBe(1);
    expect(rows().map((r) => r.id)).toEqual(["obs_solo"]);

    // Re-adding the parent as a single vector drops stale chunk rows too.
    store.put({ id: "mem_c#0", sessionId: "memory", embedding: v(1, 0), inputHash: "" });
    restarted.add("mem_c", "memory", v(1, 0), "h");
    expect(rows().map((r) => r.id)).toEqual(["obs_solo", "mem_c"]);

    restarted.clear();
    expect(rows()).toEqual([]);
    expect(restarted.size).toBe(0);
  });

  it("keeps a row's seq on re-embed and refuses a blob that is not dims*4", () => {
    vi.add("a", "s", v(1, 0), "h1");
    vi.add("b", "s", v(0, 1), "h2");
    vi.add("a", "s", v(0.5, 0.5), "h3");
    expect(rows().map((r) => [r.id, r.input_hash])).toEqual([["a", "h3"], ["b", "h2"]]);

    state.db.exec("PRAGMA ignore_check_constraints = ON");
    state.db
      .prepare("INSERT INTO vectors (id, session_id, dims, embedding, input_hash, hash_state) VALUES ('bad', 's', 2, x'0000', '', 'legacy')")
      .run();
    state.db.exec("PRAGMA ignore_check_constraints = OFF");
    expect(() => store.hydrate(new VectorIndex())).toThrow(/bad.*2 bytes/);
    expect(() =>
      state.db
        .prepare("INSERT INTO vectors (id, session_id, dims, embedding, input_hash, hash_state) VALUES ('bad2', 's', 2, x'0000', '', 'legacy')")
        .run(),
    ).toThrow(/CHECK/);
  });

  it("deleteIndexed removes content, access log, vectors and the BM25 doc together", async () => {
    setInprocStores(state, store);
    setVectorIndex(vi);
    const bm25 = getSearchIndex();
    const mem = memory("mem_d", "delete me please");
    state.set(KV.memories, "mem_d", mem);
    state.set(KV.accessLog, "mem_d", { count: 3 });
    bm25.add({ ...observation("mem_d", "memory", mem.content), title: mem.title });
    expect(store.commitEmbedding(vi, { id: "mem_d", sessionId: "memory", text: mem.title + " " + mem.content, kind: "memory" }, v(1, 0))).toBe(true);

    await deleteIndexed({} as never, KV.memories, "mem_d");

    expect(state.get(KV.memories, "mem_d")).toBeNull();
    expect(state.get(KV.accessLog, "mem_d")).toBeNull();
    expect(rows()).toEqual([]);
    expect(vi.size).toBe(0);
    expect(bm25.has("mem_d")).toBe(false);
    // Idempotent on a missing id.
    await deleteIndexed({} as never, KV.memories, "mem_d");
  });

  it("deleteIndexed takes an observation's access log and works without a vector map", async () => {
    setInprocStores(state, store);
    setVectorIndex(vi);
    state.set(KV.observations("s1"), "obs_d", observation("obs_d", "s1", "searched then forgotten"));
    state.set(KV.accessLog, "obs_d", { count: 3 });
    vi.add("obs_d", "s1", v(1, 0), "h");
    await deleteIndexed({} as never, KV.observations("s1"), "obs_d");
    expect(state.get(KV.accessLog, "obs_d")).toBeNull();
    expect(rows()).toEqual([]);

    // No embedding provider: no VectorIndex, but the persisted rows still go.
    setVectorIndex(null);
    state.set(KV.memories, "mem_n", memory("mem_n", "no provider"));
    state.db
      .prepare("INSERT INTO vectors (id, session_id, dims, embedding, input_hash, hash_state) VALUES ('mem_n', 'memory', 2, ?, '', 'legacy')")
      .run(Buffer.from(v(1, 1).buffer));
    await deleteIndexed({} as never, KV.memories, "mem_n");
    expect(state.get(KV.memories, "mem_n")).toBeNull();
    expect(rows()).toEqual([]);
  });

  it("applies map mutations in commit order when a post-commit callback opens its own transaction", () => {
    state.transaction(() => {
      state.afterCommit(() => state.transaction(() => vi.remove("x")));
      vi.add("x", "s", v(1, 0), "h");
    });
    expect(rows()).toEqual([]);
    expect(vi.size).toBe(0);
    // A rollback inside a callback drops only its own callbacks.
    state.transaction(() => {
      state.afterCommit(() => {
        try {
          state.transaction(() => {
            vi.add("y", "s", v(1, 0), "h");
            throw new Error("boom");
          });
        } catch {}
      });
      vi.add("z", "s", v(0, 1), "h");
    });
    expect(rows().map((r) => r.id)).toEqual(["z"]);
    expect(vi.size).toBe(1);

    // A callback that mutates OUTSIDE a transaction while the queue drains
    // still lands behind the work committed before it.
    state.transaction(() => {
      state.afterCommit(() => vi.remove("w"));
      vi.add("w", "s", v(1, 0), "h");
    });
    expect(rows().map((r) => r.id)).toEqual(["z"]);
    expect(vi.size).toBe(1);

    // Same for state events: delivered in commit order.
    const seen: number[] = [];
    state.watchScope(KV.memories);
    state.onEvent((e) => seen.push((e.new_value as { n: number }).n));
    state.transaction(() => {
      state.afterCommit(() => state.set(KV.memories, "k", { n: 2 }));
      state.set(KV.memories, "k", { n: 1 });
    });
    expect(seen).toEqual([1, 2]);
  });

  it("exact removal drops one row and leaves the parent's other chunks", () => {
    vi.add("p#0", "s", v(1, 0), "h");
    vi.add("p#1", "s", v(0, 1), "h");
    vi.add("q", "s", v(1, 1), "h");
    vi.remove("p#0", true);
    expect(rows().map((r) => r.id).sort()).toEqual(["p#1", "q"]);
    expect(vi.size).toBe(2);
    // Exact on a parent leaves its chunks; cascading takes them.
    vi.add("q#0", "s", v(1, 0), "h");
    vi.remove("q", true);
    expect(rows().map((r) => r.id).sort()).toEqual(["p#1", "q#0"]);
    vi.remove("q");
    expect(rows().map((r) => r.id)).toEqual(["p#1"]);
    expect(vi.size).toBe(1);
  });

  it("leaves a store-less VectorIndex untouched in behaviour", () => {
    const plain = new VectorIndex();
    plain.add("p#0", "s", v(1, 0));
    plain.add("p#1", "s", v(0, 1));
    plain.add("p", "s", v(1, 1));
    expect(plain.size).toBe(1);
    plain.remove("p");
    expect(plain.size).toBe(0);
    expect(SearchIndex).toBeDefined();
  });
});
