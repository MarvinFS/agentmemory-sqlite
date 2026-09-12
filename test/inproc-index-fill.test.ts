import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import { SqliteState } from "../src/engine/inproc/state.js";
import { SqliteVectorStore } from "../src/engine/inproc/vectors.js";
import { VectorIndex } from "../src/state/vector-index.js";
import { KV } from "../src/state/schema.js";
import { embedInputHash, memoryEmbedJobs } from "../src/state/index-corpus.js";
import { claimedNow, createIndexFill } from "../src/functions/index-fill.js";
import {
  setEmbeddingProvider,
  setInprocStores,
  setVectorIndex,
} from "../src/functions/search.js";
import type { Memory, CompressedObservation, EmbeddingProvider } from "../src/types.js";

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
    title: "title " + id,
    facts: [],
    narrative,
    concepts: [],
    files: [],
    importance: 5,
  };
}

// Deterministic 2-d provider: counts calls so the tests can assert what got
// embedded, and can be switched to fail to exercise the abort path.
function fakeProvider(): EmbeddingProvider & { calls: string[][]; failing: boolean } {
  const p = {
    name: "fake",
    dimensions: 2,
    calls: [] as string[][],
    failing: false,
    async embed(t: string) {
      return (await p.embedBatch([t]))[0];
    },
    async embedBatch(ts: string[]) {
      if (p.failing) throw new Error("provider down");
      p.calls.push(ts);
      return ts.map((t) => v(t.length % 7, 1));
    },
  };
  return p;
}

describe("mem::index-fill-missing", () => {
  let dir: string;
  let state: SqliteState;
  let vectors: SqliteVectorStore;
  let vi: VectorIndex;
  let provider: ReturnType<typeof fakeProvider>;

  const rows = () =>
    state.db
      .prepare("SELECT id, input_hash, hash_state FROM vectors ORDER BY seq")
      .all() as Array<{ id: string; input_hash: string; hash_state: string }>;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "am-inproc-fill-"));
    state = new SqliteState(join(dir, "state.sqlite"));
    vectors = new SqliteVectorStore(state);
    vi = new VectorIndex();
    vi.attachStore(vectors);
    provider = fakeProvider();
    setVectorIndex(vi);
    setEmbeddingProvider(provider);
    setInprocStores(state, vectors);
  });

  afterEach(() => {
    setInprocStores(null, null);
    setEmbeddingProvider(null);
    setVectorIndex(null);
    state.close();
    rmSync(dir, { recursive: true, force: true });
  });

  it("embeds missing and stale rows, keeps legacy rows, prunes orphans, and revalidates at apply time", async () => {
    const m1 = memory("mem_1", "no vector yet");
    const m2 = memory("mem_2", "stale vector");
    const m3 = memory("mem_3", "legacy import");
    const m4 = memory("mem_4", "already good");
    for (const m of [m1, m2, m3, m4]) state.set(KV.memories, m.id, m);
    state.set(KV.observations("s1"), "obs_1", observation("obs_1", "s1", "fresh observation"));

    // mem_2: verified row with a hash for older content -> stale.
    vi.add("mem_2", "memory", v(0, 1), "old-hash");
    // mem_3: legacy row -> trusted as-is.
    state.db
      .prepare("INSERT INTO vectors (id, session_id, dims, embedding, input_hash, hash_state) VALUES ('mem_3', 'memory', 2, ?, '', 'legacy')")
      .run(Buffer.from(v(1, 1).buffer));
    vi.hydrate([{ id: "mem_3", sessionId: "memory", embedding: v(1, 1) }]);
    // mem_4: verified and current.
    vi.add("mem_4", "memory", v(1, 0), embedInputHash(m4.title + " " + m4.content));
    // Orphans: content gone, and content superseded.
    vi.add("mem_gone", "memory", v(1, 0), "h");
    state.set(KV.memories, "mem_super", memory("mem_super", "superseded", { isLatest: false }));
    vi.add("mem_super", "memory", v(1, 0), "h");
    vi.add("obs_gone", "s9", v(1, 0), "h");

    const fill = createIndexFill(state, vectors, vi);
    const r = await fill.run();

    expect(r).toMatchObject({ skipped: false, expected: 5, present: 2, missing: 3, embedded: 3, failed: 0, pruned: 3, aborted: false });
    expect(provider.calls.flat().sort()).toEqual(
      [m1, m2].map((m) => m.title + " " + m.content).concat(["title obs_1 fresh observation"]).sort(),
    );
    const byId = new Map(rows().map((row) => [row.id, row]));
    expect([...byId.keys()].sort()).toEqual(["mem_1", "mem_2", "mem_3", "mem_4", "obs_1"]);
    expect(byId.get("mem_2")).toMatchObject({ hash_state: "verified", input_hash: embedInputHash(m2.title + " " + m2.content) });
    expect(byId.get("mem_3")).toMatchObject({ hash_state: "legacy", input_hash: "" });
    expect(vi.size).toBe(5);

    // Second run: nothing to do.
    const again = await fill.run();
    expect(again).toMatchObject({ expected: 5, present: 5, missing: 0, embedded: 0, pruned: 0 });
    expect(provider.calls).toHaveLength(1);
  });

  it("a content rewrite invalidates legacy rows in the same write, so the next pass re-embeds them", async () => {
    const m = memory("mem_l", "imported text");
    state.set(KV.memories, "mem_l", m);
    state.set(KV.observations("s1"), "obs_l", observation("obs_l", "s1", "imported observation"));
    const legacy = state.db.prepare(
      "INSERT INTO vectors (id, session_id, dims, embedding, input_hash, hash_state) VALUES (?, ?, 2, ?, '', 'legacy')",
    );
    for (const [id, sid] of [["mem_l", "memory"], ["mem_l#0", "memory"], ["mem_l#1", "memory"], ["obs_l", "s1"]]) {
      legacy.run(id, sid, Buffer.from(v(1, 1).buffer));
    }
    // Writes elsewhere leave legacy rows alone.
    state.set(KV.sessions, "mem_l", { id: "mem_l" });
    state.set(KV.memories, "mem_other", memory("mem_other", "unrelated"));
    expect(rows().filter((r) => r.hash_state === "legacy")).toHaveLength(4);

    state.set(KV.memories, "mem_l", memory("mem_l", "rewritten text"));
    const byId = new Map(rows().map((r) => [r.id, r]));
    for (const id of ["mem_l", "mem_l#0", "mem_l#1"]) {
      expect(byId.get(id)).toMatchObject({ hash_state: "verified", input_hash: "" });
    }
    expect(byId.get("obs_l")).toMatchObject({ hash_state: "legacy" });

    vi.hydrate([...rows()].map((r) => ({ id: r.id, sessionId: r.id.startsWith("obs") ? "s1" : "memory", embedding: v(1, 1) })));
    const r = await createIndexFill(state, vectors, vi).run();
    // mem_l: one current job (short text, no chunks) -> its parent row is
    // stale and the two chunk rows are orphans; obs_l stays legacy; mem_other
    // is missing and gets embedded.
    expect(r).toMatchObject({ expected: 3, present: 1, missing: 2, embedded: 2, pruned: 2 });
    const after = new Map(rows().map((r) => [r.id, r]));
    expect([...after.keys()].sort()).toEqual(["mem_l", "mem_other", "obs_l"]);
    const current = memory("mem_l", "rewritten text");
    expect(after.get("mem_l")).toMatchObject({ hash_state: "verified", input_hash: embedInputHash(current.title + " " + current.content) });
    expect(after.get("obs_l")).toMatchObject({ hash_state: "legacy", input_hash: "" });
  });

  it("prunes a stale bare-parent row without touching the memory's current chunk rows", async () => {
    process.env.AGENTMEMORY_MEMORY_CHUNKING = "true";
    try {
      // ~3000 chars against the default 1200-char chunk budget -> several chunks.
      const m = memory("mem_c", Array.from({ length: 40 }, (_, i) => `Sentence number ${i} about chunked memories, long enough to need splitting.`).join(" "));
      state.set(KV.memories, "mem_c", m);
      const jobs = memoryEmbedJobs(m);
      expect(jobs.length).toBeGreaterThan(1);
      // A stale whole-document row from before chunking (older seq, so it
      // hydrates first) plus the current chunk rows.
      state.db
        .prepare("INSERT INTO vectors (id, session_id, dims, embedding, input_hash, hash_state) VALUES ('mem_c', 'memory', 2, ?, 'old', 'verified')")
        .run(Buffer.from(v(0, 1).buffer));
      vi.hydrate([{ id: "mem_c", sessionId: "memory", embedding: v(0, 1) }]);
      for (const job of jobs) vi.add(job.id, "memory", v(1, 0), embedInputHash(job.text));
      expect(vi.size).toBe(jobs.length + 1);
      expect(rows()).toHaveLength(jobs.length + 1);

      const r = await createIndexFill(state, vectors, vi).run();
      expect(r).toMatchObject({ expected: jobs.length, present: jobs.length, missing: 0, embedded: 0, pruned: 1 });
      expect(rows().map((row) => row.id).sort()).toEqual(jobs.map((j) => j.id).sort());
      expect(vi.size).toBe(jobs.length);
      expect(vi.search(v(1, 0), 5).map((h) => h.obsId)).toEqual(["mem_c"]);
      expect(provider.calls).toHaveLength(0);
    } finally {
      delete process.env.AGENTMEMORY_MEMORY_CHUNKING;
    }
  });

  it("claimedNow checks both content kinds behind a shared id", () => {
    state.set(KV.memories, "dup", memory("dup", "superseded", { isLatest: false }));
    state.set(KV.observations("s1"), "dup", observation("dup", "s1", "live observation"));
    expect(claimedNow(state, "s1", "dup")).toBe(true);
    expect(claimedNow(state, "s2", "dup")).toBe(false);
    expect(claimedNow(state, "s1", "dup#0")).toBe(false);
    state.set(KV.memories, "dup", memory("dup", "latest again"));
    expect(claimedNow(state, "s2", "dup")).toBe(true);
  });

  it("drops a completion whose content changed mid-flight and is single-flight", async () => {
    state.set(KV.memories, "mem_r", memory("mem_r", "first text"));
    const fill = createIndexFill(state, vectors, vi);
    // Rewrite the content while the embed is in flight.
    const original = provider.embedBatch.bind(provider);
    provider.embedBatch = async (ts) => {
      state.set(KV.memories, "mem_r", memory("mem_r", "second text"));
      return original(ts);
    };
    const [r, second] = await Promise.all([fill.run(), fill.run()]);
    expect(second.skipped).toBe(true);
    expect(r).toMatchObject({ missing: 1, embedded: 0, failed: 1 });
    expect(rows()).toEqual([]);

    // The next run embeds the current text.
    provider.embedBatch = original;
    const r2 = await fill.run();
    expect(r2).toMatchObject({ missing: 1, embedded: 1 });
    const current = memory("mem_r", "second text");
    expect(rows()[0].input_hash).toBe(embedInputHash(current.title + " " + current.content));
  });

  it("memories with a non-string id or session list are not jobs, so they cannot starve the pass", async () => {
    for (let i = 0; i < 96; i++) {
      state.set(KV.memories, `mem_bad_${i}`, { ...memory(`mem_bad_${i}`, `content number ${i}`), id: i as never });
    }
    state.set(KV.memories, "mem_odd", memory("mem_odd", "odd session list", { sessionIds: "abc" as never }));
    state.set(KV.memories, "mem_ok", memory("mem_ok", "fine"));
    const r = await createIndexFill(state, vectors, vi).run();
    expect(r).toMatchObject({ expected: 2, embedded: 2, failed: 0, aborted: false });
    expect(rows().map((x) => x.id).sort()).toEqual(["mem_odd", "mem_ok"]);
    // A string where the array should be is not indexed one character at a time.
    expect(memoryEmbedJobs(memory("mem_odd", "odd", { sessionIds: "abc" as never }))[0].sessionId).toBe("memory");
    expect(memoryEmbedJobs({ ...memory("mem_x", "x"), id: 5 as never })).toEqual([]);
  });

  it("aborts after three consecutive whole-batch failures and reports the remainder as failed", async () => {
    for (let i = 0; i < 130; i++) state.set(KV.memories, `mem_${i}`, memory(`mem_${i}`, `content number ${i}`));
    provider.failing = true;
    const r = await createIndexFill(state, vectors, vi).run();
    expect(r).toMatchObject({ missing: 130, embedded: 0, failed: 130, aborted: true });
    expect(rows()).toEqual([]);
  });
});
