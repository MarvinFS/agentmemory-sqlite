import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { existsSync, mkdirSync, mkdtempSync, readdirSync, readFileSync, rmSync, statSync, unlinkSync, writeFileSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import { DatabaseSync } from "node:sqlite";

import {
  decodeRkyvString,
  decodeScopeFile,
  digestPairs,
  encodeRkyvString,
  importStateStore,
  isSkippedScope,
  scanJsonArrayRows,
  topLevelKeys,
} from "../src/ops/import-state-store.js";
import { SearchIndex } from "../src/state/search-index.js";
import { VectorIndex } from "../src/state/vector-index.js";
import { embedInputHash } from "../src/state/index-corpus.js";
import type { CompressedObservation, EmbeddingProvider, Memory } from "../src/types.js";

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

function fakeProvider(): EmbeddingProvider & { calls: string[][] } {
  const p = {
    name: "fake",
    dimensions: 4,
    calls: [] as string[][],
    async embed(t: string) {
      return (await p.embedBatch([t]))[0];
    },
    async embedBatch(ts: string[]) {
      p.calls.push(ts);
      return ts.map((t) => v(t.length % 7, 1, 0, 0));
    },
  };
  return p;
}

// Writes one scope file the way iii does: compact JSON in an rkyv String.
function writeScope(dir: string, scope: string, value: Record<string, unknown> | string): void {
  const text = typeof value === "string" ? value : JSON.stringify(value);
  writeFileSync(join(dir, encodeURIComponent(scope) + ".bin"), encodeRkyvString(text));
}

function shards(dir: string, prefix: string, serialized: string, splitAt: number[]): { scopes: Array<{ scope: string; key: string; chars: number }>; chars: number } {
  const bounds = [0, ...splitAt, serialized.length];
  const list: Array<{ scope: string; key: string; chars: number }> = [];
  for (let i = 0; i + 1 < bounds.length; i++) {
    const chunk = serialized.slice(bounds[i], bounds[i + 1]);
    const scope = `${prefix}:${String(i).padStart(5, "0")}`;
    writeScope(dir, scope, { data: chunk });
    list.push({ scope, key: "data", chars: chunk.length });
  }
  return { scopes: list, chars: serialized.length };
}

describe("import-state-store", () => {
  let root: string;
  let src: string;
  let out: string;

  beforeEach(() => {
    root = mkdtempSync(join(tmpdir(), "am-import-"));
    src = join(root, "state_store.db");
    mkdirSync(src);
    out = join(root, "agentmemory.sqlite");
  });
  afterEach(() => rmSync(root, { recursive: true, force: true }));

  it("decodes the rkyv string layout strictly, inline and out of line", () => {
    for (const text of ["{}", '{"a":1}', '{"k":"' + "x".repeat(37) + '"}', "{" + '"k":"é✓"'.repeat(50) + "}"]) {
      const buf = encodeRkyvString(text);
      const d = decodeRkyvString(buf);
      expect(d.text).toBe(text);
      expect(d.inline).toBe(Buffer.byteLength(text) <= 8);
    }
    // The encoded lengths seen on the real store: 46 -> 0xae, 485 -> 0x7a5.
    expect(encodeRkyvString("x".repeat(46)).readUInt32LE(48)).toBe(0xae);
    expect(encodeRkyvString("x".repeat(485)).readUInt32LE(488)).toBe(0x7a5);
    // Corruptions are refused.
    const good = encodeRkyvString('{"a":1,"b":2}');
    const nonZeroPad = Buffer.from(good);
    nonZeroPad[13] = 1;
    expect(() => decodeRkyvString(nonZeroPad)).toThrow(/padding/);
    const badOffset = Buffer.from(good);
    badOffset.writeInt32LE(-4, good.length - 4);
    expect(() => decodeRkyvString(badOffset)).toThrow(/start/);
    expect(() => decodeRkyvString(Buffer.from([1, 2, 3]))).toThrow(/root/);
    const badUtf8 = encodeRkyvString('{"a":"xxxxxxxx"}');
    badUtf8[6] = 0xff;
    expect(() => decodeRkyvString(badUtf8)).toThrow();
  });

  it("keeps file order for integer-like keys and detects duplicates", () => {
    expect(topLevelKeys('{"2":{"1":0},"1":[{"x":1}],"a\\"b":":",":":1}')).toEqual(["2", "1", 'a"b', ":"]);
    writeScope(src, "mem:dup", '{"a":1,"a":2}');
    expect(() => decodeScopeFile(join(src, "mem%3Adup.bin"))).toThrow(/duplicate/);
    writeScope(src, "mem:arr", "[1,2]");
    expect(() => decodeScopeFile(join(src, "mem%3Aarr.bin"))).toThrow(/not a JSON object/);
    writeScope(src, "mem:obs:s#1", { o1: { id: "o1" } });
    const d = decodeScopeFile(join(src, "mem%3Aobs%3As%231.bin"));
    expect(d.scope).toBe("mem:obs:s#1");
    expect(d.keys).toEqual(["o1"]);
  });

  it("streams array rows across chunk boundaries without splitting on ],[ inside strings", () => {
    const rows = [
      JSON.stringify(["a", { embedding: "AAAA", sessionId: "s],[weird\"x" }]),
      JSON.stringify(["b", { embedding: "BBBB", sessionId: "[" }]),
      JSON.stringify(["c", { embedding: "CCCC", sessionId: "\\" }]),
    ];
    const text = "[" + rows.join(",") + "]";
    for (const cut of [1, 5, 17, 40, 60, text.length - 2]) {
      const chunks = [text.slice(0, cut), text.slice(cut)];
      expect([...scanJsonArrayRows(chunks)]).toEqual(rows);
    }
    expect([...scanJsonArrayRows(["[", "]"])]).toEqual([]);
    expect([...scanJsonArrayRows([" [ [1] ,", " {\"a\":[2]} ] \n"])]).toEqual(["[1]", '{"a":[2]}']);
    expect(() => [...scanJsonArrayRows(["[[1,2]"])]).toThrow(/truncated/);
    expect(() => [...scanJsonArrayRows(['{"a":1}'])]).toThrow(/array/);
    // The grammar outside the rows is enforced: no primitives, no trailing
    // comma, nothing after the array, no second array.
    for (const bad of ["[null]", "[1]", '["x"]', "[[1],]", "[[1]] x", "[[1]],", "[[1]][", "[[1] [2]]", "[[1]}"]) {
      expect(() => [...scanJsonArrayRows([bad])], bad).toThrow();
    }
  });

  it("digests are length-framed, so no byte inside a key or value can move a boundary", () => {
    expect(digestPairs([["a\u00000\nb", "1"]])).not.toBe(digestPairs([["a", "0"], ["b", "1"]]));
    expect(digestPairs([["ab", "c"]])).not.toBe(digestPairs([["a", "bc"]]));
    expect(digestPairs([["a", "1"], ["b", "2"]])).not.toBe(digestPairs([["b", "2"], ["a", "1"]]));
  });

  it("skips exactly the scopes the plan names", () => {
    for (const s of ["mem:audit", "mem:health", "mem:index:bm25", "mem:index:bm25:bm25:g:00000", "mem:index:bm25:vectors:g:00001", "mem:graph:nodes", "mem:graph:edge-key"]) {
      expect(isSkippedScope(s)).toBe(true);
    }
    for (const s of ["mem:memories", "mem:obs:s1", "mem:sessions", "mem:compress-pending", "mem:insights", "mem:semantic", "mem:access", "mem:indexes", "mem:healthy"]) {
      expect(isSkippedScope(s)).toBe(false);
    }
  });

  it("imports a store end to end: kv in file order, legacy vectors, deltas, fill, publish", async () => {
    // ---- content
    const m1 = memory("mem_1", "first memory about zebras");
    const m2 = memory("mem_2", "superseded memory", { isLatest: false });
    const m3 = memory("mem_3", "third memory with no vector yet");
    writeScope(src, "mem:memories", { mem_1: m1, mem_2: m2, mem_3: m3 });
    const o1 = observation("obs_1", "s1", "compressed one");
    const rawObs = { id: "obs_raw", sessionId: "s1", timestamp: "", hookType: "post_tool_use", raw: { big: true } };
    writeScope(src, "mem:obs:s1", { obs_1: o1, obs_raw: rawObs });
    const o3 = observation("obs_3", "s2", "compressed three");
    writeScope(src, "mem:obs:s2", { obs_3: o3 });
    // A literal, because a JS object would already reorder the integer-like keys.
    writeScope(src, "mem:sessions", '{"10":{"id":"10"},"2":{"id":"2"},"s1":{"id":"s1"}}');
    writeScope(src, "mem:compress-pending", { obs_raw: { observationId: "obs_raw" } });
    writeScope(src, "mem:state", {});
    // ---- skipped scopes
    writeScope(src, "mem:health", { _probe: { ts: 1 } });
    writeScope(src, "mem:graph:nodes", { n1: { id: "n1" } });
    writeScope(src, "mem:graph:edges", { e1: { id: "e1" } });
    writeScope(src, "mem:audit", {
      a1: { id: "a1", timestamp: "2026-09-01T00:00:00Z", operation: "index_persist", functionId: "mem::index-persistence", targetIds: ["x"] },
      a2: { id: "a2", timestamp: "2026-09-03T00:00:00Z", operation: "forget", functionId: "mem::forget", targetIds: ["mem_gone"], details: { reason: "user" } },
      a3: { id: "a3", timestamp: "2026-09-02T00:00:00Z", operation: "delete", functionId: "mem::evict", targetIds: ["obs_old"], details: { reason: "expired" } },
      a4: { id: "a4", timestamp: "2026-09-02T00:00:00Z", operation: "compress", functionId: "mem::summarize", targetIds: ["obs_1"] },
    });
    // ---- old indexes: BM25 with a doc that no longer exists, vectors with an orphan
    const bm25 = new SearchIndex();
    bm25.add(o1);
    bm25.add({ ...o1, id: "mem_1", sessionId: "memory", title: m1.title, narrative: m1.content });
    bm25.add(observation("gone", "s9", "deleted since"));
    const bm25Json = bm25.serialize();
    const bmShards = shards(src, "mem:index:bm25:bm25:gen_a", bm25Json, [Math.floor(bm25Json.length / 3)]);
    const vi = new VectorIndex();
    vi.add("obs_1", "s1", v(1, 0, 0, 0));
    vi.add("mem_1", "memory", v(0, 1, 0, 0));
    vi.add("obs_3", 's],[2"x', v(0, 0, 1, 0));
    vi.add("gone_vec", "s9", v(0, 0, 0, 1));
    const vecJson = vi.serialize();
    // Split inside the second row.
    const secondRow = vecJson.indexOf("],[") + 2;
    const vecShards = shards(src, "mem:index:bm25:vectors:gen_b", vecJson, [secondRow + 30]);
    writeScope(src, "mem:index:bm25", {
      "data:manifest": { v: 1, generation: "gen_a", shards: bmShards.scopes, chars: bmShards.chars },
      "vectors:manifest": { v: 1, generation: "gen_b", shards: vecShards.scopes, chars: vecShards.chars },
      "checkpoint:rollback": { generation: "gen_old" },
    });

    const provider = fakeProvider();
    const lines: string[] = [];
    const report = await importStateStore({ src, out, embeddingProvider: provider, log: (l) => lines.push(l) });

    expect(report.published).toBe(true);
    expect(report.error).toBeUndefined();
    expect(existsSync(out)).toBe(true);
    expect(readdirSync(root).filter((f) => f.includes(".tmp-"))).toEqual([]);
    expect(existsSync(join(root, "import-report.json"))).toBe(true);
    expect(JSON.parse(readFileSync(join(root, "import-report.json"), "utf8")).published).toBe(true);

    const db = new DatabaseSync(out, { readOnly: true });
    try {
      const scopes = (db.prepare("SELECT DISTINCT scope FROM kv ORDER BY scope").all() as Array<{ scope: string }>).map((r) => r.scope);
      expect(scopes).toEqual(["mem:compress-pending", "mem:memories", "mem:obs:s1", "mem:obs:s2", "mem:sessions"]);
      // File order, not JS integer-key order.
      expect((db.prepare("SELECT key FROM kv WHERE scope = 'mem:sessions' ORDER BY seq").all() as Array<{ key: string }>).map((r) => r.key)).toEqual(["10", "2", "s1"]);
      expect(JSON.parse((db.prepare("SELECT value FROM kv WHERE scope = 'mem:memories' AND key = 'mem_1'").get() as { value: string }).value)).toEqual(m1);
      const vecs = db.prepare("SELECT id, session_id, dims, hash_state, input_hash FROM vectors ORDER BY seq").all() as Array<{ id: string; session_id: string; dims: number; hash_state: string; input_hash: string }>;
      const byId = new Map(vecs.map((r) => [r.id, r]));
      // Imported rows are legacy; gone_vec pruned (no content); mem_3 embedded (missing).
      expect([...byId.keys()].sort()).toEqual(["mem_1", "mem_3", "obs_1", "obs_3"]);
      expect(byId.get("obs_3")).toMatchObject({ session_id: 's],[2"x', dims: 4, hash_state: "legacy", input_hash: "" });
      expect(byId.get("mem_1")).toMatchObject({ hash_state: "legacy" });
      expect(byId.get("mem_3")).toMatchObject({ hash_state: "verified", input_hash: embedInputHash(m3.title + " " + m3.content) });
      expect((db.prepare("PRAGMA integrity_check").get() as { integrity_check: string }).integrity_check).toBe("ok");
    } finally {
      db.close();
    }

    expect(report.totals).toEqual({ importedScopes: 6, importedKeys: 10, skippedScopes: 9, skippedKeys: 14 });
    expect(report.observations).toEqual({ scopes: 2, rows: 3, rawOrphans: 1, rawOrphanSample: ["mem:obs:s1/obs_raw"] });
    expect(report.compressPending).toBe(1);
    expect(report.auditDeletions.map((d) => [d.operation, d.functionId, d.targetIds[0], d.reason])).toEqual([
      ["delete", "mem::evict", "obs_old", "expired"],
      ["forget", "mem::forget", "mem_gone", "user"],
    ]);
    expect(report.vectors).toMatchObject({ generation: "gen_b", shards: 2, imported: 4, dims: 4, delta: { addedCount: 1, removedCount: 1, added: ["mem_3"], removed: ["gone_vec"] } });
    expect(report.bm25.oldDocs).toBe(3);
    expect(report.bm25.rebuilt).toMatchObject({ docs: 4, rows: 6, skipped: 0 });
    expect(report.bm25.withinBudget).toBe(true);
    expect(report.bm25.delta).toMatchObject({ onlyInOldCount: 1, onlyInNewCount: 2, onlyInOld: ["gone"] });
    expect(report.bm25.delta!.onlyInNew.sort()).toEqual(["mem_3", "obs_3"]);
    expect(report.fill).toMatchObject({ expected: 4, present: 3, missing: 1, embedded: 1, pruned: 1, failed: 0, aborted: false });
    expect(provider.calls.flat()).toEqual([m3.title + " " + m3.content]);
    expect(report.readiness).toEqual({ hydrated: 4, integrityCheck: "ok", digestRecheck: "ok" });
    expect(report.warnings).toEqual([]);
    const digest = report.scopes.find((s) => s.scope === "mem:memories")!.digest;
    expect(digest).toBe(digestPairs([["mem_1", JSON.stringify(m1)], ["mem_2", JSON.stringify(m2)], ["mem_3", JSON.stringify(m3)]]));

    // Running again refuses to clobber, then replaces on request.
    await expect(importStateStore({ src, out, embeddingProvider: provider, log: () => {} })).rejects.toThrow(/--replace/);
    const again = await importStateStore({ src, out, embeddingProvider: provider, replace: true, log: () => {} });
    expect(again.published).toBe(true);
  });

  it("fails closed: a bad scope file leaves no output and no temp file", async () => {
    writeScope(src, "mem:memories", { mem_1: memory("mem_1", "ok") });
    writeScope(src, "mem:sessions", '{"a":1,"a":2}');
    await expect(importStateStore({ src, out, embeddingProvider: fakeProvider(), log: () => {} })).rejects.toThrow(/duplicate keys/);
    expect(existsSync(out)).toBe(false);
    expect(readdirSync(root)).toEqual(["state_store.db"]);
  });

  it("fails closed on a shard length mismatch, a bad vector row, and a missing provider", async () => {
    writeScope(src, "mem:memories", { mem_1: memory("mem_1", "ok") });
    const vi = new VectorIndex();
    vi.add("mem_1", "memory", v(1, 0, 0, 0));
    const vecJson = vi.serialize();
    const s = shards(src, "mem:index:bm25:vectors:g", vecJson, []);
    writeScope(src, "mem:index:bm25", { "vectors:manifest": { v: 1, shards: [{ ...s.scopes[0], chars: s.chars + 1 }], chars: s.chars + 1 } });
    await expect(importStateStore({ src, out, embeddingProvider: fakeProvider(), log: () => {} })).rejects.toThrow(/manifest says/);
    expect(existsSync(out)).toBe(false);

    writeScope(src, "mem:index:bm25", { "vectors:manifest": { v: 1, shards: s.scopes, chars: s.chars } });
    writeScope(src, "mem:index:bm25:vectors:g:00000", { data: '[["mem_1",{"embedding":"AAA","sessionId":"memory"}]]' });
    writeScope(src, "mem:index:bm25", { "vectors:manifest": { v: 1, shards: [{ ...s.scopes[0], chars: 52 }], chars: 52 } });
    await expect(importStateStore({ src, out, embeddingProvider: fakeProvider(), log: () => {} })).rejects.toThrow(/base64/);

    writeScope(src, "mem:index:bm25:vectors:g:00000", { data: vecJson });
    writeScope(src, "mem:index:bm25", { "vectors:manifest": { v: 1, shards: s.scopes, chars: s.chars } });
    await expect(importStateStore({ src, out, embeddingProvider: null, log: () => {} })).rejects.toThrow(/no embedding provider/);
    expect(existsSync(out)).toBe(false);
    // --no-fill publishes without a provider and says so.
    const r = await importStateStore({ src, out, embeddingProvider: null, fill: false, log: () => {} });
    expect(r.published).toBe(true);
    expect(r.fill).toBeNull();
    expect(r.warnings.some((w) => /fill pass skipped/.test(w))).toBe(true);
    expect(readdirSync(root).filter((f) => f.includes(".tmp-"))).toEqual([]);
  });

  it("accepts a root whose offset bytes contain a brace byte", () => {
    // 33536 bytes -> offset -33536 = 0xFFFF7D00: a '}' (0x7d) inside the root.
    const body = '{"k":"' + "x".repeat(33536 - 8) + '"}';
    expect(Buffer.byteLength(body)).toBe(33536);
    const buf = encodeRkyvString(body);
    expect(buf.subarray(buf.length - 4).includes(0x7d)).toBe(true);
    writeFileSync(join(src, "mem%3Abrace.bin"), buf);
    expect(decodeScopeFile(join(src, "mem%3Abrace.bin")).keys).toEqual(["k"]);
    // 8000 bytes: the encoded LENGTH contains 0x7d.
    const body2 = '{"k":"' + "x".repeat(7992) + '"}';
    expect(Buffer.byteLength(body2)).toBe(8000);
    writeScope(src, "mem:brace2", body2);
    expect(decodeScopeFile(join(src, "mem%3Abrace2.bin")).keys).toEqual(["k"]);
    // Multi-byte content: bounds are byte positions, not character positions.
    writeScope(src, "mem:mb1", '{"\u00e9":0}'); // 8 bytes, inline root
    writeScope(src, "mem:mb2", '{"\u00e9":"\u00fcn\u00efcode \u00fcn\u00efcode"}');
    expect(decodeScopeFile(join(src, "mem%3Amb1.bin")).keys).toEqual(["\u00e9"]);
    expect(decodeScopeFile(join(src, "mem%3Amb2.bin")).keys).toEqual(["\u00e9"]);
  });

  it("--replace refuses a destination whose WAL holds frames and never touches its sidecars", async () => {
    writeScope(src, "mem:memories", { mem_1: memory("mem_1", "ok") });
    writeFileSync(out, "old");
    writeFileSync(`${out}-wal`, Buffer.alloc(40));
    const run = () => importStateStore({ src, out, replace: true, fill: false, embeddingProvider: null, log: () => {} });
    await expect(run()).rejects.toThrow(/uncheckpointed frames/);
    expect(readFileSync(out, "utf8")).toBe("old");
    expect(statSync(`${out}-wal`).size).toBe(40);
    expect(readdirSync(root).filter((f) => f.includes(".tmp-"))).toEqual([]);
    // A header-only WAL is a cleanly closed database: replaced, its sidecars gone.
    writeFileSync(`${out}-wal`, Buffer.alloc(32));
    const r = await run();
    expect(r.published).toBe(true);
    expect(r.readiness).toMatchObject({ hydrated: 0, integrityCheck: "ok", digestRecheck: "ok" });
    expect(readdirSync(root).sort()).toEqual(["agentmemory.sqlite", "import-report.json", "state_store.db"]);
  });

  it("refuses an orphan WAL with frames next to a missing destination, before the rename", async () => {
    writeScope(src, "mem:memories", { mem_1: memory("mem_1", "ok") });
    writeFileSync(`${out}-wal`, Buffer.alloc(40));
    await expect(importStateStore({ src, out, fill: false, embeddingProvider: null, log: () => {} })).rejects.toThrow(/uncheckpointed frames/);
    expect(existsSync(out)).toBe(false);
    expect(statSync(`${out}-wal`).size).toBe(40);
    expect(readdirSync(root).filter((f) => f.includes(".tmp-"))).toEqual([]);
    unlinkSync(`${out}-wal`);
    // A hot rollback journal is refused the same way.
    writeFileSync(`${out}-journal`, "hot");
    await expect(importStateStore({ src, out, fill: false, embeddingProvider: null, log: () => {} })).rejects.toThrow(/-journal exists/);
    expect(existsSync(out)).toBe(false);
    unlinkSync(`${out}-journal`);
  });

  it("an unwritable report path fails before anything is published", async () => {
    writeScope(src, "mem:memories", { mem_1: memory("mem_1", "ok") });
    const reportPath = join(root, "no-such-dir", "import-report.json");
    await expect(importStateStore({ src, out, reportPath, fill: false, embeddingProvider: null, log: () => {} })).rejects.toThrow(/ENOENT/);
    expect(readdirSync(root)).toEqual(["state_store.db"]);
  });

  it("validate-only decodes everything, lists every bad file, and writes nothing", async () => {
    writeScope(src, "mem:memories", { mem_1: memory("mem_1", "ok") });
    writeScope(src, "mem:graph:nodes", { n: 1 });
    // A file that happens to carry this run's temporary name is not ours to delete.
    const decoy = `${out}.tmp-${process.pid}`;
    writeFileSync(decoy, "not yours");
    const r = await importStateStore({ src, out, validateOnly: true, log: () => {} });
    expect(r.published).toBe(false);
    expect(r.scopes.map((s) => [s.scope, s.action, s.keys])).toEqual([["mem:graph:nodes", "skipped", 1], ["mem:memories", "imported", 1]]);
    expect(readFileSync(decoy, "utf8")).toBe("not yours");
    unlinkSync(decoy);
    expect(readdirSync(root)).toEqual(["state_store.db"]);

    writeScope(src, "mem:bad1", '{"a":1,"a":2}');
    writeFileSync(join(src, "mem%3Abad2.bin"), Buffer.from([1, 2, 3]));
    let report: { scopes: Array<{ scope: string; action: string }> } | undefined;
    await importStateStore({ src, out, validateOnly: true, log: () => {} }).catch((e) => (report = e.report));
    expect(report!.scopes.map((s) => [s.scope, s.action])).toEqual([
      ["mem:bad1", "failed"],
      ["mem:bad2", "failed"],
      ["mem:graph:nodes", "skipped"],
      ["mem:memories", "imported"],
    ]);
    // An import stops at the first bad file.
    await expect(importStateStore({ src, out, embeddingProvider: fakeProvider(), log: () => {} })).rejects.toThrow(/duplicate keys/);
    expect(readdirSync(root)).toEqual(["state_store.db"]);
  });
});
