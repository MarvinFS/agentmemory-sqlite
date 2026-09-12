#!/usr/bin/env node
// Import the iii engine's `state_store.db/` directory into the inproc
// daemon's SQLite file (plan step 8). Node only; run on the VM with the old
// daemon STOPPED and the daemon's environment exported (EMBEDDING_PROVIDER,
// OPENAI_EMBEDDING_*, AGENTMEMORY_INDEX_EXCLUDE_TOOL_PREFIXES,
// AGENTMEMORY_MEMORY_CHUNKING), because the fill pass and the eligibility
// rules read the same variables the daemon does.
//
//   node dist/import-state-store.mjs --src ~/data/state_store.db \
//        --out ~/data/agentmemory.sqlite [--report import-report.json]
//        [--validate-only] [--no-fill] [--allow-slow-bm25] [--replace]
//
// Every check must pass before anything is published: the import writes a
// temporary DB next to `--out`, checkpoints and closes it, runs
// `integrity_check` on a fresh read-only connection, and only then renames it
// into place. Any failure deletes the temporary DB and leaves the old store
// untouched. `import-report.json` is written next to the published file (or
// printed on failure).
import { createHash } from "node:crypto";
import { existsSync, readdirSync, readFileSync, renameSync, statSync, unlinkSync, writeFileSync } from "node:fs";
import { basename, dirname, join } from "node:path";
import { DatabaseSync } from "node:sqlite";
import { parseArgs } from "node:util";

import { SqliteState } from "../engine/inproc/state.js";
import { SqliteVectorStore } from "../engine/inproc/vectors.js";
import { VectorIndex } from "../state/vector-index.js";
import { KV } from "../state/schema.js";
import {
  getSearchIndex,
  rebuildBm25FromContent,
  setEmbeddingProvider,
  setInprocStores,
  setVectorIndex,
} from "../functions/search.js";
import { createIndexFill, type FillReport } from "../functions/index-fill.js";
import { createEmbeddingProvider } from "../providers/embedding/index.js";
import type { EmbeddingProvider } from "../types.js";

// ---------------------------------------------------------------- rkyv decoder
//
// Each scope file is one rkyv 0.8 archived `String` of compact JSON
// `{key: value, ...}` (iii 0.11.2 `kv.rs`). The archive's root is the last 8
// bytes. Out-of-line strings (len > 8): `u32 LE` encoded length whose first
// byte is `0b10xxxxxx`, then `i32 LE` offset from the root to the bytes;
// the bytes start the file and are zero-padded to 4. Inline strings
// (len <= 8) are the 8 root bytes themselves, `0xff`-padded. Decoding is
// strict: any layout the engine could not have written fails the file.

const utf8 = new TextDecoder("utf-8", { fatal: true });

export function decodeRkyvString(buf: Buffer): { text: string; inline: boolean; len: number } {
  if (buf.length < 8) throw new Error("shorter than the 8-byte rkyv root");
  const root = buf.subarray(buf.length - 8);
  if ((root[0] & 0xc0) === 0x80) {
    const l = root.readUInt32LE(0);
    const len = (l & 0x3f) | ((l >>> 8) << 6);
    const off = root.readInt32LE(4);
    const start = buf.length - 8 + off;
    if (start !== 0) throw new Error(`string does not start the file (start=${start})`);
    if (len > buf.length - 8) throw new Error(`length ${len} exceeds the ${buf.length - 8} bytes before the root`);
    const pad = buf.length - 8 - len;
    if (pad > 3) throw new Error(`${pad} bytes between the string and the root, expected at most 3 of padding`);
    for (let i = len; i < buf.length - 8; i++) {
      if (buf[i] !== 0) throw new Error(`non-zero padding byte at ${i}`);
    }
    return { text: utf8.decode(buf.subarray(0, len)), inline: false, len };
  }
  if (buf.length !== 8) throw new Error("inline string root with extra bytes before it");
  let end = root.indexOf(0xff);
  if (end < 0) end = 8;
  for (let i = end; i < 8; i++) {
    if (root[i] !== 0xff) throw new Error("inline string padding is not 0xff");
  }
  return { text: utf8.decode(root.subarray(0, end)), inline: true, len: end };
}

// The inverse, used by the tests to build a store directory.
export function encodeRkyvString(text: string): Buffer {
  const bytes = Buffer.from(text, "utf8");
  if (bytes.length <= 8) {
    const root = Buffer.alloc(8, 0xff);
    bytes.copy(root);
    return root;
  }
  const padded = Math.ceil(bytes.length / 4) * 4;
  const out = Buffer.alloc(padded + 8, 0);
  bytes.copy(out);
  const len = bytes.length;
  out.writeUInt32LE(((len & 0x3f) | 0x80 | ((len & ~0x3f) << 2)) >>> 0, padded);
  out.writeInt32LE(-padded, padded + 4);
  return out;
}

// Top-level keys of a JSON object text, in file order, from a string- and
// escape-aware scan. `JSON.parse` reorders integer-like keys and folds
// duplicates; this does neither, so it both preserves IndexMap order and
// detects duplicate keys when compared with `Object.keys(parsed).length`.
export function topLevelKeys(text: string): string[] {
  const keys: string[] = [];
  let depth = 0;
  let inStr = false;
  let esc = false;
  let strStart = -1;
  let lastStr: string | null = null;
  for (let i = 0; i < text.length; i++) {
    const c = text.charCodeAt(i);
    if (inStr) {
      if (esc) esc = false;
      else if (c === 92) esc = true;
      else if (c === 34) {
        inStr = false;
        if (depth === 1) lastStr = text.slice(strStart, i + 1);
      }
      continue;
    }
    if (c === 34) {
      inStr = true;
      strStart = i;
    } else if (c === 123 || c === 91) depth++;
    else if (c === 125 || c === 93) depth--;
    else if (c === 58 && depth === 1) {
      if (lastStr === null) throw new Error(`':' without a key at offset ${i}`);
      keys.push(JSON.parse(lastStr) as string);
      lastStr = null;
    }
  }
  if (depth !== 0 || inStr) throw new Error("unbalanced JSON text");
  return keys;
}

export type DecodedScope = {
  scope: string;
  file: string;
  bytes: number;
  keys: string[];
  value: Record<string, unknown>;
};

export function decodeScopeFile(path: string): DecodedScope {
  const file = basename(path);
  if (!file.endsWith(".bin")) throw new Error(`${file}: not a .bin scope file`);
  const scope = decodeURIComponent(file.slice(0, -4));
  const buf = readFileSync(path);
  const { text, len } = decodeRkyvString(buf);
  if (text.length === 0) throw new Error(`${scope}: empty string`);
  if (text[0] !== "{" || text[text.length - 1] !== "}") {
    throw new Error(`${scope}: string is not a JSON object (starts ${JSON.stringify(text.slice(0, 8))})`);
  }
  // The plan's cross-check: the first '{' and the last '}' before the root
  // must be the string's bounds, so no padding byte looks like content. The
  // 8 root bytes are excluded: an encoded length or offset can legitimately
  // contain 0x7b / 0x7d.
  if (len > 8 && (buf.indexOf(0x7b) !== 0 || buf.lastIndexOf(0x7d, buf.length - 9) !== len - 1)) {
    throw new Error(`${scope}: bytes between the archived string and the root look like JSON`);
  }
  const value = JSON.parse(text) as unknown;
  if (typeof value !== "object" || value === null || Array.isArray(value)) {
    throw new Error(`${scope}: top level is not an object`);
  }
  const keys = topLevelKeys(text);
  const parsedCount = Object.keys(value).length;
  if (keys.length !== parsedCount) {
    throw new Error(`${scope}: ${keys.length} keys in the text but ${parsedCount} distinct - duplicate keys`);
  }
  return { scope, file, bytes: buf.length, keys, value: value as Record<string, unknown> };
}

// Rows of a JSON array streamed across chunk boundaries. Yields each
// top-level element's text (`[...]` or `{...}`), string- and escape-aware, so
// a `],[` inside a string never splits a row and a row may straddle chunks.
// The array grammar is enforced outside the rows: only whitespace, one `,`
// between rows and the closing `]` may appear there, and nothing but
// whitespace may follow the array. A primitive element, a trailing comma or
// trailing content fails the scan instead of being skipped.
export function* scanJsonArrayRows(chunks: Iterable<string>): IterableIterator<string> {
  // Outside a row: before `[`; a row or `]` may follow; `,` or `]` may
  // follow; a row must follow (after `,`); only whitespace may follow.
  let outer: "start" | "elem" | "sep" | "row" | "end" = "start";
  let depth = 0; // nesting inside the current row
  let inStr = false;
  let esc = false;
  let row: string[] | null = null;
  const ws = (c: number) => c === 32 || c === 9 || c === 10 || c === 13;
  for (const chunk of chunks) {
    let segStart = row ? 0 : -1;
    for (let i = 0; i < chunk.length; i++) {
      const c = chunk.charCodeAt(i);
      if (row) {
        if (inStr) {
          if (esc) esc = false;
          else if (c === 92) esc = true;
          else if (c === 34) inStr = false;
          continue;
        }
        if (c === 34) inStr = true;
        else if (c === 91 || c === 123) depth++;
        else if ((c === 93 || c === 125) && --depth === 0) {
          row.push(chunk.slice(segStart, i + 1));
          yield row.join("");
          row = null;
          segStart = -1;
          outer = "sep";
        }
        continue;
      }
      if (ws(c)) continue;
      const shown = JSON.stringify(chunk[i]);
      if (outer === "start") {
        if (c !== 91) throw new Error(`expected a JSON array, got ${shown}`);
        outer = "elem";
      } else if (outer === "elem" || outer === "row") {
        if (c === 93 && outer === "elem") outer = "end";
        else if (c === 91 || c === 123) {
          row = [];
          depth = 1;
          segStart = i;
        } else throw new Error(`unexpected ${shown} at array level: rows must be arrays or objects`);
      } else if (outer === "sep") {
        if (c === 44) outer = "row";
        else if (c === 93) outer = "end";
        else throw new Error(`expected ',' or ']' after a row, got ${shown}`);
      } else throw new Error(`content after the array: ${shown}`);
    }
    if (row && segStart >= 0) row.push(chunk.slice(segStart));
  }
  if (outer === "start") throw new Error("no array found");
  if (outer !== "end") throw new Error("array truncated");
}

// ---------------------------------------------------------------- scopes

const SKIP_EXACT = new Set(["mem:audit", "mem:health", KV.bm25Index]);
export function isSkippedScope(scope: string): boolean {
  return (
    SKIP_EXACT.has(scope) || scope.startsWith("mem:graph:") || scope.startsWith(`${KV.bm25Index}:`)
  );
}

const OBS_PREFIX = KV.observations("");

// sha256 over length-framed `key` and `json` per pair, in order (netstring
// style, so no byte inside a key or value can move a boundary). Used on the
// source and on a `SELECT ... ORDER BY seq` from the DB, so a dropped,
// duplicated or reordered row changes the digest.
export function digestPairs(pairs: Iterable<[string, string]>): string {
  const h = createHash("sha256");
  for (const [k, v] of pairs) {
    h.update(`${Buffer.byteLength(k)}:`).update(k).update(`${Buffer.byteLength(v)}:`).update(v);
  }
  return h.digest("hex");
}

const encode = (v: unknown): string => JSON.stringify(v) ?? "null";

function digestSource(d: DecodedScope): string {
  return digestPairs(d.keys.map((k) => [k, encode(d.value[k])] as [string, string]));
}

function countScope(db: DatabaseSync, scope: string): number {
  return (db.prepare("SELECT count(*) AS n FROM kv WHERE scope = ?").get(scope) as { n: number }).n;
}

function digestDb(db: DatabaseSync, scope: string): string {
  const rows = db
    .prepare("SELECT key, value FROM kv WHERE scope = ? ORDER BY seq")
    .iterate(scope) as Iterable<{ key: string; value: string }>;
  return digestPairs((function* () {
    for (const r of rows) yield [r.key, r.value] as [string, string];
  })());
}

type ShardManifest = { v: 1; generation?: string; shards: Array<{ scope: string; key: string; chars: number }>; chars: number };

function validManifest(m: unknown): m is ShardManifest {
  const x = m as ShardManifest;
  return (
    !!x && x.v === 1 && Array.isArray(x.shards) && x.shards.length > 0 && Number.isInteger(x.chars) && x.chars >= 0 &&
    x.shards.every((s) => s && typeof s.scope === "string" && typeof s.key === "string" && Number.isInteger(s.chars) && s.chars >= 0)
  );
}

// ---------------------------------------------------------------- report

export type ImportReport = {
  startedAt: string;
  finishedAt: string;
  src: string;
  out: string;
  published: boolean;
  files: number;
  scopes: Array<{ scope: string; keys: number; bytes: number; action: "imported" | "skipped" | "failed"; digest?: string; error?: string }>;
  totals: { importedScopes: number; importedKeys: number; skippedScopes: number; skippedKeys: number };
  observations: { scopes: number; rows: number; rawOrphans: number; rawOrphanSample: string[] };
  compressPending: number;
  auditDeletions: Array<{ timestamp: string; operation: string; functionId: string; targetIds: string[]; reason?: string }>;
  vectors: {
    generation: string | null;
    shards: number;
    imported: number;
    dims: number | null;
    delta: { addedCount: number; removedCount: number; added: string[]; removed: string[] } | null;
  } | null;
  bm25: {
    oldDocs: number | null;
    rebuilt: { docs: number; rows: number; skipped: number; ms: number; readMs: number; indexMs: number } | null;
    budgetMs: number;
    withinBudget: boolean | null;
    delta: { onlyInOldCount: number; onlyInNewCount: number; onlyInOld: string[]; onlyInNew: string[] } | null;
  };
  fill: FillReport | null;
  readiness: { hydrated: number; integrityCheck: string; digestRecheck: string } | null;
  warnings: string[];
  error?: string;
};

export type ImportOptions = {
  src: string;
  out: string;
  reportPath?: string;
  validateOnly?: boolean;
  fill?: boolean; // default true
  embeddingProvider?: EmbeddingProvider | null; // default: createEmbeddingProvider()
  allowSlowBm25?: boolean;
  replace?: boolean;
  budgetMs?: number; // default 30_000
  log?: (line: string) => void;
};

const CAP = 500;
const cap = (xs: string[]) => xs.slice(0, CAP);
const mib = (n: number) => (n / 1048576).toFixed(1);

// ---------------------------------------------------------------- import

export async function importStateStore(opts: ImportOptions): Promise<ImportReport> {
  const log = opts.log ?? ((line: string) => console.log(line));
  const budgetMs = opts.budgetMs ?? 30_000;
  const report: ImportReport = {
    startedAt: new Date().toISOString(),
    finishedAt: "",
    src: opts.src,
    out: opts.out,
    published: false,
    files: 0,
    scopes: [],
    totals: { importedScopes: 0, importedKeys: 0, skippedScopes: 0, skippedKeys: 0 },
    observations: { scopes: 0, rows: 0, rawOrphans: 0, rawOrphanSample: [] },
    compressPending: 0,
    auditDeletions: [],
    vectors: null,
    bm25: { oldDocs: null, rebuilt: null, budgetMs, withinBudget: null, delta: null },
    fill: null,
    readiness: null,
    warnings: [],
  };
  const warn = (msg: string) => {
    report.warnings.push(msg);
    log(`warn ${msg}`);
  };

  const tmp = `${opts.out}.tmp-${process.pid}`;
  const sidecars = (p: string) => [p, `${p}-wal`, `${p}-shm`];
  const cleanupTmp = () => {
    if (opts.validateOnly) return; // never created anything
    for (const f of sidecars(tmp)) if (existsSync(f)) unlinkSync(f);
  };
  let state: SqliteState | null = null;
  let prev: string | null = null; // --replace: the previous database, kept until the published one is checked

  try {
    if (!existsSync(opts.src) || !statSync(opts.src).isDirectory()) throw new Error(`--src ${opts.src} is not a directory`);
    if (existsSync(opts.out) && !opts.replace && !opts.validateOnly) {
      throw new Error(`--out ${opts.out} exists; pass --replace to rename over it`);
    }
    cleanupTmp(); // a stale temporary file from an earlier run with this pid

    const files = readdirSync(opts.src).filter((f) => f.endsWith(".bin")).sort((a, b) => {
      const sa = decodeURIComponent(a.slice(0, -4));
      const sb = decodeURIComponent(b.slice(0, -4));
      return sa < sb ? -1 : sa > sb ? 1 : 0;
    });
    report.files = files.length;
    if (files.length === 0) throw new Error(`no .bin files in ${opts.src}`);

    if (!opts.validateOnly) {
      state = new SqliteState(tmp);
      new SqliteVectorStore(state); // creates the vectors table
    }
    const db = state?.db ?? null;
    const insert = db?.prepare("INSERT INTO kv (scope, key, value, updated_at) VALUES (?, ?, ?, ?)") ?? null;
    const now = Date.now();

    // ---- pass over every scope file: decode, validate, import or skip.
    const indexScopeFiles = new Map<string, string>(); // mem:index:bm25* scope -> path
    let bm25Scope: DecodedScope | null = null;
    let failedFiles = 0;
    for (const file of files) {
      const path = join(opts.src, file);
      let d: DecodedScope;
      try {
        d = decodeScopeFile(path);
      } catch (err) {
        // Validate-only keeps going so one run lists every bad file (on a
        // live store a file caught mid-rewrite fails transiently; re-run to
        // tell those from real damage). An import stops at the first.
        if (!opts.validateOnly) throw err;
        const msg = err instanceof Error ? err.message : String(err);
        report.scopes.push({ scope: decodeURIComponent(file.slice(0, -4)), keys: 0, bytes: statSync(path).size, action: "failed", error: msg });
        failedFiles++;
        log(`FAIL ${msg}`);
        continue;
      }
      if (isSkippedScope(d.scope)) {
        report.scopes.push({ scope: d.scope, keys: d.keys.length, bytes: d.bytes, action: "skipped" });
        report.totals.skippedScopes++;
        report.totals.skippedKeys += d.keys.length;
        log(`skip ${d.scope}  ${d.keys.length} keys  ${mib(d.bytes)} MiB`);
        if (d.scope === "mem:audit") report.auditDeletions = auditDeletions(d);
        if (d.scope === KV.bm25Index) bm25Scope = d;
        if (d.scope.startsWith(`${KV.bm25Index}:`)) indexScopeFiles.set(d.scope, path);
        continue;
      }
      if (d.scope.startsWith(OBS_PREFIX)) {
        report.observations.scopes++;
        report.observations.rows += d.keys.length;
        for (const k of d.keys) {
          const o = d.value[k] as { title?: unknown } | null;
          if (!o || typeof o !== "object" || !o.title) {
            report.observations.rawOrphans++;
            if (report.observations.rawOrphanSample.length < 20) report.observations.rawOrphanSample.push(`${d.scope}/${k}`);
          }
        }
      }
      if (d.scope === KV.compressPending) report.compressPending = d.keys.length;
      const digest = digestSource(d);
      if (db && insert) {
        db.exec("BEGIN IMMEDIATE");
        try {
          for (const k of d.keys) insert.run(d.scope, k, encode(d.value[k]), now);
          db.exec("COMMIT");
        } catch (err) {
          db.exec("ROLLBACK");
          throw err;
        }
        const n = countScope(db, d.scope);
        if (n !== d.keys.length) throw new Error(`${d.scope}: ${n} rows after insert, ${d.keys.length} keys in the file`);
        const back = digestDb(db, d.scope);
        if (back !== digest) throw new Error(`${d.scope}: digest mismatch after insert (${digest} vs ${back})`);
      }
      report.scopes.push({ scope: d.scope, keys: d.keys.length, bytes: d.bytes, action: "imported", digest });
      report.totals.importedScopes++;
      report.totals.importedKeys += d.keys.length;
      log(`ok   ${d.scope}  ${d.keys.length} keys  ${mib(d.bytes)} MiB`);
    }

    // ---- vectors from the shards named by the manifest.
    const shardData = (scope: string): string => {
      const path = indexScopeFiles.get(scope);
      if (!path) throw new Error(`shard scope ${scope} has no file`);
      const d = decodeScopeFile(path);
      const data = d.value["data"];
      if (typeof data !== "string") throw new Error(`shard ${scope}: key "data" is not a string`);
      return data;
    };
    const loadShards = (m: ShardManifest, label: string): string[] => {
      const chunks: string[] = [];
      let total = 0;
      for (const s of m.shards) {
        const data = shardData(s.scope);
        if (data.length !== s.chars) throw new Error(`${label} shard ${s.scope}: ${data.length} chars, manifest says ${s.chars}`);
        chunks.push(data);
        total += data.length;
      }
      if (total !== m.chars) throw new Error(`${label}: shards total ${total} chars, manifest says ${m.chars}`);
      return chunks;
    };

    let importedVectorIds: Set<string> | null = null;
    let vectorDims: number | null = null;
    const vm = bm25Scope?.value["vectors:manifest"];
    if (vm === undefined) {
      warn("no vectors:manifest - no vectors imported; the fill pass will embed everything eligible");
    } else {
      if (!validManifest(vm)) throw new Error("vectors:manifest is invalid");
      const chunks = loadShards(vm, "vectors");
      importedVectorIds = new Set();
      const ins = db?.prepare(
        "INSERT INTO vectors (id, session_id, dims, embedding, input_hash, hash_state) VALUES (?, ?, ?, ?, '', 'legacy')",
      );
      db?.exec("BEGIN IMMEDIATE");
      try {
        for (const rowText of scanJsonArrayRows(chunks)) {
          const row = JSON.parse(rowText) as unknown;
          if (!Array.isArray(row) || row.length < 2) throw new Error(`vector row is not [id, entry]: ${rowText.slice(0, 80)}`);
          const [id, entry] = row as [unknown, { embedding?: unknown; sessionId?: unknown }];
          if (typeof id !== "string" || !id) throw new Error(`vector row with a non-string id: ${rowText.slice(0, 80)}`);
          if (!entry || typeof entry.embedding !== "string" || typeof entry.sessionId !== "string") {
            throw new Error(`vector ${id}: entry shape invalid`);
          }
          if (!/^[A-Za-z0-9+/]*={0,2}$/.test(entry.embedding) || entry.embedding.length % 4 !== 0) {
            throw new Error(`vector ${id}: embedding is not base64`);
          }
          const blob = Buffer.from(entry.embedding, "base64");
          if (blob.length === 0 || blob.length % 4 !== 0) throw new Error(`vector ${id}: ${blob.length} bytes is not a Float32 array`);
          const dims = blob.length / 4;
          if (vectorDims === null) vectorDims = dims;
          else if (dims !== vectorDims) throw new Error(`vector ${id}: dims ${dims}, previous rows ${vectorDims}`);
          if (importedVectorIds.has(id)) throw new Error(`vector ${id}: duplicate id`);
          importedVectorIds.add(id);
          ins?.run(id, entry.sessionId, dims, blob);
        }
        db?.exec("COMMIT");
      } catch (err) {
        db?.exec("ROLLBACK");
        throw err;
      }
      report.vectors = {
        generation: vm.generation ?? null,
        shards: vm.shards.length,
        imported: importedVectorIds.size,
        dims: vectorDims,
        delta: null,
      };
      log(`ok   vectors  ${importedVectorIds.size} rows from ${vm.shards.length} shards, dims ${vectorDims}`);
    }

    // ---- old BM25 doc ids, for the corpus delta (acceptance, step 9.7a).
    let oldBm25Ids: Set<string> | null = null;
    const bm = bm25Scope?.value["data:manifest"];
    if (bm !== undefined) {
      try {
        if (!validManifest(bm)) throw new Error("data:manifest is invalid");
        const json = loadShards(bm, "bm25").join("");
        const data = JSON.parse(json) as { entries?: unknown };
        if (!Array.isArray(data.entries)) throw new Error("serialized BM25 has no entries");
        oldBm25Ids = new Set((data.entries as Array<[string, unknown]>).map((e) => e[0]));
        report.bm25.oldDocs = oldBm25Ids.size;
        log(`ok   old BM25  ${oldBm25Ids.size} docs from ${bm.shards.length} shards`);
      } catch (err) {
        warn(`old BM25 index unreadable, corpus delta unavailable: ${err instanceof Error ? err.message : String(err)}`);
      }
    } else {
      warn("no data:manifest - old BM25 doc ids unavailable, corpus delta unavailable");
    }

    if (opts.validateOnly || !state || !db) {
      report.finishedAt = new Date().toISOString();
      if (failedFiles > 0) throw new Error(`${failedFiles} scope file(s) failed to decode (see report.scopes[].error)`);
      log("validate-only: nothing written");
      return report;
    }

    // ---- BM25 rebuild timing on the imported content (step 4 gate).
    {
      const t0 = performance.now();
      const r = await rebuildBm25FromContent(db);
      const ms = Math.round(performance.now() - t0);
      report.bm25.rebuilt = { docs: r.docs, rows: r.rows, skipped: r.skipped, ms, readMs: Math.round(r.readMs), indexMs: Math.round(r.indexMs) };
      report.bm25.withinBudget = ms <= budgetMs;
      log(`ok   BM25 rebuilt  ${r.docs} docs of ${r.rows} rows in ${ms} ms (read ${Math.round(r.readMs)}, index ${Math.round(r.indexMs)})${r.skipped ? `, ${r.skipped} skipped` : ""}`);
      if (!report.bm25.withinBudget) {
        const msg = `BM25 rebuild took ${ms} ms, over the ${budgetMs} ms budget`;
        if (!opts.allowSlowBm25) throw new Error(`${msg}; pass --allow-slow-bm25 to proceed knowingly`);
        warn(msg);
      }
      if (oldBm25Ids) {
        const newIds = new Set(getSearchIndex().ids());
        const onlyInOld = [...oldBm25Ids].filter((id) => !newIds.has(id));
        const onlyInNew = [...newIds].filter((id) => !oldBm25Ids!.has(id));
        report.bm25.delta = { onlyInOldCount: onlyInOld.length, onlyInNewCount: onlyInNew.length, onlyInOld: cap(onlyInOld), onlyInNew: cap(onlyInNew) };
        log(`ok   BM25 delta  only-in-old ${onlyInOld.length}, only-in-new ${onlyInNew.length}`);
      }
      getSearchIndex().clear();
    }

    // ---- fill pass (step 5) once, inside the importer.
    if (opts.fill !== false) {
      const provider = opts.embeddingProvider === undefined ? createEmbeddingProvider() : opts.embeddingProvider;
      if (!provider) throw new Error("no embedding provider in the environment; export the daemon's env or pass --no-fill");
      if (vectorDims !== null && provider.dimensions !== vectorDims) {
        throw new Error(`imported vectors have ${vectorDims} dims, provider ${provider.name} declares ${provider.dimensions}`);
      }
      const vectors = new SqliteVectorStore(state);
      const vi = new VectorIndex();
      vi.attachStore(vectors);
      const hydrated = vectors.hydrate(vi);
      setVectorIndex(vi);
      setEmbeddingProvider(provider);
      setInprocStores(state, vectors);
      try {
        const fill = await createIndexFill(state, vectors, vi).run();
        report.fill = fill;
        log(`ok   fill pass  expected ${fill.expected}, present ${fill.present}, embedded ${fill.embedded}, pruned ${fill.pruned}, failed ${fill.failed}${fill.aborted ? ", ABORTED" : ""} (hydrated ${hydrated})`);
        if (fill.aborted) throw new Error("fill pass aborted (embedding provider failing)");
        if (fill.failed > 0) warn(`fill pass: ${fill.failed} jobs failed; the hourly pass retries them after cutover`);
        const after = new Set(rowIds(db));
        const base = importedVectorIds ?? new Set<string>();
        const added = [...after].filter((id) => !base.has(id));
        const removed = [...base].filter((id) => !after.has(id));
        if (report.vectors) report.vectors.delta = { addedCount: added.length, removedCount: removed.length, added: cap(added), removed: cap(removed) };
        else report.vectors = { generation: null, shards: 0, imported: 0, dims: provider.dimensions, delta: { addedCount: added.length, removedCount: 0, added: cap(added), removed: [] } };
      } finally {
        setInprocStores(null, null);
        setEmbeddingProvider(null);
        setVectorIndex(null);
      }
    } else {
      warn("fill pass skipped (--no-fill); eligible content without a vector stays unembedded until the daemon's hourly pass");
    }

    // ---- verify the temporary file: checkpoint + close, integrity_check on
    // a fresh read-only connection, then the readiness dry run (hydrate,
    // per-scope row count and digest re-check). Every check runs before the
    // rename, so a failure deletes the temporary file and the old store is
    // untouched.
    state.close();
    state = null;
    // After a TRUNCATE checkpoint and close, a WAL with frames (more than its
    // 32-byte header) means data that did not reach the main file. Re-opening
    // (even read-only) re-creates an empty -wal/-shm pair; they hold nothing
    // and must not be left behind.
    const dropSidecars = (base: string, when: string) => {
      for (const f of [`${base}-wal`, `${base}-shm`]) {
        if (!existsSync(f)) continue;
        if (f.endsWith("-wal") && statSync(f).size > 32) throw new Error(`${basename(f)} has frames ${when}`);
        unlinkSync(f);
      }
    };
    dropSidecars(tmp, "after close");
    const ro = new DatabaseSync(tmp, { readOnly: true });
    let integrity: string;
    try {
      const rows = ro.prepare("PRAGMA integrity_check").all() as Array<{ integrity_check: string }>;
      integrity = rows.map((r) => r.integrity_check).join("; ");
    } finally {
      ro.close();
    }
    dropSidecars(tmp, "after the read-only integrity check");
    if (integrity !== "ok") throw new Error(`integrity_check: ${integrity}`);
    {
      const pub = new SqliteState(tmp);
      let hydrated: number;
      try {
        const vectors = new SqliteVectorStore(pub);
        hydrated = vectors.hydrate(new VectorIndex());
        const count = vectors.count();
        if (hydrated !== count) throw new Error(`hydrated ${hydrated} vectors, table has ${count}`);
        for (const s of report.scopes) {
          if (s.action !== "imported") continue;
          const n = countScope(pub.db, s.scope);
          if (n !== s.keys) throw new Error(`${s.scope}: ${n} rows on re-open, ${s.keys} imported`);
          const back = digestDb(pub.db, s.scope);
          if (back !== s.digest) throw new Error(`${s.scope}: digest mismatch on re-open`);
        }
      } finally {
        pub.close();
      }
      dropSidecars(tmp, "after the readiness dry run");
      report.readiness = { hydrated, integrityCheck: integrity, digestRecheck: "ok" };
      log(`ok   readiness  hydrated ${hydrated} vectors, ${report.totals.importedScopes} scope counts and digests match`);
    }

    // ---- publish by rename. The destination's sidecars are checked first,
    // with or without its main file: a WAL with frames or a rollback journal
    // means a database in use or not closed cleanly, whose data the rename
    // would orphan (or SQLite would replay into the new file), so that is
    // refused; nothing of the destination is touched before the rename. The
    // report is written before the rename, so a bad path or a full disk
    // fails while the temporary file can still be deleted; it is rewritten
    // once the published file has been checked. With --replace the previous
    // database is kept beside the new one until then.
    const destWal = `${opts.out}-wal`;
    if (existsSync(destWal) && statSync(destWal).size > 32) {
      throw new Error(`${basename(destWal)} has uncheckpointed frames: the database at --out is in use or was not closed cleanly, refusing to publish over it`);
    }
    if (existsSync(`${opts.out}-journal`)) throw new Error(`${basename(opts.out)}-journal exists (hot rollback journal), refusing to publish over it`);
    if (existsSync(opts.out) && !opts.replace) throw new Error(`--out ${opts.out} appeared during the import`);
    const reportPath = opts.reportPath ?? join(dirname(opts.out), "import-report.json");
    const writeReport = () => writeFileSync(reportPath, JSON.stringify(report, null, 2) + "\n");
    writeReport();
    if (existsSync(opts.out)) {
      const aside = `${opts.out}.replaced-${process.pid}`;
      if (existsSync(aside)) throw new Error(`${basename(aside)} exists (left by an earlier --replace): move it away first`);
      renameSync(opts.out, aside);
      prev = aside;
    }
    renameSync(tmp, opts.out);
    report.published = true;
    dropSidecars(opts.out, "left by the replaced file");
    log(`ok   published ${opts.out}`);
    {
      // The renamed file opens and holds what was imported.
      const check = new DatabaseSync(opts.out, { readOnly: true });
      try {
        const n = (check.prepare("SELECT count(*) AS n FROM kv").get() as { n: number }).n;
        if (n !== report.totals.importedKeys) throw new Error(`published file holds ${n} rows, imported ${report.totals.importedKeys}`);
      } finally {
        check.close();
      }
      dropSidecars(opts.out, "after the published-file check");
    }
    report.finishedAt = new Date().toISOString();
    writeReport();
    log(`ok   report ${reportPath}`);
    if (prev) {
      // Last: every fallible step is done, the replaced database can go.
      unlinkSync(prev);
      prev = null;
    }
    return report;
  } catch (err) {
    report.error = err instanceof Error ? err.message : String(err);
    report.finishedAt = new Date().toISOString();
    try {
      state?.close();
    } catch {}
    // The previous database first (it is the data), the temporary file after;
    // neither cleanup failure may hide the other or the original error.
    const note = (what: string, e: unknown) => {
      report.error += ` (${what}: ${e instanceof Error ? e.message : String(e)})`;
    };
    if (prev && existsSync(prev)) {
      // The rename of the temporary file failed: the previous database goes
      // back; after a successful rename it stays beside the new one.
      try {
        if (!existsSync(opts.out)) renameSync(prev, opts.out);
        else report.error += ` (previous database kept at ${prev})`;
      } catch (e) {
        note(`previous database left at ${prev}`, e);
      }
    }
    if (!report.published) {
      try {
        cleanupTmp();
      } catch (e) {
        note(`temporary file left at ${tmp}`, e);
      }
    }
    log(`FAIL ${report.error}`);
    throw Object.assign(err instanceof Error ? err : new Error(String(err)), { report });
  }
}

function* rowIds(db: DatabaseSync): IterableIterator<string> {
  for (const r of db.prepare("SELECT id FROM vectors").iterate() as Iterable<{ id: string }>) yield r.id;
}

// Delete-class audit entries, oldest first: what the cutover reconciliation
// subtracts from the journal's acknowledged ids (step 9.3). Read here because
// the scope itself is not imported.
function auditDeletions(d: DecodedScope): ImportReport["auditDeletions"] {
  const out: ImportReport["auditDeletions"] = [];
  for (const k of d.keys) {
    const e = d.value[k] as { timestamp?: string; operation?: string; functionId?: string; targetIds?: unknown; details?: { reason?: string } } | null;
    if (!e || typeof e !== "object") continue;
    const fn = e.functionId ?? "";
    if (fn === "mem::index-persistence") continue;
    const deleteClass = e.operation === "forget" || e.operation === "delete" || /forget|delete|evict|retention|governance/i.test(fn);
    if (!deleteClass) continue;
    out.push({
      timestamp: e.timestamp ?? "",
      operation: e.operation ?? "",
      functionId: fn,
      targetIds: Array.isArray(e.targetIds) ? (e.targetIds as string[]) : [],
      reason: e.details?.reason,
    });
  }
  return out.sort((a, b) => (a.timestamp < b.timestamp ? -1 : a.timestamp > b.timestamp ? 1 : 0));
}

// ---------------------------------------------------------------- cli

async function main(): Promise<void> {
  const { values } = parseArgs({
    options: {
      src: { type: "string" },
      out: { type: "string" },
      report: { type: "string" },
      "validate-only": { type: "boolean", default: false },
      "no-fill": { type: "boolean", default: false },
      "allow-slow-bm25": { type: "boolean", default: false },
      replace: { type: "boolean", default: false },
    },
  });
  if (!values.src || !values.out) {
    console.error("usage: import-state-store --src <state_store.db dir> --out <agentmemory.sqlite> [--report file] [--validate-only] [--no-fill] [--allow-slow-bm25] [--replace]");
    process.exitCode = 2;
    return;
  }
  try {
    const report = await importStateStore({
      src: values.src,
      out: values.out,
      reportPath: values.report,
      validateOnly: values["validate-only"],
      fill: !values["no-fill"],
      allowSlowBm25: values["allow-slow-bm25"],
      replace: values.replace,
    });
    if (values["validate-only"]) console.log(JSON.stringify(report, null, 2));
  } catch (err) {
    // exitCode, not exit(): a large report on a pipe must finish writing.
    const report = (err as { report?: ImportReport }).report;
    if (report) console.error(JSON.stringify(report, null, 2));
    process.exitCode = 1;
  }
}

const invokedDirectly = process.argv[1] && /import-state-store\.(mjs|ts|js)$/.test(process.argv[1]);
if (invokedDirectly) void main();
