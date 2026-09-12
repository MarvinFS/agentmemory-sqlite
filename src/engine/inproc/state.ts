// SQLite-backed replacement for the iii engine's five `state::*` functions.
//
// Semantics are copied from iii 0.11.2 (`engine/src/builtins/kv.rs` +
// `engine/src/workers/state/state.rs`), not from the SDK's type declarations,
// because the daemon depends on engine behaviour the types do not describe:
//
//   - a `set` on an existing key keeps that key's insertion position
//     (`IndexMap::insert`), which `remember.ts` supersession and BM25 tie
//     order both read as "insertion order";
//   - `delete` preserves the order of the remaining keys (`shift_remove`);
//   - `update` on a missing key auto-creates it as `{}` and never errors;
//   - an `UpdateOp` `path` is a FLAT top-level key, never a dotted path
//     (`map.insert(path.0, value)`), and an empty path replaces the value;
//   - state triggers fire after the write with
//     `{type, event_type, scope, key, old_value, new_value}`.
//
// `seq` is the rowid, so it is monotonic across inserts and untouched by an
// in-place update; `ORDER BY seq` reproduces the engine's insertion order.
import { DatabaseSync } from "node:sqlite";

export type StateEventType =
  | "state:created"
  | "state:updated"
  | "state:deleted";

export type StateEvent = {
  type: "state";
  event_type: StateEventType;
  scope: string;
  key: string;
  old_value: unknown;
  new_value: unknown;
};

export type UpdateOp = {
  type: string;
  path?: string;
  value?: unknown;
  by?: number;
};

// The op names iii accepts. Anything else fails deserialization inside the
// engine and surfaces as an UPDATE_ERROR, so reject loudly rather than
// silently dropping a write shape production never had.
const KNOWN_OPS = new Set(["set", "merge", "increment", "decrement", "remove"]);

const SCHEMA = `
CREATE TABLE IF NOT EXISTS kv (
  seq        INTEGER PRIMARY KEY,
  scope      TEXT    NOT NULL,
  key        TEXT    NOT NULL,
  value      TEXT    NOT NULL,
  updated_at INTEGER NOT NULL,
  UNIQUE(scope, key)
);
CREATE INDEX IF NOT EXISTS kv_scope_seq ON kv(scope, seq);
`;

function encode(value: unknown): string {
  const json = JSON.stringify(value);
  // `undefined` has no JSON encoding; iii's Value has no undefined either, so
  // it would have arrived as null over the wire.
  return json === undefined ? "null" : json;
}

function isPlainObject(v: unknown): v is Record<string, unknown> {
  return typeof v === "object" && v !== null && !Array.isArray(v);
}

// `obj[key] = v` walks the prototype chain: a key of `__proto__` reassigns the
// prototype instead of storing a property, and JSON.stringify then drops it —
// the write is silently lost. iii's serde_json map stores it as an ordinary
// key, so define it as an own property.
function setKey(obj: Record<string, unknown>, key: string, value: unknown): void {
  Object.defineProperty(obj, key, {
    value,
    writable: true,
    enumerable: true,
    configurable: true,
  });
}

function applyOps(current: unknown, ops: UpdateOp[]): unknown {
  let value = current;
  for (const op of ops) {
    if (!op || typeof op.type !== "string" || !KNOWN_OPS.has(op.type)) {
      throw new Error(
        `state::update: unsupported op ${JSON.stringify(op?.type ?? op)}`,
      );
    }
    const path = op.path ?? "";
    switch (op.type) {
      case "set":
        if (path === "") {
          if (op.value !== undefined) value = op.value;
        } else if (isPlainObject(value)) {
          setKey(value, path, op.value === undefined ? null : op.value);
        }
        break;
      case "merge":
        // iii supports root-level merge only; a pathful merge is a no-op there.
        if (path === "" && isPlainObject(value) && isPlainObject(op.value)) {
          for (const k of Object.keys(op.value)) setKey(value, k, op.value[k]);
        }
        break;
      case "increment":
      case "decrement": {
        if (!isPlainObject(value)) break;
        const by = typeof op.by === "number" ? op.by : 0;
        const existing = Object.hasOwn(value, path) ? value[path] : undefined;
        if (typeof existing === "number" && Number.isInteger(existing)) {
          setKey(value, path, op.type === "increment" ? existing + by : existing - by);
        } else if (existing === undefined) {
          setKey(value, path, op.type === "increment" ? by : -by);
        } else {
          // iii replaces a non-integer with `by` on increment, 0 on decrement.
          setKey(value, path, op.type === "increment" ? by : 0);
        }
        break;
      }
      case "remove":
        if (isPlainObject(value)) Reflect.deleteProperty(value, path);
        break;
    }
  }
  return value;
}

export class SqliteState {
  readonly db: DatabaseSync;
  private listener: ((event: StateEvent) => void) | null = null;
  // Scopes with a registered `type:"state"` trigger. Everything else skips the
  // emit entirely, so the payload clone below costs nothing on the hot write
  // paths (memories, observations) that nobody subscribes to.
  private watched = new Set<string>();
  private watchAll = false;
  // Work deferred until the outermost COMMIT succeeds: state-trigger
  // deliveries (iii fires them after the write lands; emitting mid-transaction
  // would let a subscriber act on a row a later ROLLBACK erases) and the
  // in-memory vector-index updates that must never run ahead of their rows.
  // Callbacks registered by the open transaction; moved to `committed` on
  // COMMIT, dropped on ROLLBACK.
  private pending: Array<() => void> = [];
  // Committed callbacks awaiting delivery, drained FIFO and never re-entered:
  // a callback that opens its own transaction queues behind the work already
  // committed ahead of it, so map state follows commit order.
  private committed: Array<() => void> = [];
  private draining = false;
  // Runs inside the writing transaction on every set/update, before COMMIT.
  // The vector store uses it to invalidate imported (`legacy`) rows whose
  // content was just rewritten, in the same transaction as the rewrite.
  private writeHook: ((scope: string, key: string) => void) | null = null;

  constructor(path: string) {
    this.db = new DatabaseSync(path);
    this.db.exec("PRAGMA journal_mode = WAL");
    // Authoritative store, tiny write rate: durability of an acknowledged
    // write beats the fsync it costs.
    this.db.exec("PRAGMA synchronous = FULL");
    this.db.exec("PRAGMA foreign_keys = ON");
    this.db.exec(SCHEMA);
  }

  onEvent(listener: (event: StateEvent) => void): void {
    this.listener = listener;
  }

  onWrite(hook: (scope: string, key: string) => void): void {
    this.writeHook = hook;
  }

  // Called by the shim for each `type:"state"` trigger. A trigger without a
  // scope in its config watches every scope, as iii's scope filter does.
  watchScope(scope?: string): void {
    if (scope === undefined) this.watchAll = true;
    else this.watched.add(scope);
  }

  private emit(event: StateEvent): void {
    if (!this.listener) return;
    if (!this.watchAll && !this.watched.has(event.scope)) return;
    // set/update/delete already decode `old_value`/`new_value` from the stored
    // rows, so they never alias the caller's input — but they DO hand the same
    // objects back as the call's return value. Snapshot once more here so the
    // trigger payload is nobody else's object in either direction. Only
    // watched scopes reach this line, so the cost stays off the hot paths.
    const detached: StateEvent = {
      ...event,
      old_value: JSON.parse(encode(event.old_value)),
      new_value: JSON.parse(encode(event.new_value)),
    };
    this.afterCommit(() => this.deliver(detached));
  }

  // Runs `fn` once the enclosing transaction has committed, or immediately
  // when none is open. Discarded on ROLLBACK.
  afterCommit(fn: () => void): void {
    if (this.db.isTransaction) {
      this.pending.push(fn);
    } else {
      // Through the queue even with no transaction open: a callback that runs
      // while an earlier commit is still draining must land behind it.
      this.committed.push(fn);
      this.drain();
    }
  }

  private deliver(event: StateEvent): void {
    try {
      this.listener?.(event);
    } catch {
      // A trigger must never fail the write that produced it; iii spawns the
      // trigger fan-out on a separate task for the same reason.
    }
  }

  // Runs `fn` inside a transaction. Nested calls join the outer one so the
  // vector write paths can compose with these without a second BEGIN (SQLite
  // has no nested transactions).
  transaction<T>(fn: () => T): T {
    if (this.db.isTransaction) return fn();
    this.db.exec("BEGIN IMMEDIATE");
    let out: T;
    try {
      out = fn();
      this.db.exec("COMMIT");
    } catch (err) {
      try {
        this.db.exec("ROLLBACK");
      } catch {}
      this.pending.length = 0;
      throw err;
    }
    this.committed.push(...this.pending);
    this.pending = [];
    this.drain();
    return out;
  }

  private drain(): void {
    if (this.draining) return;
    this.draining = true;
    try {
      for (let fn = this.committed.shift(); fn; fn = this.committed.shift()) {
        try {
          fn();
        } catch {
          // Deferred work must never fail the write that already committed.
        }
      }
    } finally {
      this.draining = false;
    }
  }

  // iii distinguishes a missing key (`None`) from a stored JSON null, and that
  // distinction decides `state:created` vs `state:updated` and whether a
  // delete fires a trigger at all. Keep it internally; `get()` flattens both
  // to null the way `state::get` does over the wire.
  private read(scope: string, key: string): { found: boolean; value: unknown } {
    const row = this.db
      .prepare("SELECT value FROM kv WHERE scope = ? AND key = ?")
      .get(scope, key) as { value?: string } | undefined;
    if (row === undefined) return { found: false, value: null };
    return { found: true, value: JSON.parse(row.value as string) };
  }

  get(scope: string, key: string): unknown {
    return this.read(scope, key).value;
  }

  list(scope: string): unknown[] {
    const rows = this.db
      .prepare("SELECT value FROM kv WHERE scope = ? ORDER BY seq")
      .all(scope) as Array<{ value: string }>;
    return rows.map((r) => JSON.parse(r.value));
  }

  // Raw upsert used by set/update. Keeps the existing row's `seq`. Returns the
  // stored encoding so callers can hand back exactly what landed in the row
  // instead of the caller's own object.
  private put(scope: string, key: string, value: unknown): string {
    const json = encode(value);
    this.db
      .prepare(
        `INSERT INTO kv (scope, key, value, updated_at) VALUES (?, ?, ?, ?)
         ON CONFLICT(scope, key) DO UPDATE SET
           value = excluded.value, updated_at = excluded.updated_at`,
      )
      .run(scope, key, json, Date.now());
    this.writeHook?.(scope, key);
    return json;
  }

  set(scope: string, key: string, value: unknown): {
    old_value: unknown;
    new_value: unknown;
  } {
    const { found, old_value, new_value } = this.transaction(() => {
      const prev = this.read(scope, key);
      const json = this.put(scope, key, value);
      // Decode what was stored: iii answers with the serde_json Value it
      // persisted, so neither the return nor the trigger payload can alias
      // (and later be mutated through) the caller's object.
      return {
        found: prev.found,
        old_value: prev.value,
        new_value: JSON.parse(json),
      };
    });
    this.emit({
      type: "state",
      event_type: found ? "state:updated" : "state:created",
      scope,
      key,
      old_value,
      new_value,
    });
    return { old_value, new_value };
  }

  // set() per row in one transaction, so a chunk costs one fsync instead of
  // one per row (a decay run was 45k of them - day-0 soak finding). The inner
  // set() calls join this transaction, and their events go out through
  // afterCommit, so each row still fires its created/updated event after the
  // COMMIT and none fire on a rollback. Nothing here awaits, so BEGIN is never
  // held across a turn of the event loop; the caller bounds the chunk
  // (StateKV.setMany, 100 rows).
  setMany(scope: string, entries: Array<{ key: string; value: unknown }>): number {
    return this.transaction(() => {
      for (const { key, value } of entries) this.set(scope, key, value);
      return entries.length;
    });
  }

  update(scope: string, key: string, ops: UpdateOp[]): {
    old_value: unknown;
    new_value: unknown;
  } {
    const result = this.transaction(() => {
      const prev = this.read(scope, key);
      // iii auto-creates a missing key as an empty object, appended at the end.
      const base = prev.found ? structuredClone(prev.value) : {};
      const json = this.put(scope, key, applyOps(base, ops));
      return {
        found: prev.found,
        old_value: prev.value,
        new_value: JSON.parse(json),
      };
    });
    this.emit({
      type: "state",
      event_type: result.found ? "state:updated" : "state:created",
      scope,
      key,
      old_value: result.old_value,
      new_value: result.new_value,
    });
    return { old_value: result.old_value, new_value: result.new_value };
  }

  delete(scope: string, key: string): unknown {
    const { found, old_value } = this.transaction(() => {
      const prev = this.read(scope, key);
      this.db
        .prepare("DELETE FROM kv WHERE scope = ? AND key = ?")
        .run(scope, key);
      return { found: prev.found, old_value: prev.value };
    });
    // iii only fires the trigger when a row was actually removed.
    if (found) {
      this.emit({
        type: "state",
        event_type: "state:deleted",
        scope,
        key,
        old_value,
        new_value: null,
      });
    }
    return old_value;
  }

  close(): void {
    try {
      this.db.exec("PRAGMA wal_checkpoint(TRUNCATE)");
    } catch {}
    this.db.close();
  }
}

// The store is synchronous and an `async` wrapper settles in a microtask, so
// a loop of awaited state calls never lets the event loop turn: 45k decay
// writes held every request for 95 s (day-0 soak finding). After each call,
// once ~20 ms of wall time has passed since the last yield, park on
// setImmediate so queued I/O gets a turn. The store call itself still runs
// before the first await, which is what sdk.ts relies on for payload
// isolation.
const BUSY_BUDGET_MS = 20;
let lastYield = performance.now();
async function cooperate<T>(result: T): Promise<T> {
  if (performance.now() - lastYield >= BUSY_BUDGET_MS) {
    await new Promise<void>((resolve) => setImmediate(resolve));
    lastYield = performance.now();
  }
  return result;
}

// The function handlers, in the exact shapes `src/state/kv.ts` sends and the
// engine returns (`state::set-many` is inproc-only; kv.ts never sends it to
// iii). Registered on the shim by `sdk.ts`.
export function stateFunctions(
  store: SqliteState,
): Record<string, (payload: any) => Promise<unknown>> {
  return {
    "state::get": async (p) => cooperate(store.get(p.scope, p.key)),
    "state::set": async (p) => cooperate(store.set(p.scope, p.key, p.value)),
    "state::update": async (p) => cooperate(store.update(p.scope, p.key, p.ops ?? [])),
    "state::delete": async (p) => cooperate(store.delete(p.scope, p.key)),
    "state::list": async (p) => cooperate(store.list(p.scope)),
    "state::set-many": async (p) => cooperate(store.setMany(p.scope, p.entries ?? [])),
  };
}
