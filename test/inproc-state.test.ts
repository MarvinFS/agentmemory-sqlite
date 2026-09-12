import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";

import {
  SqliteState,
  stateFunctions,
  type StateEvent,
} from "../src/engine/inproc/state.js";

// Behaviour asserted here is iii 0.11.2's, read from
// engine/src/builtins/kv.rs and engine/src/workers/state/state.rs. The
// daemon depends on it; a change that "looks nicer" here is a data bug.
describe("inproc state store", () => {
  let dir: string;
  let store: SqliteState;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "am-inproc-state-"));
    store = new SqliteState(join(dir, "state.sqlite"));
  });

  afterEach(() => {
    store.close();
    rmSync(dir, { recursive: true, force: true });
  });

  it("round-trips all five operations", async () => {
    const fns = stateFunctions(store);

    expect(await fns["state::get"]({ scope: "s", key: "missing" })).toBeNull();

    await fns["state::set"]({ scope: "s", key: "a", value: { n: 1 } });
    expect(await fns["state::get"]({ scope: "s", key: "a" })).toEqual({ n: 1 });

    await fns["state::update"]({
      scope: "s",
      key: "a",
      ops: [{ type: "set", path: "n", value: 2 }],
    });
    expect(await fns["state::get"]({ scope: "s", key: "a" })).toEqual({ n: 2 });

    expect(await fns["state::list"]({ scope: "s" })).toEqual([{ n: 2 }]);

    await fns["state::delete"]({ scope: "s", key: "a" });
    expect(await fns["state::get"]({ scope: "s", key: "a" })).toBeNull();
    expect(await fns["state::list"]({ scope: "s" })).toEqual([]);
  });

  it("state::set and state::update return {old_value, new_value}", async () => {
    const fns = stateFunctions(store);
    expect(await fns["state::set"]({ scope: "s", key: "a", value: 1 })).toEqual({
      old_value: null,
      new_value: 1,
    });
    expect(await fns["state::set"]({ scope: "s", key: "a", value: 2 })).toEqual({
      old_value: 1,
      new_value: 2,
    });
    expect(
      await fns["state::update"]({
        scope: "s",
        key: "o",
        ops: [{ type: "set", path: "x", value: true }],
      }),
    ).toEqual({ old_value: null, new_value: { x: true } });
  });

  it("list returns insertion order, and a re-set keeps its position", () => {
    store.set("s", "a", 1);
    store.set("s", "b", 2);
    store.set("s", "c", 3);
    expect(store.list("s")).toEqual([1, 2, 3]);

    // IndexMap::insert semantics: updating `a` must NOT move it to the end.
    store.set("s", "a", 99);
    expect(store.list("s")).toEqual([99, 2, 3]);
  });

  it("delete preserves the order of the remaining keys", () => {
    store.set("s", "a", 1);
    store.set("s", "b", 2);
    store.set("s", "c", 3);
    store.delete("s", "b");
    expect(store.list("s")).toEqual([1, 3]);
    // A re-added key is appended, not restored to its old slot.
    store.set("s", "b", 4);
    expect(store.list("s")).toEqual([1, 3, 4]);
  });

  it("scopes are isolated", () => {
    store.set("one", "a", 1);
    store.set("two", "a", 2);
    expect(store.list("one")).toEqual([1]);
    expect(store.list("two")).toEqual([2]);
    expect(store.get("one", "a")).toBe(1);
  });

  it("update paths are flat top-level keys, never dotted nesting", () => {
    store.set("s", "k", { a: { b: 1 } });
    store.update("s", "k", [{ type: "set", path: "a.b", value: 2 }]);
    // iii does `map.insert(path, value)` — a literal "a.b" key, and the real
    // nested value is untouched.
    expect(store.get("s", "k")).toEqual({ a: { b: 1 }, "a.b": 2 });
  });

  it("update with an empty path replaces the whole value", () => {
    store.set("s", "k", { a: 1 });
    store.update("s", "k", [{ type: "set", path: "", value: { b: 2 } }]);
    expect(store.get("s", "k")).toEqual({ b: 2 });
  });

  it("update auto-creates a missing key as an empty object", () => {
    store.update("s", "fresh", [
      { type: "set", path: "status", value: "completed" },
    ]);
    expect(store.get("s", "fresh")).toEqual({ status: "completed" });
  });

  it("update applies multiple ops in order", () => {
    store.set("s", "k", { name: "A", counter: 0 });
    store.update("s", "k", [
      { type: "set", path: "name", value: "B" },
      { type: "increment", path: "counter", by: 5 },
      { type: "decrement", path: "counter", by: 2 },
      { type: "merge", value: { extra: "field" } },
      { type: "remove", path: "name" },
    ]);
    expect(store.get("s", "k")).toEqual({ counter: 3, extra: "field" });
  });

  it("rejects an unknown update op and leaves the value untouched", () => {
    store.set("s", "k", { a: 1 });
    expect(() =>
      store.update("s", "k", [{ type: "append", path: "a", value: 2 }]),
    ).toThrow(/unsupported op/);
    expect(store.get("s", "k")).toEqual({ a: 1 });
  });

  it("emits iii-shaped state events for set, update and delete", () => {
    const events: StateEvent[] = [];
    store.onEvent((e) => events.push(e));
    store.watchScope("sessions");

    store.set("sessions", "s1", { observationCount: 1 });
    store.set("sessions", "s1", { observationCount: 2 });
    store.update("sessions", "s1", [
      { type: "set", path: "status", value: "completed" },
    ]);
    store.delete("sessions", "s1");
    store.delete("sessions", "gone");

    expect(events.map((e) => e.event_type)).toEqual([
      "state:created",
      "state:updated",
      "state:updated",
      "state:deleted",
    ]);
    expect(events[0]).toMatchObject({
      type: "state",
      scope: "sessions",
      key: "s1",
      old_value: null,
      new_value: { observationCount: 1 },
    });
    expect(events[1].old_value).toEqual({ observationCount: 1 });
    expect(events[3]).toMatchObject({
      event_type: "state:deleted",
      new_value: null,
    });
  });

  it("only emits for scopes a trigger watches", () => {
    const events: StateEvent[] = [];
    store.onEvent((e) => events.push(e));
    store.watchScope("sessions");
    store.set("memories", "m1", { a: 1 });
    store.set("sessions", "s1", { a: 1 });
    expect(events.map((e) => e.scope)).toEqual(["sessions"]);
  });

  it("holds events until the outermost commit and drops them on rollback", () => {
    const events: StateEvent[] = [];
    store.onEvent((e) => events.push(e));
    store.watchScope("sessions");

    store.transaction(() => {
      store.set("sessions", "s1", { a: 1 });
      // Still inside the transaction: nothing may have escaped yet.
      expect(events).toEqual([]);
    });
    expect(events).toHaveLength(1);

    expect(() =>
      store.transaction(() => {
        store.set("sessions", "s2", { a: 1 });
        throw new Error("abort");
      }),
    ).toThrow("abort");
    // The row rolled back, so no subscriber may have seen it.
    expect(events).toHaveLength(1);
    expect(store.get("sessions", "s2")).toBeNull();
  });

  it("event payloads and return values do not alias the caller's object", () => {
    const events: StateEvent[] = [];
    store.onEvent((e) => events.push(e));
    store.watchScope("sessions");

    const value = { observationCount: 1 };
    const setResult = store.set("sessions", "s1", value);
    value.observationCount = 99;

    expect(events[0].new_value).toEqual({ observationCount: 1 });
    expect(setResult.new_value).toEqual({ observationCount: 1 });
    expect(store.get("sessions", "s1")).toEqual({ observationCount: 1 });

    const nested = { deep: 1 };
    const updateResult = store.update("sessions", "s1", [
      { type: "set", path: "meta", value: nested },
    ]);
    nested.deep = 99;
    expect(updateResult.new_value).toMatchObject({ meta: { deep: 1 } });
    expect(store.get("sessions", "s1")).toMatchObject({ meta: { deep: 1 } });
  });

  it("event payloads do not alias the returned value either", () => {
    const events: StateEvent[] = [];
    store.onEvent((e) => events.push(e));
    store.watchScope("sessions");

    store.transaction(() => {
      const result = store.set("sessions", "s1", { n: 1 });
      // Events are buffered until commit; mutating the return here must not
      // rewrite what the subscriber is about to be told.
      (result.new_value as { n: number }).n = 99;
    });

    expect(events[0].new_value).toEqual({ n: 1 });
    expect(store.get("sessions", "s1")).toEqual({ n: 1 });

    // And the reverse: a listener mutating its payload cannot reach a caller.
    events[0].new_value = { n: -1 };
    expect(store.get("sessions", "s1")).toEqual({ n: 1 });
  });

  it("stores a __proto__ key as an ordinary property, as iii does", () => {
    store.update("s", "k", [
      { type: "set", path: "__proto__", value: { x: 1 } },
      { type: "set", path: "ok", value: 1 },
    ]);
    const stored = store.get("s", "k") as Record<string, unknown>;
    expect(Object.hasOwn(stored, "__proto__")).toBe(true);
    expect(stored["__proto__"]).toEqual({ x: 1 });
    expect(stored["ok"]).toBe(1);
  });

  it("persists across a reopen with WAL enabled", () => {
    const path = join(dir, "reopen.sqlite");
    const first = new SqliteState(path);
    first.set("s", "a", { keep: true });
    expect(
      (first.db.prepare("PRAGMA journal_mode").get() as { journal_mode: string })
        .journal_mode,
    ).toBe("wal");
    first.close();

    const second = new SqliteState(path);
    expect(second.get("s", "a")).toEqual({ keep: true });
    second.close();
  });
});

describe("inproc state store: batched writes and event-loop cooperation", () => {
  let dir: string;
  let store: SqliteState;

  beforeEach(() => {
    dir = mkdtempSync(join(tmpdir(), "am-inproc-state-many-"));
    store = new SqliteState(join(dir, "state.sqlite"));
  });

  afterEach(() => {
    store.close();
    rmSync(dir, { recursive: true, force: true });
  });

  it("state::set-many upserts the batch in one transaction and emits per-row events after commit", async () => {
    const fns = stateFunctions(store);
    const events: StateEvent[] = [];
    store.watchScope("s");
    store.onEvent((e) => events.push(e));
    await fns["state::set"]({ scope: "s", key: "a", value: 1 });
    events.length = 0;

    const n = await fns["state::set-many"]({
      scope: "s",
      entries: [
        { key: "a", value: 2 },
        { key: "b", value: 3 },
      ],
    });

    expect(n).toBe(2);
    expect(await fns["state::list"]({ scope: "s" })).toEqual([2, 3]);
    expect(events.map((e) => [e.event_type, e.key, e.old_value, e.new_value])).toEqual([
      ["state:updated", "a", 1, 2],
      ["state:created", "b", null, 3],
    ]);
    expect(store.db.isTransaction).toBe(false);
  });

  it("state::set-many rolls the whole batch back when one row cannot be encoded", async () => {
    const fns = stateFunctions(store);
    await expect(
      fns["state::set-many"]({
        scope: "s",
        entries: [
          { key: "x", value: 1 },
          { key: "y", value: 10n },
        ],
      }),
    ).rejects.toThrow();
    expect(await fns["state::get"]({ scope: "s", key: "x" })).toBeNull();
    expect(store.db.isTransaction).toBe(false);
  });

  it("a long run of awaited state calls lets a macrotask run before it finishes", async () => {
    const fns = stateFunctions(store);
    await fns["state::set"]({ scope: "s", key: "k", value: { n: 0 } });
    let i = 0;
    let firedAt = -1;
    setImmediate(() => {
      firedAt = i;
    });
    const rounds = 20_000;
    for (i = 0; i < rounds; i++) {
      await fns["state::get"]({ scope: "s", key: "k" });
    }
    expect(firedAt).toBeGreaterThanOrEqual(0);
    expect(firedAt).toBeLessThan(rounds);
  });
});
