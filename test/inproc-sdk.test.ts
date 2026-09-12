import { describe, it, expect, beforeEach, afterEach } from "vitest";
import { mkdtempSync, rmSync } from "node:fs";
import { tmpdir } from "node:os";
import { join } from "node:path";
import WebSocket from "ws";

import { createInprocSdk, type InprocSdk } from "../src/engine/inproc/sdk.js";

// The contract asserted here is iii 0.11.2's HTTP, state and stream
// behaviour, read from engine/src/workers/{rest_api,state,stream}. The daemon
// is written against that behaviour; "nicer" here is a production bug.
describe("inproc sdk shim", () => {
  let dir: string;
  let sdk: InprocSdk;
  let base: string;

  const SECRET = "s3cret";

  beforeEach(async () => {
    dir = mkdtempSync(join(tmpdir(), "am-inproc-sdk-"));
    sdk = createInprocSdk({
      restPort: 0,
      streamsPort: 0,
      sqlitePath: join(dir, "state.sqlite"),
    });
    await sdk.listening();
    base = `http://127.0.0.1:${sdk.ports().rest}`;
    sdk.setReady();

    // The daemon's real middleware, verbatim from src/triggers/api.ts.
    sdk.registerFunction("middleware::api-auth", async (input: any) => {
      const headers = input?.request?.headers || {};
      const auth = headers["authorization"] || headers["Authorization"];
      if (auth !== `Bearer ${SECRET}`) {
        return {
          action: "respond",
          response: { status_code: 401, body: { error: "unauthorized" } },
        };
      }
      return { action: "continue" };
    });
  });

  afterEach(async () => {
    await sdk.shutdown();
    rmSync(dir, { recursive: true, force: true });
  });

  function route(
    functionId: string,
    handler: (req: any) => Promise<unknown>,
    config: Record<string, unknown>,
  ) {
    sdk.registerFunction(functionId, handler as never);
    sdk.registerTrigger({ type: "http", function_id: functionId, config } as never);
  }

  it("answers 503 on readyz and every route but livez until setReady()", async () => {
    await sdk.shutdown();
    sdk = createInprocSdk({ restPort: 0, streamsPort: 0, sqlitePath: join(dir, "closed.sqlite") });
    await sdk.listening();
    base = `http://127.0.0.1:${sdk.ports().rest}`;
    route("api::liveness", async () => ({ status_code: 200, body: { status: "ok" } }), {
      api_path: "/agentmemory/livez",
      http_method: "GET",
    });
    route("api::x", async () => ({ status_code: 200, body: { ok: true } }), {
      api_path: "/agentmemory/x",
      http_method: "GET",
    });

    let res = await fetch(`${base}/agentmemory/readyz`);
    expect(res.status).toBe(503);
    expect(await res.json()).toEqual({ status: "starting" });
    res = await fetch(`${base}/agentmemory/x`);
    expect(res.status).toBe(503);
    expect(await res.json()).toEqual({ error: "starting" });
    res = await fetch(`${base}/agentmemory/nope`);
    expect(res.status).toBe(503);
    res = await fetch(`${base}/agentmemory/livez`);
    expect(res.status).toBe(200);

    sdk.setReady();
    res = await fetch(`${base}/agentmemory/readyz`);
    expect(res.status).toBe(200);
    expect(await res.json()).toEqual({ status: "ready" });
    res = await fetch(`${base}/agentmemory/x`);
    expect(res.status).toBe(200);
    res = await fetch(`${base}/agentmemory/nope`);
    expect(res.status).toBe(404);
  });

  it("maps a request into iii's ApiRequest shape", async () => {
    let seen: any;
    route(
      "api::echo",
      async (req) => {
        seen = req;
        return { status_code: 200, body: { ok: true } };
      },
      { api_path: "/agentmemory/echo", http_method: "POST" },
    );

    const res = await fetch(`${base}/agentmemory/echo?limit=5`, {
      method: "POST",
      headers: { "content-type": "application/json", "x-probe": "yes" },
      body: JSON.stringify({ hello: "world" }),
    });

    expect(res.status).toBe(200);
    expect(res.headers.get("content-type")).toBe("application/json");
    expect(await res.json()).toEqual({ ok: true });

    expect(seen.method).toBe("POST");
    expect(seen.path).toBe("/agentmemory/echo");
    expect(seen.path_params).toEqual({});
    expect(seen.query_params).toEqual({ limit: "5" });
    expect(seen.body).toEqual({ hello: "world" });
    expect(seen.headers["x-probe"]).toBe("yes");
    expect(seen.trigger).toEqual({
      type: "http",
      path: "/agentmemory/echo",
      method: "POST",
    });
  });

  it("collapses a repeated query key to a scalar, as the engine does", async () => {
    let seen: any;
    route(
      "api::q",
      async (req) => {
        seen = req;
        return { status_code: 200, body: {} };
      },
      { api_path: "/agentmemory/q", http_method: "GET" },
    );

    await fetch(`${base}/agentmemory/q?tag=a&tag=b&one=1`);
    // HashMap<String, String>: last value wins, never an array.
    expect(seen.query_params).toEqual({ tag: "b", one: "1" });
    expect(Array.isArray(seen.query_params.tag)).toBe(false);
  });

  it("extracts path parameters and prefers a static route over a param route", async () => {
    route(
      "api::by-id",
      async (req) => ({ status_code: 200, body: { id: req.path_params.id } }),
      { api_path: "/agentmemory/memories/:id", http_method: "GET" },
    );
    route("api::fixed", async () => ({ status_code: 200, body: { fixed: true } }), {
      api_path: "/agentmemory/memories/fixed",
      http_method: "GET",
    });

    expect(await (await fetch(`${base}/agentmemory/memories/mem%2F1`)).json()).toEqual({
      id: "mem/1",
    });
    // Same segment count as the :id route; the static one must win.
    expect(await (await fetch(`${base}/agentmemory/memories/fixed`)).json()).toEqual({
      fixed: true,
    });
    expect((await fetch(`${base}/agentmemory/memories/a/b`)).status).toBe(404);
  });

  it("answers 400 for an unparseable request target and survives a malformed upgrade", async () => {
    const { request } = await import("node:http");
    const rawStatus = (port: number, target: string, headers: Record<string, string>) =>
      new Promise<number | "closed">((resolve) => {
        const req = request({ host: "127.0.0.1", port, method: "GET", path: target, headers }, (res) => {
          resolve(res.statusCode ?? 0);
          res.resume();
        });
        req.on("error", () => resolve("closed"));
        req.on("upgrade", () => resolve(101));
        req.end();
      });

    expect(await rawStatus(sdk.ports().rest, "//[/", {})).toBe(400);
    const upgrade = {
      connection: "Upgrade",
      upgrade: "websocket",
      "sec-websocket-version": "13",
      "sec-websocket-key": "dGhlIHNhbXBsZSBub25jZQ==",
    };
    expect(await rawStatus(sdk.ports().streams, "//[/", upgrade)).toBe("closed");
    // Still alive afterwards.
    expect((await fetch(`${base}/agentmemory/livez-probe`)).status).toBe(404);
  });

  it("keeps the LAST of a repeated header, as axum's HashMap collect does", async () => {
    route("api::whoami", async () => ({ status_code: 200, body: { ok: true } }), {
      api_path: "/agentmemory/whoami",
      http_method: "GET",
      middleware_function_ids: ["middleware::api-auth"],
    });
    const { request } = await import("node:http");
    const statusWith = (headers: string[]) =>
      new Promise<number>((resolve) => {
        const req = request(
          { host: "127.0.0.1", port: sdk.ports().rest, method: "GET", path: "/agentmemory/whoami" },
          (res) => {
            resolve(res.statusCode ?? 0);
            res.resume();
          },
        );
        // setHeader with an array writes one line per value, in order.
        req.setHeader("Authorization", headers);
        req.end();
      });
    expect(await statusWith([`Bearer ${SECRET}`, "Bearer wrong"])).toBe(401);
    expect(await statusWith(["Bearer wrong", `Bearer ${SECRET}`])).toBe(200);
  });

  it("runs middleware before the handler and honours respond / continue", async () => {
    let handlerRuns = 0;
    route(
      "api::guarded",
      async () => {
        handlerRuns++;
        return { status_code: 200, body: { ok: true } };
      },
      {
        api_path: "/agentmemory/guarded",
        http_method: "GET",
        middleware_function_ids: ["middleware::api-auth"],
      },
    );

    const missing = await fetch(`${base}/agentmemory/guarded`);
    expect(missing.status).toBe(401);
    expect(await missing.json()).toEqual({ error: "unauthorized" });

    const wrong = await fetch(`${base}/agentmemory/guarded`, {
      headers: { authorization: "Bearer nope" },
    });
    expect(wrong.status).toBe(401);
    expect(handlerRuns).toBe(0);

    const ok = await fetch(`${base}/agentmemory/guarded`, {
      headers: { authorization: `Bearer ${SECRET}` },
    });
    expect(ok.status).toBe(200);
    expect(handlerRuns).toBe(1);
  });

  it("drops handler-returned headers and always answers JSON, as iii does", async () => {
    route(
      "api::html",
      async () => ({
        status_code: 200,
        headers: { "Content-Type": "text/html" },
        body: "<h1>hi</h1>",
      }),
      { api_path: "/agentmemory/html", http_method: "GET" },
    );

    const res = await fetch(`${base}/agentmemory/html`);
    expect(res.headers.get("content-type")).toBe("application/json");
    expect(await res.text()).toBe('"<h1>hi</h1>"');
  });

  it("parses a body only for application/json", async () => {
    let seen: any;
    route(
      "api::body",
      async (req) => {
        seen = req;
        return { status_code: 200, body: {} };
      },
      { api_path: "/agentmemory/body", http_method: "POST" },
    );

    await fetch(`${base}/agentmemory/body`, {
      method: "POST",
      headers: { "content-type": "text/plain" },
      body: '{"a":1}',
    });
    expect(seen.body).toBeNull();

    await fetch(`${base}/agentmemory/body`, {
      method: "POST",
      headers: { "content-type": "application/json" },
      body: "not json",
    });
    expect(seen.body).toBeNull();
  });

  it("answers 404 for an unknown route and 500 for a throwing handler", async () => {
    route(
      "api::boom",
      async () => {
        throw new Error("kaboom");
      },
      { api_path: "/agentmemory/boom", http_method: "GET" },
    );

    expect((await fetch(`${base}/agentmemory/nope`)).status).toBe(404);

    const res = await fetch(`${base}/agentmemory/boom`);
    expect(res.status).toBe(500);
    const body = (await res.json()) as { error: string; error_id: string };
    expect(body.error).toBe("kaboom");
    expect(body.error_id).toBeTruthy();
  });

  it("rejects an unknown function id with iii's error code", async () => {
    await expect(
      sdk.trigger({ function_id: "mem::nope", payload: {} }),
    ).rejects.toMatchObject({
      code: "function_not_found",
      message: "Function not found",
      function_id: "mem::nope",
    });
  });

  it("returns immediately for a void trigger and still runs the handler", async () => {
    let ran = 0;
    let finished = false;
    let release: () => void = () => {};
    const gate = new Promise<void>((r) => (release = r));
    sdk.registerFunction("mem::slow", (async () => {
      ran++;
      await gate;
      finished = true;
      return { done: true };
    }) as never);

    const out = await sdk.trigger({
      function_id: "mem::slow",
      payload: {},
      action: { type: "void" },
    });
    // Resolved while the handler is still parked on its gate.
    expect(out).toBeUndefined();
    expect(ran).toBe(1);
    expect(finished).toBe(false);
    release();
    await gate;
    await new Promise((r) => setImmediate(r));
    expect(finished).toBe(true);
  });

  it("consumes the payload synchronously, so a later mutation by the caller is not written", async () => {
    const value: Record<string, unknown> = { n: 1 };
    const pending = sdk.trigger({
      function_id: "state::set",
      payload: { scope: "mem:sessions", key: "s", value },
    });
    value.n = 2;
    await pending;
    expect(sdk.store.get("mem:sessions", "s")).toEqual({ n: 1 });
  });

  it("swallows a failing void trigger instead of rejecting", async () => {
    sdk.registerFunction("mem::void-boom", (async () => {
      throw new Error("nope");
    }) as never);
    await expect(
      sdk.trigger({
        function_id: "mem::void-boom",
        payload: {},
        action: { type: "void" },
      }),
    ).resolves.toBeUndefined();
  });

  it("fires a state trigger for its scope with iii's payload", async () => {
    const seen: any[] = [];
    sdk.registerFunction("event::session::changed", (async (p: any) => {
      seen.push(p);
      return { ok: true };
    }) as never);
    sdk.registerTrigger({
      type: "state",
      function_id: "event::session::changed",
      config: { scope: "mem:sessions" },
    } as never);

    await sdk.trigger({
      function_id: "state::set",
      payload: { scope: "mem:sessions", key: "s1", value: { observationCount: 1 } },
    });
    await sdk.trigger({
      function_id: "state::set",
      payload: { scope: "mem:memories", key: "m1", value: { title: "x" } },
    });
    // The fan-out is fire-and-forget, exactly as the engine spawns it.
    await new Promise((r) => setTimeout(r, 20));

    expect(seen).toHaveLength(1);
    expect(seen[0]).toMatchObject({
      type: "state",
      event_type: "state:created",
      scope: "mem:sessions",
      key: "s1",
      old_value: null,
      new_value: { observationCount: 1 },
    });
  });

  it("records a durable:subscriber trigger without inventing a topic bus", async () => {
    sdk.registerFunction("event::observation", (async () => ({ ok: true })) as never);
    sdk.registerTrigger({
      type: "durable:subscriber",
      function_id: "event::observation",
      config: { topic: "agentmemory.observation" },
    } as never);

    const triggers = await sdk.listTriggers();
    expect(
      triggers.find((t) => t.trigger_type === "durable:subscriber"),
    ).toMatchObject({
      function_id: "event::observation",
      config: { topic: "agentmemory.observation" },
    });
    // Nothing publishes to topics; the function is still directly invokable,
    // which is how the daemon actually calls it.
    await expect(
      sdk.trigger({ function_id: "event::observation", payload: {} }),
    ).resolves.toEqual({ ok: true });
  });

  it("delivers stream::set and stream::send to a direct-path subscriber", async () => {
    const url = `ws://127.0.0.1:${sdk.ports().streams}/stream/mem-live/viewer`;
    const ws = new WebSocket(url);
    const messages: any[] = [];
    ws.on("message", (raw) => messages.push(JSON.parse(raw.toString())));
    await new Promise((r) => ws.once("open", r));

    await sdk.trigger({
      function_id: "stream::set",
      payload: {
        stream_name: "mem-live",
        group_id: "viewer",
        item_id: "obs_1",
        data: { type: "compressed", observation: { id: "obs_1" } },
      },
    });
    await sdk.trigger({
      function_id: "stream::send",
      payload: {
        stream_name: "mem-live",
        group_id: "viewer",
        id: "raw-obs_1",
        type: "raw_observation",
        data: { sessionId: "s1" },
      },
    });
    await new Promise((r) => setTimeout(r, 50));
    ws.close();

    expect(messages).toHaveLength(2);
    expect(messages[0]).toMatchObject({
      type: "stream",
      streamName: "mem-live",
      groupId: "viewer",
      id: "obs_1",
      event: { type: "create", data: { type: "compressed" } },
    });
    expect(messages[1]).toMatchObject({
      type: "stream",
      id: "raw-obs_1",
      event: { type: "event", event: { type: "raw_observation" } },
    });
  });

  it("subscribes on a join message and reports update for a replaced item", async () => {
    const ws = new WebSocket(`ws://127.0.0.1:${sdk.ports().streams}`);
    const messages: any[] = [];
    ws.on("message", (raw) => messages.push(JSON.parse(raw.toString())));
    await new Promise((r) => ws.once("open", r));
    ws.send(
      JSON.stringify({
        type: "join",
        data: { subscriptionId: "viewer-1", streamName: "mem-live", groupId: "viewer" },
      }),
    );
    await new Promise((r) => setTimeout(r, 20));

    const set = (data: unknown) =>
      sdk.trigger({
        function_id: "stream::set",
        payload: {
          stream_name: "mem-live",
          group_id: "viewer",
          item_id: "obs_1",
          data,
        },
      });
    await set({ v: 1 });
    await set({ v: 2 });
    // A different group must not reach this subscriber.
    await sdk.trigger({
      function_id: "stream::set",
      payload: {
        stream_name: "mem-live",
        group_id: "other",
        item_id: "obs_2",
        data: { v: 3 },
      },
    });
    await new Promise((r) => setTimeout(r, 50));
    ws.close();

    expect(messages.map((m) => m.event.type)).toEqual(["create", "update"]);
  });

  it("shuts down with a connected client that never joined", async () => {
    const ws = new WebSocket(`ws://127.0.0.1:${sdk.ports().streams}/`);
    await new Promise((r) => ws.once("open", r));
    const closed = new Promise((r) => ws.once("close", r));
    const done = sdk.shutdown();
    await expect(Promise.race([done, new Promise((_, rej) => setTimeout(() => rej(new Error("hung")), 3000))])).resolves.toBeUndefined();
    await closed;
    // afterEach shuts down again; make that a no-op on the closed store.
    sdk = createInprocSdk({ restPort: 0, streamsPort: 0, sqlitePath: join(dir, "again.sqlite") });
    await sdk.listening();
  });

  it("disconnects a subscriber whose socket buffer exceeds the cap", async () => {
    await sdk.shutdown();
    sdk = createInprocSdk({
      restPort: 0,
      streamsPort: 0,
      sqlitePath: join(dir, "slow.sqlite"),
      maxSocketBufferBytes: 1024,
    });
    await sdk.listening();
    const ws = new WebSocket(`ws://127.0.0.1:${sdk.ports().streams}/stream/mem-live/viewer`);
    await new Promise((r) => ws.once("open", r));
    // Stop reading so frames queue up server-side.
    ws.pause();
    const closed = new Promise((r) => ws.once("close", r));
    // The kernel absorbs the first megabytes into its own buffers before
    // ws.bufferedAmount starts growing, so keep publishing until the server
    // cuts the socket or we have sent far more than any loopback window.
    const big = "x".repeat(1024 * 1024);
    let done = false;
    closed.then(() => (done = true));
    for (let i = 0; i < 256 && !done; i++) {
      await sdk.trigger({
        function_id: "stream::send",
        payload: { stream_name: "mem-live", group_id: "viewer", type: "t", data: big },
      });
      await new Promise((r) => setImmediate(r));
    }
    // A paused socket cannot observe the server's RST; resume so the client
    // drains what the kernel buffered and then sees the connection gone. Had
    // the server kept sending, the client would read every frame and stay open.
    ws.resume();
    await expect(Promise.race([closed, new Promise((_, rej) => setTimeout(() => rej(new Error("still connected")), 5000))])).resolves.toBeDefined();
  });

  it("reports itself as the single worker for the health monitor", async () => {
    const result = (await sdk.trigger({
      function_id: "engine::workers::list",
      payload: {},
    })) as { workers: Array<{ status: string }> };
    expect(result.workers).toHaveLength(1);
    expect(result.workers[0].status).toBe("connected");
  });

  it("exposes no getMeter or connection-state emitter", () => {
    // src/index.ts's hasGetMeter() and src/health/monitor.ts's
    // `typeof sdk.on === "function"` both fall back correctly when absent.
    expect((sdk as any).getMeter).toBeUndefined();
    expect((sdk as any).on).toBeUndefined();
  });
});
