// In-process replacement for the iii engine.
//
// `createInprocSdk()` returns an `ISdk` the daemon can be registered against
// exactly as it registers against `registerWorker()`, but every call stays in
// this process: functions live in a Map, HTTP triggers are served by a
// `node:http` server, state functions run on `node:sqlite` (state.ts), and the
// stream feed is a `ws` server. No engine, no WebSocket worker bus, no
// 180 s invocation ceiling, no state_store.db.
//
// Behaviour is copied from iii 0.11.2, read from the engine source rather than
// the SDK's `.d.mts`, because the two disagree in places the daemon can see:
//
//   - `engine/src/workers/rest_api/types.rs`: `query_params`, `path_params`
//     and `headers` are `HashMap<String, String>` — SCALAR. A repeated query
//     key collapses to one value; the SDK's `string | string[]` is wider than
//     the engine ever produces.
//   - `engine/src/workers/rest_api/views.rs`: the body is parsed only for
//     `application/json` (invalid JSON and every other content type arrive as
//     `null`); the handler's return supplies `status_code` and `body` ONLY —
//     headers it returns are dropped and the response is always JSON;
//     middleware sees `{phase, request:{path_params, query_params, headers,
//     method}, context:{}}` and answers `{action:"continue"}` or
//     `{action:"respond", response:{status_code, headers, body}}`.
//   - `engine/src/workers/stream/{stream,structs}.rs`: subscribers receive
//     `{type:"stream", timestamp, streamName, groupId, id, event}` where
//     `event` is `{type:"create"|"update", data}` for `stream::set` and
//     `{type:"event", event:{type, data}}` for `stream::send`.
//   - `iii-sdk/dist/index.mjs`: a failed invocation rejects with an object
//     carrying `code` (`function_not_found` / `invocation_failed`) and
//     `message`, which `src/index.ts`'s unhandledRejection handler reads.
import { createServer, type IncomingMessage, type Server, type ServerResponse } from "node:http";
import { randomUUID } from "node:crypto";
import type { ISdk } from "iii-sdk";
import { WebSocketServer, type WebSocket } from "ws";

import { SqliteState, stateFunctions, type StateEvent } from "./state.js";
import { logger } from "../../logger.js";

type Handler = (payload: any) => Promise<unknown>;

// Rejections carry the engine's error code and the function id alongside a
// real Error, so `err instanceof Error` still yields a usable message in the
// daemon's log lines while `err.code` / `err.function_id` stay exactly what
// `src/index.ts:170`'s unhandledRejection handler reads.
export class InprocInvocationError extends Error {
  constructor(
    message: string,
    readonly code: string,
    readonly function_id: string,
    readonly stacktrace?: string,
  ) {
    super(message);
    this.name = "InprocInvocationError";
  }
}

type HttpRoute = {
  functionId: string;
  middleware: string[];
  // Route pattern split on "/", with ":name" marking a parameter.
  segments: string[];
  params: boolean;
  path: string;
};

type Subscription = { ws: WebSocket; streamName: string; groupId: string };

// One bounded ring for the whole stream store. iii's kv-backed stream adapter
// grows a group per session and never evicts; this caps both the item count
// and the bytes so a long-lived daemon cannot accumulate stream state.
const STREAM_MAX_ITEMS = 2000;
const STREAM_MAX_BYTES = 32 * 1024 * 1024;
// A subscriber that stops reading accumulates frames in its socket buffer,
// outside the ring above. Past this many queued bytes it is disconnected.
const DEFAULT_MAX_SOCKET_BUFFER_BYTES = 8 * 1024 * 1024;

// Requests larger than this are refused. iii buffered the whole body and then
// tried to hand it to the worker over a WebSocket frame the engine caps at
// 16 MiB (see src/state/frame-guard.ts), so an oversized body killed the
// worker connection. One clean 413 is strictly better and costs nothing.
const DEFAULT_MAX_BODY_BYTES = 16 * 1024 * 1024;

export type InprocSdkOptions = {
  restPort: number;
  streamsPort: number;
  sqlitePath: string;
  host?: string;
  maxBodyBytes?: number;
  maxSocketBufferBytes?: number;
};

export type InprocSdk = ISdk & {
  /** The SQLite store behind the `state::*` functions. */
  readonly store: SqliteState;
  /** Resolves once both listeners are bound (or rejects if either cannot be). */
  listening(): Promise<void>;
  /**
   * Opens the gate: until called, `/agentmemory/readyz` answers 503 and every
   * other route except `/agentmemory/livez` answers 503 too (plan step 6).
   */
  setReady(): void;
  /** Actually-bound ports. Meaningful after `listening()` resolves. */
  ports(): { rest: number; streams: number };
};

// axum's HeaderMap -> HashMap<String,String> collect keeps the LAST value of a
// repeated header. Node's parsed `req.headers` cannot reproduce that: it keeps
// the FIRST `authorization` (and a few others) and comma-joins the rest, so a
// request carrying a good bearer followed by a bad one would authenticate here
// and be refused by iii. Walk the wire-order pairs instead.
function scalarHeaders(req: IncomingMessage): Record<string, string> {
  const out: Record<string, string> = {};
  const raw = req.rawHeaders;
  for (let i = 0; i + 1 < raw.length; i += 2) out[raw[i].toLowerCase()] = raw[i + 1];
  return out;
}

// `new URL()` throws on request targets like `//[/`; a throw here must become
// a 400 (or a dropped upgrade), never an uncaught exception.
function parseRequestUrl(req: IncomingMessage): URL | null {
  try {
    return new URL(req.url ?? "/", "http://localhost");
  } catch {
    return null;
  }
}

function compileRoute(path: string): Pick<HttpRoute, "segments" | "params"> {
  const segments = path.split("/");
  return { segments, params: segments.some((s) => s.startsWith(":")) };
}

function matchRoute(
  route: HttpRoute,
  parts: string[],
): Record<string, string> | null {
  if (route.segments.length !== parts.length) return null;
  const out: Record<string, string> = {};
  for (let i = 0; i < parts.length; i++) {
    const seg = route.segments[i];
    if (seg.startsWith(":")) {
      let value = parts[i];
      try {
        value = decodeURIComponent(value);
      } catch {
        // Leave a malformed escape sequence as-is rather than 400ing; the
        // handler validates the id anyway.
      }
      out[seg.slice(1)] = value;
    } else if (seg !== parts[i]) {
      return null;
    }
  }
  return out;
}

export function createInprocSdk(opts: InprocSdkOptions): InprocSdk {
  const host = opts.host ?? "127.0.0.1";
  const maxBodyBytes = opts.maxBodyBytes ?? DEFAULT_MAX_BODY_BYTES;
  const maxSocketBufferBytes = opts.maxSocketBufferBytes ?? DEFAULT_MAX_SOCKET_BUFFER_BYTES;

  const functions = new Map<string, Handler>();
  // Static routes are keyed "METHOD /path"; only parameterised ones need a
  // scan, and the daemon has exactly one (`/agentmemory/memories/:id`).
  const staticRoutes = new Map<string, HttpRoute>();
  const paramRoutes: Array<{ method: string; route: HttpRoute }> = [];
  // Recorded so `listTriggers()` is honest. Nothing in the daemon publishes to
  // a topic — the four `event::*` functions are invoked by function id — so
  // there is no emitter to feed them.
  const subscriberTriggers: Array<{ topic: string; functionId: string }> = [];

  const store = new SqliteState(opts.sqlitePath);
  for (const [id, handler] of Object.entries(stateFunctions(store))) {
    functions.set(id, handler);
  }

  // Readiness: index.ts flips this after the DB write probe, vector hydration
  // and the BM25 rebuild. Routes register after the port binds, so without the
  // gate an early client would get 404s and empty search results.
  let ready = false;
  const READYZ = "/agentmemory/readyz";
  const LIVEZ = "/agentmemory/livez";

  // ---------------------------------------------------------------- invoke

  function invoke(functionId: string, payload: unknown): Promise<unknown> {
    const handler = functions.get(functionId);
    if (!handler) {
      return Promise.reject(
        new InprocInvocationError(
          "Function not found",
          "function_not_found",
          functionId,
        ),
      );
    }
    // Call the handler NOW rather than on a later microtask. iii serialises
    // the payload synchronously inside trigger(), so a caller that mutates
    // its object after the call never affects the write; the state handlers
    // here are synchronous up to their return, so calling them eagerly gives
    // the same guarantee without cloning every payload.
    return new Promise<unknown>((resolve, reject) => {
      try {
        resolve(handler(payload));
      } catch (err) {
        reject(err);
      }
    })
      .catch((err) => {
        const isError = err instanceof Error;
        throw new InprocInvocationError(
          isError ? err.message : String(err),
          "invocation_failed",
          functionId,
          isError ? err.stack : undefined,
        );
      });
  }

  // ------------------------------------------------------- state triggers

  const stateTriggers: Array<{
    scope?: string;
    key?: string;
    functionId: string;
  }> = [];

  store.onEvent((event: StateEvent) => {
    for (const { scope, key, functionId } of stateTriggers) {
      if (scope !== undefined && scope !== event.scope) continue;
      if (key !== undefined && key !== event.key) continue;
      // iii spawns the trigger fan-out on its own task: the write has already
      // committed and its result is not held up by the handler.
      void invoke(functionId, event).catch((err) => {
        logger.warn("state trigger failed", {
          functionId,
          scope: event.scope,
          key: event.key,
          error: err instanceof Error ? err.message : String(err),
        });
      });
    }
  });

  // ------------------------------------------------------------- streams

  // group key -> item id -> data, plus a global insertion-ordered ring so the
  // eviction is by age across every group.
  const streamItems = new Map<string, Map<string, unknown>>();
  const streamOrder: Array<{ group: string; itemId: string; bytes: number }> = [];
  let streamBytes = 0;
  const subscriptions = new Set<Subscription>();

  const groupKey = (streamName: string, groupId: string) =>
    `${streamName} ${groupId}`;

  function broadcast(message: Record<string, unknown>): void {
    const streamName = message.streamName as string;
    const groupId = message.groupId as string;
    const text = JSON.stringify(message);
    for (const sub of subscriptions) {
      if (sub.streamName !== streamName || sub.groupId !== groupId) continue;
      if (sub.ws.bufferedAmount > maxSocketBufferBytes) {
        // Not reading: cut it loose rather than grow the heap on its behalf.
        // terminate() fires "close", whose handler drops its subscriptions.
        sub.ws.terminate();
        continue;
      }
      try {
        sub.ws.send(text);
      } catch {
        // A dead socket is reaped by its own close handler.
      }
    }
  }

  function evictStream(): void {
    while (
      streamOrder.length > STREAM_MAX_ITEMS ||
      streamBytes > STREAM_MAX_BYTES
    ) {
      const oldest = streamOrder.shift();
      if (!oldest) break;
      streamBytes -= oldest.bytes;
      const items = streamItems.get(oldest.group);
      if (!items) continue;
      items.delete(oldest.itemId);
      if (items.size === 0) streamItems.delete(oldest.group);
    }
  }

  functions.set("stream::set", async (p: any) => {
    const key = groupKey(p.stream_name, p.group_id);
    let items = streamItems.get(key);
    if (!items) {
      items = new Map();
      streamItems.set(key, items);
    }
    const existed = items.has(p.item_id);
    const old_value = existed ? items.get(p.item_id) : undefined;
    items.set(p.item_id, p.data);

    const bytes = Buffer.byteLength(JSON.stringify(p.data) ?? "", "utf8");
    const priorIndex = existed
      ? streamOrder.findIndex((e) => e.group === key && e.itemId === p.item_id)
      : -1;
    if (priorIndex >= 0) {
      streamBytes -= streamOrder[priorIndex].bytes;
      streamOrder.splice(priorIndex, 1);
    }
    streamOrder.push({ group: key, itemId: p.item_id, bytes });
    streamBytes += bytes;
    evictStream();

    broadcast({
      type: "stream",
      timestamp: Date.now(),
      streamName: p.stream_name,
      groupId: p.group_id,
      id: p.item_id,
      event: { type: existed ? "update" : "create", data: p.data },
    });
    return { old_value, new_value: p.data };
  });

  functions.set("stream::send", async (p: any) => {
    broadcast({
      type: "stream",
      timestamp: Date.now(),
      streamName: p.stream_name,
      groupId: p.group_id,
      id: p.id ?? null,
      // `stream::send`'s input names the event type `type`
      // (`#[serde(rename = "type")] event_type` on StreamSendInput).
      event: { type: "event", event: { type: p.type, data: p.data } },
    });
    return null;
  });

  // Health monitor's only engine call. One worker, this process.
  functions.set("engine::workers::list", async () => ({
    workers: [
      {
        id: "inproc",
        name: "agentmemory",
        runtime: "node",
        status: "connected",
        connected_at_ms: Date.now(),
        function_count: functions.size,
        functions: Array.from(functions.keys()),
        active_invocations: 0,
      },
    ],
  }));

  // ----------------------------------------------------------------- http

  async function readBody(req: IncomingMessage): Promise<Buffer | null> {
    const chunks: Buffer[] = [];
    let size = 0;
    for await (const chunk of req) {
      size += chunk.length;
      if (size > maxBodyBytes) return null;
      chunks.push(chunk as Buffer);
    }
    return Buffer.concat(chunks);
  }

  function respond(
    res: ServerResponse,
    status: number,
    body: unknown,
  ): void {
    if (body === undefined) {
      // iii's streaming fallback: the invocation returned nothing, so the
      // response carries a status and an empty body.
      res.writeHead(status);
      res.end();
      return;
    }
    const payload = JSON.stringify(body) ?? "null";
    res.writeHead(status, {
      "content-type": "application/json",
      "content-length": Buffer.byteLength(payload),
    });
    res.end(payload);
  }

  function findRoute(
    method: string,
    pathname: string,
  ): { route: HttpRoute; pathParams: Record<string, string> } | null {
    const exact = staticRoutes.get(`${method} ${pathname}`);
    if (exact) return { route: exact, pathParams: {} };
    const parts = pathname.split("/");
    for (const candidate of paramRoutes) {
      if (candidate.method !== method) continue;
      const pathParams = matchRoute(candidate.route, parts);
      if (pathParams) return { route: candidate.route, pathParams };
    }
    return null;
  }

  const httpServer = createServer((req, res) => {
    void handleRequest(req, res).catch((err) => {
      logger.error("inproc http handler crashed", {
        url: req.url,
        error: err instanceof Error ? err.message : String(err),
      });
      if (!res.headersSent) respond(res, 500, { error: "internal server error" });
      else res.end();
    });
  });

  async function handleRequest(
    req: IncomingMessage,
    res: ServerResponse,
  ): Promise<void> {
    const method = (req.method ?? "GET").toUpperCase();
    const url = parseRequestUrl(req);
    if (!url) {
      respond(res, 400, { error: "invalid request target" });
      return;
    }
    if (url.pathname === READYZ) {
      respond(res, ready ? 200 : 503, { status: ready ? "ready" : "starting" });
      return;
    }
    if (!ready && url.pathname !== LIVEZ) {
      respond(res, 503, { error: "starting" });
      return;
    }
    const match = findRoute(method, url.pathname);
    if (!match) {
      respond(res, 404, { error: "not found" });
      return;
    }

    const query_params: Record<string, string> = {};
    // serde_urlencoded into a HashMap keeps the last value for a repeated key.
    for (const [k, v] of url.searchParams) query_params[k] = v;
    const headers = scalarHeaders(req);

    for (const mwId of match.route.middleware) {
      let result: any;
      try {
        result = await invoke(mwId, {
          phase: "preHandler",
          request: {
            path_params: match.pathParams,
            query_params,
            headers,
            method,
          },
          context: {},
        });
      } catch (err) {
        logger.error("middleware failed", {
          middleware: mwId,
          error: err instanceof Error ? err.message : String(err),
        });
        respond(res, 500, {
          error: err instanceof Error ? err.message : String(err),
          error_id: randomUUID(),
        });
        return;
      }
      if (result?.action === "respond") {
        const response = result.response ?? {};
        const extra: Record<string, string> = {};
        // A middleware short-circuit DOES get its headers applied by iii,
        // unlike a handler return.
        for (const [k, v] of Object.entries(response.headers ?? {})) {
          if (typeof v === "string") extra[k] = v;
        }
        const payload = JSON.stringify(response.body ?? null) ?? "null";
        res.writeHead(response.status_code ?? 200, {
          ...extra,
          "content-type": "application/json",
          "content-length": Buffer.byteLength(payload),
        });
        res.end(payload);
        return;
      }
      // "continue", anything else, or no result at all: iii proceeds.
    }

    const raw = await readBody(req);
    if (raw === null) {
      respond(res, 413, { error: "request body too large" });
      return;
    }
    const contentType = headers["content-type"] ?? "";
    let body: unknown = null;
    if (contentType.includes("application/json")) {
      try {
        body = JSON.parse(raw.toString("utf8"));
      } catch {
        body = null;
      }
    }

    let result: any;
    try {
      result = await invoke(match.route.functionId, {
        query_params,
        path_params: match.pathParams,
        headers,
        path: match.route.path,
        method,
        body,
        trigger: { type: "http", path: match.route.path, method },
      });
    } catch (err) {
      const errorId = randomUUID();
      logger.error("Internal server error", {
        function_id: match.route.functionId,
        error_id: errorId,
        error: err instanceof Error ? err.message : String(err),
        stacktrace: err instanceof InprocInvocationError ? err.stacktrace : undefined,
      });
      respond(res, 500, {
        error: err instanceof Error ? err.message : String(err),
        error_id: errorId,
      });
      return;
    }

    if (result === undefined || result === null) {
      respond(res, 200, undefined);
      return;
    }
    // iii's HttpResponse::from_function_return: status_code (default 200) and
    // body (default {}). Headers the handler returns are DROPPED, and the body
    // always goes out as JSON.
    const status =
      typeof result.status_code === "number" ? result.status_code : 200;
    respond(res, status, result.body === undefined ? {} : result.body);
  }

  // ------------------------------------------------------------------- ws

  const wss = new WebSocketServer({ noServer: true });
  const streamServer = createServer((_req, res) => {
    res.writeHead(426);
    res.end();
  });

  streamServer.on("upgrade", (req, socket, head) => {
    const url = parseRequestUrl(req);
    if (!url) {
      socket.destroy();
      return;
    }
    wss.handleUpgrade(req, socket, head, (ws) => {
      const subs: Subscription[] = [];

      // Direct form: /stream/<streamName>/<groupId> subscribes on connect,
      // which is what the viewer tries first (WS_DIRECT_URL).
      const parts = url.pathname.split("/").filter(Boolean);
      if (parts.length === 3 && parts[0] === "stream") {
        const sub = { ws, streamName: parts[1], groupId: parts[2] };
        subscriptions.add(sub);
        subs.push(sub);
      }

      ws.on("message", (raw) => {
        let msg: any;
        try {
          msg = JSON.parse(raw.toString());
        } catch {
          return;
        }
        if (msg?.type === "join" && msg.data?.streamName) {
          const sub = {
            ws,
            streamName: String(msg.data.streamName),
            groupId: String(msg.data.groupId ?? ""),
          };
          subscriptions.add(sub);
          subs.push(sub);
          return;
        }
        if (msg?.type === "leave") {
          for (const sub of subs.splice(0)) subscriptions.delete(sub);
          return;
        }
        logger.info("inproc stream: ignoring unsupported message", {
          type: typeof msg?.type === "string" ? msg.type : "unknown",
        });
      });

      const drop = () => {
        for (const sub of subs) subscriptions.delete(sub);
      };
      ws.on("close", drop);
      ws.on("error", drop);
    });
  });

  // -------------------------------------------------------------- listen

  const bound = Promise.all([
    bind(httpServer, opts.restPort, host, "REST API"),
    bind(streamServer, opts.streamsPort, host, "stream feed"),
  ]).then(() => undefined);
  // A bind failure must surface through `listening()`, not as an unhandled
  // rejection that the daemon's suppressing handler swallows.
  bound.catch(() => {});

  const notSupported = (what: string) => () => {
    throw new Error(
      `${what} is not supported in AGENTMEMORY_ENGINE=inproc mode`,
    );
  };

  const boundPort = (server: Server, requested: number): number => {
    const addr = server.address();
    return addr && typeof addr === "object" ? addr.port : requested;
  };

  const sdk: InprocSdk = {
    store,
    listening: () => bound,
    ports: () => ({
      rest: boundPort(httpServer, opts.restPort),
      streams: boundPort(streamServer, opts.streamsPort),
    }),

    registerFunction(functionId: string, handler: any) {
      if (typeof handler !== "function") {
        throw new Error(
          `HTTP-invoked functions are not supported in inproc mode: ${functionId}`,
        );
      }
      functions.set(functionId, handler as Handler);
      return { id: functionId, unregister: () => void functions.delete(functionId) };
    },

    registerTrigger(trigger: any) {
      const { type, function_id, config } = trigger;
      if (type === "http") {
        const path = String(config?.api_path ?? "");
        const method = String(config?.http_method ?? "GET").toUpperCase();
        const route: HttpRoute = {
          functionId: function_id,
          middleware: Array.isArray(config?.middleware_function_ids)
            ? config.middleware_function_ids.map(String)
            : [],
          path,
          ...compileRoute(path),
        };
        if (route.params) paramRoutes.push({ method, route });
        else staticRoutes.set(`${method} ${path}`, route);
        return {
          unregister: () => {
            if (route.params) {
              const i = paramRoutes.findIndex((e) => e.route === route);
              if (i >= 0) paramRoutes.splice(i, 1);
            } else {
              staticRoutes.delete(`${method} ${path}`);
            }
          },
        };
      }
      if (type === "state") {
        const entry = {
          scope: config?.scope as string | undefined,
          key: config?.key as string | undefined,
          functionId: function_id,
        };
        stateTriggers.push(entry);
        store.watchScope(entry.scope);
        return {
          unregister: () => {
            const i = stateTriggers.indexOf(entry);
            if (i >= 0) stateTriggers.splice(i, 1);
          },
        };
      }
      if (type === "durable:subscriber") {
        const entry = { topic: String(config?.topic ?? ""), functionId: function_id };
        subscriberTriggers.push(entry);
        return {
          unregister: () => {
            const i = subscriberTriggers.indexOf(entry);
            if (i >= 0) subscriberTriggers.splice(i, 1);
          },
        };
      }
      logger.warn("inproc: ignoring unsupported trigger type", {
        type,
        function_id,
      });
      return { unregister: () => {} };
    },

    trigger<_TInput, TOutput>(request: any): Promise<TOutput> {
      if (request?.action?.type === "void") {
        void invoke(request.function_id, request.payload).catch((err) => {
          logger.warn("void trigger failed", {
            function_id: request.function_id,
            error: err instanceof Error ? err.message : String(err),
          });
        });
        return Promise.resolve(undefined as TOutput);
      }
      return invoke(request.function_id, request.payload) as Promise<TOutput>;
    },

    async listFunctions() {
      return Array.from(functions.keys()).map((function_id) => ({ function_id }));
    },

    async listTriggers() {
      const http = [
        ...Array.from(staticRoutes.entries()).map(([k, r]) => ({
          id: k,
          trigger_type: "http",
          function_id: r.functionId,
          config: { api_path: r.path },
        })),
        ...paramRoutes.map(({ method, route }) => ({
          id: `${method} ${route.path}`,
          trigger_type: "http",
          function_id: route.functionId,
          config: { api_path: route.path },
        })),
      ];
      return [
        ...http,
        ...stateTriggers.map((t) => ({
          id: `state ${t.scope ?? "*"}`,
          trigger_type: "state",
          function_id: t.functionId,
          config: { scope: t.scope },
        })),
        ...subscriberTriggers.map((t) => ({
          id: `durable:subscriber ${t.topic}`,
          trigger_type: "durable:subscriber",
          function_id: t.functionId,
          config: { topic: t.topic },
        })),
      ];
    },

    async listTriggerTypes() {
      return [
        { id: "http", description: "HTTP route" },
        { id: "state", description: "State change" },
        { id: "durable:subscriber", description: "Topic subscription" },
      ];
    },

    registerService: () => {},
    onFunctionsAvailable: () => () => {},
    registerTriggerType: notSupported("registerTriggerType") as never,
    unregisterTriggerType: notSupported("unregisterTriggerType") as never,
    createChannel: notSupported("createChannel") as never,
    createStream: notSupported("createStream") as never,

    setReady() {
      ready = true;
    },

    async shutdown() {
      // wss.close() only resolves once every tracked client is gone, and a
      // client that never joined (or left) is not in `subscriptions`.
      for (const client of wss.clients) client.terminate();
      subscriptions.clear();
      await Promise.all([
        new Promise<void>((resolve) => wss.close(() => resolve())),
        new Promise<void>((resolve) => streamServer.close(() => resolve())),
        new Promise<void>((resolve) => httpServer.close(() => resolve())),
      ]);
      store.close();
    },
  } as InprocSdk;

  return sdk;
}

function bind(
  server: Server,
  port: number,
  host: string,
  what: string,
): Promise<void> {
  return new Promise((resolve, reject) => {
    const onError = (err: Error) => {
      server.off("listening", onListening);
      reject(new Error(`inproc ${what} could not bind ${host}:${port}: ${err.message}`));
    };
    const onListening = () => {
      server.off("error", onError);
      resolve();
    };
    server.once("error", onError);
    server.once("listening", onListening);
    server.listen(port, host);
  });
}
