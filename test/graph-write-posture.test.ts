import { describe, it, expect, afterEach, vi } from "vitest";

// Regression test for the fork's graph-off posture against the 0.9.29 hazard.
//
// Stock 0.9.29 fires mem::graph-extract on every session stop and runs a
// keyless heuristic pass (a node per obs.files[] entry, a node per concept),
// so an install with graph extraction off still grows the graph. This fork
// must never write a graph row unless extraction is armed
// (GRAPH_EXTRACTION_ENABLED=true) AND the graph leg is not killed
// (AGENTMEMORY_GRAPH_LEG=off). Production runs flag=on + leg=off.
//
// Deliberately no mock of config.js or graph-indexes.js: both read
// process.env on every call, so the real gates are exercised.

vi.mock("../src/logger.js", () => ({
  logger: { info: vi.fn(), warn: vi.fn(), error: vi.fn() },
  bootLog: vi.fn(),
}));

import { registerEventTriggers } from "../src/triggers/events.js";
import { persistGraphDelta, registerGraphFunction } from "../src/functions/graph.js";
import { registerExportImportFunction } from "../src/functions/export-import.js";
import { KV } from "../src/state/schema.js";
import type { CompressedObservation } from "../src/types.js";

const SESSION = "ses_posture";

// The teeth: non-empty files + concepts make the heuristic pass produce
// nodes, and the provider returns a parseable entity, so any missing guard
// turns into a mem:graph:* write.
const OBS: CompressedObservation = {
  id: "obs_posture_1",
  sessionId: SESSION,
  timestamp: "2026-09-01T10:00:00Z",
  type: "file_edit",
  title: "Edit the graph writer",
  facts: ["touched graph.ts"],
  narrative: "Changed persistGraphDelta in src/functions/graph.ts",
  concepts: ["graph", "posture"],
  files: ["src/functions/graph.ts"],
  importance: 7,
};

// Spy KV: serves the seeded observation, records every set as [scope, key].
function spyKV() {
  const store = new Map<string, Map<string, unknown>>();
  const sets: Array<[string, string]> = [];
  store.set(KV.observations(SESSION), new Map([[OBS.id, OBS]]));
  return {
    sets,
    get: async <T>(scope: string, key: string): Promise<T | null> =>
      (store.get(scope)?.get(key) as T) ?? null,
    set: async <T>(scope: string, key: string, data: T): Promise<T> => {
      sets.push([scope, key]);
      if (!store.has(scope)) store.set(scope, new Map());
      store.get(scope)!.set(key, data);
      return data;
    },
    delete: async (scope: string, key: string): Promise<void> => {
      store.get(scope)?.delete(key);
    },
    update: async (): Promise<void> => {},
    list: async <T>(scope: string): Promise<T[]> =>
      Array.from(store.get(scope)?.values() ?? []) as T[],
  };
}

// Dispatching sdk: session::stopped really reaches mem::graph-extract.
function dispatchingSdk() {
  const fns = new Map<string, Function>();
  const dispatched: string[] = [];
  const trigger = async (
    idOrInput: string | { function_id: string; payload?: unknown },
    data?: unknown,
  ) => {
    const id = typeof idOrInput === "string" ? idOrInput : idOrInput.function_id;
    const payload = typeof idOrInput === "string" ? data : idOrInput.payload;
    dispatched.push(id);
    const fn = fns.get(id);
    if (!fn) throw new Error(`No function: ${id}`);
    return fn(payload);
  };
  return {
    dispatched,
    registerFunction: (id: string, handler: Function) => fns.set(id, handler),
    registerTrigger: () => {},
    trigger,
  };
}

const provider = {
  name: "test",
  compress: vi.fn().mockResolvedValue(
    `<entities><entity type="file" name="src/functions/graph.ts"/></entities>`,
  ),
};

function graphWrites(kv: ReturnType<typeof spyKV>): Array<[string, string]> {
  return kv.sets.filter(([scope]) => scope.startsWith("mem:graph"));
}

async function stopSession(kv: ReturnType<typeof spyKV>) {
  const sdk = dispatchingSdk();
  sdk.registerFunction("mem::summarize", async () => ({ summary: "s" }));
  registerGraphFunction(sdk as never, kv as never, provider as never);
  registerEventTriggers(sdk as never, kv as never);
  await sdk.trigger("event::session::stopped", { sessionId: SESSION });
  // fireVoid is fire-and-forget; let the dispatched handler settle.
  await new Promise((r) => setTimeout(r, 20));
  return sdk;
}

const ENV_KEYS = ["GRAPH_EXTRACTION_ENABLED", "AGENTMEMORY_GRAPH_LEG", "CONSOLIDATION_ENABLED", "AGENTMEMORY_REFLECT"];
const ORIG: Record<string, string | undefined> = {};
for (const k of ENV_KEYS) ORIG[k] = process.env[k];

function setEnv(values: Record<string, string | undefined>) {
  for (const k of ENV_KEYS) {
    const v = values[k];
    if (v === undefined) delete process.env[k];
    else process.env[k] = v;
  }
}

describe("graph write posture (graph-off fork)", () => {
  afterEach(() => {
    setEnv(ORIG);
    provider.compress.mockClear();
  });

  it("flag off, leg unset: session stop never dispatches graph-extract and writes no graph row", async () => {
    setEnv({ GRAPH_EXTRACTION_ENABLED: "false", CONSOLIDATION_ENABLED: "false" });
    const kv = spyKV();
    const sdk = await stopSession(kv);
    expect(sdk.dispatched).not.toContain("mem::graph-extract");
    expect(graphWrites(kv)).toEqual([]);
  });

  it("flag on, leg off (production): no graph-extract from session stop, and a direct call writes nothing", async () => {
    setEnv({ GRAPH_EXTRACTION_ENABLED: "true", AGENTMEMORY_GRAPH_LEG: "off", CONSOLIDATION_ENABLED: "false" });
    const kv = spyKV();
    const sdk = await stopSession(kv);
    expect(sdk.dispatched).not.toContain("mem::graph-extract");

    // The REST/MCP surface can still call the function directly.
    const result = (await sdk.trigger("mem::graph-extract", { observations: [OBS] })) as {
      success: boolean;
      nodesAdded: number;
      edgesAdded: number;
    };
    expect(result).toMatchObject({ success: true, nodesAdded: 0, edgesAdded: 0 });
    expect(provider.compress).not.toHaveBeenCalled();
    expect(graphWrites(kv)).toEqual([]);

    // The chokepoint every writer funnels through must refuse on its own too.
    const node = { id: "gn_1", type: "file", name: "src/functions/graph.ts", properties: {}, sourceObservationIds: [OBS.id], createdAt: OBS.timestamp };
    expect(await persistGraphDelta(kv as never, [node as never], [], [OBS.id])).toEqual({ newNodeCount: 0, newEdgeCount: 0 });
    expect(graphWrites(kv)).toEqual([]);
  });

  it("replace-import with writes off neither wipes nor restores the graph scope", async () => {
    // mem::import bypasses persistGraphDelta, so it carries the same gate itself;
    // wipe and restore must agree or a replace-import empties the graph.
    setEnv({ GRAPH_EXTRACTION_ENABLED: "false", CONSOLIDATION_ENABLED: "false" });
    const kv = spyKV();
    const existing = { id: "gn_existing", type: "file", name: "old.ts", properties: {}, sourceObservationIds: [], createdAt: OBS.timestamp };
    await kv.set(KV.graphNodes, existing.id, existing);
    kv.sets.length = 0;
    const sdk = dispatchingSdk();
    registerExportImportFunction(sdk as never, kv as never);
    const result = (await sdk.trigger("mem::import", {
      exportData: {
        version: "0.9.29", exportedAt: OBS.timestamp, sessions: [], observations: {}, memories: [], summaries: [],
        graphNodes: [{ ...existing, id: "gn_new", name: "new.ts" }], graphEdges: [],
      },
      strategy: "replace",
    })) as { success: boolean };
    expect(result.success).toBe(true);
    expect(await kv.get(KV.graphNodes, existing.id)).toEqual(existing);
    expect(await kv.get(KV.graphNodes, "gn_new")).toBeNull();
    expect(graphWrites(kv)).toEqual([]);
  });

  it("positive control: flag on, leg unset writes graph rows and side-indexes", async () => {
    setEnv({ GRAPH_EXTRACTION_ENABLED: "true", CONSOLIDATION_ENABLED: "false" });
    const kv = spyKV();
    const sdk = await stopSession(kv);
    expect(sdk.dispatched).toContain("mem::graph-extract");
    const scopes = new Set(graphWrites(kv).map(([scope]) => scope));
    expect(scopes.has(KV.graphNodes)).toBe(true);
    expect(scopes.has(KV.graphNameIndex)).toBe(true);
  });
});
