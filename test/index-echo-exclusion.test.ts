import { describe, it, expect, beforeEach, afterEach, vi } from "vitest";

vi.mock("../src/logger.js", () => ({
  logger: { info: vi.fn(), warn: vi.fn(), error: vi.fn() },
}));

import {
  isIndexExcluded,
  rebuildIndex,
  getSearchIndex,
  setVectorIndex,
  setEmbeddingProvider,
  getVectorIndex,
} from "../src/functions/search.js";
import { VectorIndex } from "../src/state/vector-index.js";
import { KV } from "../src/state/schema.js";
import type { CompressedObservation, Session } from "../src/types.js";

const FLAG = "AGENTMEMORY_INDEX_EXCLUDE_TOOL_PREFIXES";
const ECHO_PREFIX = "mcp__plugin_agentmemory_agentmemory__";

function mockKV() {
  const store = new Map<string, Map<string, unknown>>();
  return {
    get: async <T>(scope: string, key: string): Promise<T | null> =>
      (store.get(scope)?.get(key) as T) ?? null,
    set: async <T>(scope: string, key: string, data: T): Promise<T> => {
      if (!store.has(scope)) store.set(scope, new Map());
      store.get(scope)!.set(key, data);
      return data;
    },
    delete: async (scope: string, key: string): Promise<void> => {
      store.get(scope)?.delete(key);
    },
    list: async <T>(scope: string): Promise<T[]> =>
      Array.from(store.get(scope)?.values() ?? []) as T[],
  };
}

function obs(id: string, toolName?: string): CompressedObservation {
  return {
    id,
    sessionId: "ses_1",
    timestamp: "2026-01-01T00:00:00Z",
    type: "search",
    title: toolName ? `Memory search: ${id}` : `Real work ${id}`,
    facts: [],
    narrative: "residentKey teleport vnet mainpc-rdp",
    concepts: [],
    files: [],
    importance: 5,
    ...(toolName ? { toolName } : {}),
  };
}

describe("index exclusion by toolName prefix", () => {
  const saved = process.env[FLAG];

  beforeEach(() => {
    delete process.env[FLAG];
    getSearchIndex().clear();
    setVectorIndex(null);
    setEmbeddingProvider(null);
  });
  afterEach(() => {
    if (saved === undefined) delete process.env[FLAG];
    else process.env[FLAG] = saved;
    getSearchIndex().clear();
    setVectorIndex(null);
    setEmbeddingProvider(null);
  });

  it("excludes nothing when the flag is unset (default behaviour)", () => {
    expect(isIndexExcluded(obs("obs_1", `${ECHO_PREFIX}memory_smart_search`))).toBe(false);
  });

  it("matches on prefix, and only on the configured prefixes", () => {
    process.env[FLAG] = ECHO_PREFIX;
    expect(isIndexExcluded(obs("o", `${ECHO_PREFIX}memory_smart_search`))).toBe(true);
    expect(isIndexExcluded(obs("o", `${ECHO_PREFIX}memory_recall`))).toBe(true);
    expect(isIndexExcluded(obs("o", "Bash"))).toBe(false);
    expect(isIndexExcluded(obs("o", "mcp__veeam-rag__hybrid_search"))).toBe(false);
    // A record with no toolName (every pre-existing row on disk) is never
    // matched by this predicate — that is why legacy echoes need a
    // separate, explicitly reviewed deletion pass.
    expect(isIndexExcluded(obs("o"))).toBe(false);
  });

  it("accepts a comma-separated list and tolerates whitespace", () => {
    process.env[FLAG] = ` ${ECHO_PREFIX} , other__ `;
    expect(isIndexExcluded(obs("o", `${ECHO_PREFIX}x`))).toBe(true);
    expect(isIndexExcluded(obs("o", "other__thing"))).toBe(true);
    expect(isIndexExcluded(obs("o", "unrelated"))).toBe(false);
  });

  it("keeps the row in KV while dropping it from BOTH indexes", async () => {
    process.env[FLAG] = ECHO_PREFIX;
    const kv = mockKV();
    const session: Session = {
      id: "ses_1",
      project: "demo",
      cwd: "/tmp/demo",
      startedAt: "2026-01-01T00:00:00Z",
      status: "completed",
      observationCount: 2,
    };
    await kv.set(KV.sessions, session.id, session);

    const echo = obs("obs_echo", `${ECHO_PREFIX}memory_smart_search`);
    const real = obs("obs_real");
    await kv.set(KV.observations("ses_1"), echo.id, echo);
    await kv.set(KV.observations("ses_1"), real.id, real);

    setVectorIndex(new VectorIndex());
    setEmbeddingProvider({
      name: "test",
      dimensions: 3,
      embed: async () => new Float32Array([0.1, 0.2, 0.3]),
      embedBatch: async (t: string[]) =>
        t.map(() => new Float32Array([0.1, 0.2, 0.3])),
    });

    const count = await rebuildIndex(kv as never);

    // Indexed: only the real observation.
    expect(count).toBe(1);
    expect(getSearchIndex().has("obs_real")).toBe(true);
    expect(getSearchIndex().has("obs_echo")).toBe(false);
    expect(
      getVectorIndex()!
        .search(new Float32Array([0.1, 0.2, 0.3]), 10)
        .map((r) => r.obsId),
    ).toEqual(["obs_real"]);

    // Kept: the echo is still fully readable from KV, so timeline,
    // session history, the viewer and expandIds all still see it.
    const stored = await kv.get<CompressedObservation>(
      KV.observations("ses_1"),
      "obs_echo",
    );
    expect(stored).not.toBeNull();
    expect(stored!.title).toContain("Memory search:");
    expect(
      (await kv.list<CompressedObservation>(KV.observations("ses_1"))).length,
    ).toBe(2);
  });
});
