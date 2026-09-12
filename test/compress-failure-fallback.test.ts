import { describe, it, expect, vi } from "vitest";
import type { RawObservation } from "../src/types.js";

vi.mock("../src/logger.js", () => ({
  logger: { info: vi.fn(), warn: vi.fn(), error: vi.fn() },
}));

function mockKV() {
  const store = new Map<string, Map<string, unknown>>();
  return {
    store,
    get: async <T>(scope: string, key: string): Promise<T | null> =>
      (store.get(scope)?.get(key) as T) ?? null,
    set: async <T>(scope: string, key: string, data: T): Promise<T> => {
      if (!store.has(scope)) store.set(scope, new Map());
      store.get(scope)!.set(key, data);
      return data;
    },
    delete: async (scope: string, key: string) => {
      store.get(scope)?.delete(key);
    },
    list: async <T>(scope: string): Promise<T[]> => {
      const m = store.get(scope);
      return m ? (Array.from(m.values()) as T[]) : [];
    },
  };
}

function mockSdk() {
  const fns = new Map<string, Function>();
  return {
    fns,
    registerFunction: (idOrOpts: string | { id: string }, fn: Function) => {
      fns.set(typeof idOrOpts === "string" ? idOrOpts : idOrOpts.id, fn);
    },
    trigger: async (
      idOrInput: string | { function_id: string; payload: unknown },
      data?: unknown,
    ) => {
      const id =
        typeof idOrInput === "string" ? idOrInput : idOrInput.function_id;
      const payload = typeof idOrInput === "string" ? data : idOrInput.payload;
      const fn = fns.get(id);
      return fn ? fn(payload) : null;
    },
  };
}

const RAW: RawObservation = {
  id: "obs_test_orphan",
  sessionId: "ses_test",
  timestamp: new Date().toISOString(),
  hookType: "post_tool_use",
  toolName: "Read",
  toolInput: { file_path: "src/foo.ts" },
  toolOutput: "file contents here",
};

async function runCompress(compress: () => Promise<string>) {
  const { registerCompressFunction } = await import(
    "../src/functions/compress.js"
  );
  const sdk = mockSdk();
  const kv = mockKV();
  registerCompressFunction(sdk as never, kv as never, {
    compress,
  } as never);

  const result = (await sdk.trigger("mem::compress", {
    observationId: RAW.id,
    sessionId: RAW.sessionId,
    raw: RAW,
  })) as { success: boolean; error?: string };

  return {
    result,
    stored: kv.store.get(`mem:obs:${RAW.sessionId}`),
    pending: kv.store.get("mem:compress-pending"),
    kv,
  };
}

// The bug this guards: a transient provider 5xx used to return early WITHOUT
// writing KV.observations, and nothing re-compresses later - so the record
// stayed on disk as raw bytes but was absent from both search legs forever.
describe("mem::compress never orphans an observation on failure", () => {
  it("provider throws (Anthropic 529): still stores an indexable record", async () => {
    const { result, stored, pending } = await runCompress(async () => {
      throw new Error("529 overloaded_error");
    });

    // The synthetic truncates toolOutput, so the untouched original must be
    // parked for the re-compression sweep or the detail is gone for good.
    const parked = pending?.get(RAW.id) as
      | { raw: RawObservation; attempts: number }
      | undefined;
    expect(parked, "raw was not parked for recovery").toBeDefined();
    expect(parked!.raw.toolOutput).toBe(RAW.toolOutput);
    // First failure = attempt 1; the sweep quarantines at 5.
    expect(parked!.attempts).toBe(1);

    expect(result.success).toBe(false);
    expect(result.error).toBe("compression_failed");

    expect(stored, "observation was orphaned").toBeDefined();
    const obs = stored!.get(RAW.id) as {
      id: string;
      title: string;
      confidence: number;
    };
    expect(obs).toBeDefined();
    expect(obs.id).toBe(RAW.id);
    expect(obs.title).toBe("Read");
    // 0.3 is the marker a re-compression sweep selects on.
    expect(obs.confidence).toBe(0.3);
  });

  it("provider returns unparseable output: still stores an indexable record", async () => {
    const { result, stored } = await runCompress(async () => "not xml at all");

    expect(result.success).toBe(false);
    expect(result.error).toBe("parse_failed");
    expect(stored?.get(RAW.id), "observation was orphaned").toBeDefined();
  });

  it("provider succeeds: stores the real compression, not the degraded one", async () => {
    const { result, stored } = await runCompress(
      async () => `<observation>
  <type>file_read</type>
  <title>Read src/foo.ts</title>
  <facts><fact>a</fact><fact>b</fact><fact>c</fact></facts>
  <narrative>Read the file to inspect its contents before editing it.</narrative>
  <concepts><concept>file inspection</concept></concepts>
  <files><file>src/foo.ts</file></files>
  <importance>3</importance>
</observation>`,
    );

    expect(result.success).toBe(true);
    const obs = stored!.get(RAW.id) as { confidence: number; title: string };
    expect(obs.title).toBe("Read src/foo.ts");
    expect(obs.confidence).toBeGreaterThan(0.3);
  });

  it("a later successful re-compression clears the recovery queue", async () => {
    const { registerCompressFunction } = await import(
      "../src/functions/compress.js"
    );
    const sdk = mockSdk();
    const kv = mockKV();
    let failNext = true;
    registerCompressFunction(sdk as never, kv as never, {
      compress: async () => {
        if (failNext) throw new Error("529 overloaded_error");
        return `<observation>
  <type>file_read</type>
  <title>Read src/foo.ts</title>
  <facts><fact>a</fact><fact>b</fact><fact>c</fact></facts>
  <narrative>Read the file to inspect its contents before editing it.</narrative>
  <concepts><concept>file inspection</concept></concepts>
  <files><file>src/foo.ts</file></files>
  <importance>3</importance>
</observation>`;
      },
    } as never);

    const payload = {
      observationId: RAW.id,
      sessionId: RAW.sessionId,
      raw: RAW,
    };

    await sdk.trigger("mem::compress", payload);
    expect(kv.store.get("mem:compress-pending")?.get(RAW.id)).toBeDefined();

    failNext = false;
    await sdk.trigger("mem::compress", payload);
    expect(
      kv.store.get("mem:compress-pending")?.get(RAW.id),
      "recovery queue entry outlived a successful re-compression",
    ).toBeUndefined();
  });
});
