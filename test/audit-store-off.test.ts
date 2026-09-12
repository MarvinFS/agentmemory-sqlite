import { describe, it, expect, afterEach, vi } from "vitest";
import { recordAudit, safeAudit, queryAudit } from "../src/functions/audit.js";
import type { StateKV } from "../src/state/kv.js";

function fakeKv() {
  return {
    set: vi.fn(async () => undefined),
    list: vi.fn(async () => [{ id: "aud_x", timestamp: "2026-01-01T00:00:00.000Z", operation: "delete" }]),
  } as unknown as StateKV & { set: ReturnType<typeof vi.fn>; list: ReturnType<typeof vi.fn> };
}

describe("AGENTMEMORY_AUDIT_STORE=off", () => {
  const saved = process.env.AGENTMEMORY_AUDIT_STORE;
  const savedLog = process.env.AGENTMEMORY_AUDIT_LOG;
  afterEach(() => {
    if (saved === undefined) delete process.env.AGENTMEMORY_AUDIT_STORE;
    else process.env.AGENTMEMORY_AUDIT_STORE = saved;
    if (savedLog === undefined) delete process.env.AGENTMEMORY_AUDIT_LOG;
    else process.env.AGENTMEMORY_AUDIT_LOG = savedLog;
    vi.restoreAllMocks();
  });

  it("emits one structured line with the target ids and writes nothing to KV", async () => {
    process.env.AGENTMEMORY_AUDIT_STORE = "off";
    process.env.AGENTMEMORY_AUDIT_LOG = "all";
    const kv = fakeKv();
    const out: string[] = [];
    // logger.audit writes `[agentmemory] audit <operation> <json>` to stderr.
    vi.spyOn(process.stderr, "write").mockImplementation(((chunk: unknown) => {
      out.push(String(chunk));
      return true;
    }) as never);

    const entry = await recordAudit(kv, "forget", "mem::forget", ["mem_1", "obs_2"], { reason: "test" });
    await safeAudit(kv, "delete", "mem::governance-delete", ["mem_3"], { reason: "bulk" });

    expect(entry.targetIds).toEqual(["mem_1", "obs_2"]);
    expect(kv.set).not.toHaveBeenCalled();
    const text = out.join("\n");
    expect(text).toContain("mem::forget");
    expect(text).toContain("mem_1");
    expect(text).toContain("obs_2");
    expect(text).toContain("mem::governance-delete");
    expect(text).toContain("mem_3");

    expect(await queryAudit(kv, { operation: "delete" })).toEqual([]);
    expect(kv.list).not.toHaveBeenCalled();
  });

  it("AGENTMEMORY_AUDIT_LOG=deletions keeps only the paths that remove data", async () => {
    process.env.AGENTMEMORY_AUDIT_STORE = "off";
    process.env.AGENTMEMORY_AUDIT_LOG = "deletions";
    const kv = fakeKv();
    const out: string[] = [];
    vi.spyOn(process.stderr, "write").mockImplementation(((chunk: unknown) => {
      out.push(String(chunk));
      return true;
    }) as never);
    await recordAudit(kv, "observe", "mem::observe", ["obs_1"]);
    await recordAudit(kv, "compress", "mem::compress", ["obs_1"]);
    await recordAudit(kv, "forget", "mem::forget", ["mem_9"]);
    await recordAudit(kv, "lesson_delete", "mem::lesson-delete", ["lsn_9"]);
    // Destructive variants under a broader name stay; their repair/merge siblings go.
    await recordAudit(kv, "heal", "mem::heal", ["lease_del"], { entityType: "lease", action: "delete" });
    await recordAudit(kv, "heal", "mem::heal", ["act_upd"], { entityType: "action", action: "update" });
    await recordAudit(kv, "mesh_sync", "mem::mesh-remove", ["peer_del"], { action: "mesh.remove" });
    await recordAudit(kv, "mesh_sync", "mem::mesh-sync", ["peer_sync"], { action: "mesh.sync" });
    await recordAudit(kv, "import", "mem::import", [], { strategy: "replace", stats: { memories: 3 } });
    await recordAudit(kv, "import", "mem::import", [], { strategy: "merge", stats: { memories: 3 } });
    // Soft deletions: a lesson decayed to deleted, an insight sweep that soft-deleted, a sketch discarded by heal.
    await recordAudit(kv, "lesson_strengthen", "mem::lesson-decay-sweep", ["lsn_soft"], { action: "soft-delete", reason: "decay-sweep" });
    await recordAudit(kv, "lesson_strengthen", "mem::lesson-decay-sweep", ["lsn_decay"], { action: "decay", reason: "decay-sweep" });
    await recordAudit(kv, "reflect", "mem::insight-decay-sweep", ["ins_soft"], { event: "insight.decay", decayed: 4, softDeleted: 1 });
    await recordAudit(kv, "reflect", "mem::insight-decay-sweep", ["ins_decay"], { event: "insight.decay", decayed: 4, softDeleted: 0 });
    await recordAudit(kv, "heal", "mem::heal", ["sk_disc"], { entityType: "sketch", reason: "expired-sketch", newStatus: "discarded" });
    const text = out.join("\n");
    expect(text).not.toContain("obs_1");
    expect(text).toContain("lease_del");
    expect(text).not.toContain("act_upd");
    expect(text).toContain("peer_del");
    expect(text).not.toContain("peer_sync");
    expect(text).toContain("\"strategy\":\"replace\"");
    expect(text).not.toContain("\"strategy\":\"merge\"");
    expect(text).toContain("lsn_soft");
    expect(text).not.toContain("lsn_decay");
    expect(text).toContain("ins_soft");
    expect(text).not.toContain("ins_decay");
    expect(text).toContain("sk_disc");
    expect(text).toContain("] audit forget ");
    expect(text).toContain("mem_9");
    expect(text).toContain("lsn_9");
    expect(kv.set).not.toHaveBeenCalled();

    out.length = 0;
    process.env.AGENTMEMORY_AUDIT_LOG = "off";
    await recordAudit(kv, "forget", "mem::forget", ["mem_10"]);
    expect(out).toEqual([]);
  });

  it("keeps the KV store for any other value", async () => {
    process.env.AGENTMEMORY_AUDIT_STORE = "on";
    const kv = fakeKv();
    await recordAudit(kv, "forget", "mem::forget", ["mem_1"]);
    expect(kv.set).toHaveBeenCalledTimes(1);
    expect(await queryAudit(kv)).toHaveLength(1);
  });
});
