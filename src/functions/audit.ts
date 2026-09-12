import type { AuditEntry } from "../types.js";
import { KV, generateId } from "../state/schema.js";
import type { StateKV } from "../state/kv.js";
import { logger } from "../logger.js";

// Audit coverage policy (issue #125).
//
// Every structural deletion of a memory, observation, session, or
// semantic row MUST call recordAudit. Two shapes are allowed, keyed to
// whether the caller is scoped or bulk:
//
//   Scoped deletions — a user-visible, per-call action removing a
//   bounded set of items. Emit ONE audit row per call with targetIds
//   populated. Examples: mem::governance-delete, mem::forget.
//
//   Bulk deletions — automatic sweeps (retention, TTL eviction,
//   auto-forget) that can remove hundreds of rows per invocation.
//   Emit ONE batched audit row per invocation with targetIds listing
//   every removed id and details.evicted holding the count. Per-item
//   audit rows would flood the audit log during routine sweeps.
//
//   Either shape is required; silent deletes are not acceptable.
//
// operation field:
//   - "delete"          — permanent removal (governance, retention sweep, evict).
//   - "forget"          — forget/removal flows. Scoped when emitted by
//                         mem::forget (user-initiated); bulk-batched when
//                         emitted by mem::auto-forget (automatic sweep).
//   - everything else   — see AuditEntry["operation"] union in src/types.ts.
//
// When adding a new deletion path, add an explicit recordAudit call
// BEFORE kv.delete(...) and match one of the two shapes above.
//
// Storage: with AGENTMEMORY_AUDIT_STORE=off the entry is emitted as one
// structured log line (journald -> Loki keeps the evidence: operation,
// functionId, targetIds, details) and nothing is written to KV; the audit
// query then answers an empty list. Any other value keeps the KV store.
//
// Which entries become lines when the store is off: AGENTMEMORY_AUDIT_LOG=
// all (default) | deletions | off. `deletions` keeps every path that removes
// data - the six removal operations plus the removals that hide under a
// broader name: heal with action "delete", mesh peer removal, an import
// that replaces the store, and the soft deletions (a lesson or insight
// decayed to deleted, a sketch discarded by heal) - and drops the
// per-operation chatter (observe, compress, heal repairs ...): a few lines
// a month instead of ~100 KB a day. The details checks mirror the payloads
// at those call sites; a new deletion path needs either a listed operation
// or one of these markers.

function auditStoreOff(): boolean {
  return (process.env.AGENTMEMORY_AUDIT_STORE ?? "").toLowerCase() === "off";
}

const DELETION_OPS: ReadonlySet<AuditEntry["operation"]> = new Set([
  "forget", "delete", "lesson_delete", "slot_delete", "core_remove", "sketch_discard",
]);

function isDeletion(operation: AuditEntry["operation"], details: Record<string, unknown>): boolean {
  if (DELETION_OPS.has(operation)) return true;
  if (details.action === "delete" || details.action === "soft-delete" || details.action === "mesh.remove") return true;
  if (details.newStatus === "discarded") return true;
  if (typeof details.softDeleted === "number" && details.softDeleted > 0) return true;
  return operation === "import" && details.strategy === "replace";
}

function auditLineWanted(operation: AuditEntry["operation"], details: Record<string, unknown>): boolean {
  const mode = (process.env.AGENTMEMORY_AUDIT_LOG ?? "all").toLowerCase();
  if (mode === "off") return false;
  if (mode === "deletions") return isDeletion(operation, details);
  return true;
}

export async function recordAudit(
  kv: StateKV,
  operation: AuditEntry["operation"],
  functionId: string,
  targetIds: string[],
  details: Record<string, unknown> = {},
  qualityScore?: number,
  userId?: string,
): Promise<AuditEntry> {
  const entry: AuditEntry = {
    id: generateId("aud"),
    timestamp: new Date().toISOString(),
    operation,
    userId,
    functionId,
    targetIds,
    details,
    qualityScore,
  };
  if (auditStoreOff()) {
    if (!auditLineWanted(operation, details)) return entry;
    logger.audit(operation, {
      auditId: entry.id,
      functionId,
      targetIds,
      targetCount: targetIds.length,
      details,
      ...(qualityScore !== undefined ? { qualityScore } : {}),
      ...(userId !== undefined ? { userId } : {}),
    });
    return entry;
  }
  await kv.set(KV.audit, entry.id, entry);
  return entry;
}

export async function safeAudit(
  kv: StateKV,
  operation: AuditEntry["operation"],
  functionId: string,
  targetIds: string[],
  details: Record<string, unknown> = {},
  qualityScore?: number,
  userId?: string,
): Promise<void> {
  try {
    await recordAudit(kv, operation, functionId, targetIds, details, qualityScore, userId);
  } catch (err) {
    try {
      logger.warn("audit write failed", {
        functionId,
        operation,
        targetIds,
        error: err instanceof Error ? err.message : String(err),
      });
    } catch {}
  }
}

export async function queryAudit(
  kv: StateKV,
  filter?: {
    operation?: AuditEntry["operation"];
    dateFrom?: string;
    dateTo?: string;
    limit?: number;
  },
): Promise<AuditEntry[]> {
  if (auditStoreOff()) return [];
  const all = await kv.list<AuditEntry>(KV.audit);
  let entries = [...all].sort(
    (a, b) => new Date(b.timestamp).getTime() - new Date(a.timestamp).getTime(),
  );

  if (filter?.operation) {
    entries = entries.filter((e) => e.operation === filter.operation);
  }
  if (filter?.dateFrom) {
    const from = new Date(filter.dateFrom).getTime();
    if (Number.isNaN(from)) {
      throw new Error(`Invalid dateFrom: ${filter.dateFrom}`);
    }
    entries = entries.filter((e) => new Date(e.timestamp).getTime() >= from);
  }
  if (filter?.dateTo) {
    const to = new Date(filter.dateTo).getTime();
    if (Number.isNaN(to)) {
      throw new Error(`Invalid dateTo: ${filter.dateTo}`);
    }
    entries = entries.filter((e) => new Date(e.timestamp).getTime() <= to);
  }

  return entries.slice(0, filter?.limit || 100);
}
