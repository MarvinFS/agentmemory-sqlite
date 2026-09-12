import { describe, expect, it } from "vitest";
import { evaluateHealth } from "../src/health/thresholds.js";
import type { HealthSnapshot } from "../src/types.js";

const MB = 1024 * 1024;

function snap(over: Partial<HealthSnapshot> = {}): HealthSnapshot {
  return {
    connectionState: "connected",
    workers: [],
    memory: { heapUsed: 0, heapTotal: 1, heapLimit: 4096 * MB, rss: 0, external: 0 },
    cpu: { userMicros: 0, systemMicros: 0, percent: 0 },
    eventLoopLagMs: 0,
    uptimeSeconds: 1,
    kvConnectivity: { status: "ok", latencyMs: 1 },
    status: "healthy",
    alerts: [],
    ...over,
  };
}

describe("evaluateHealth memory severity", () => {
  it("stays healthy when heap fills a tiny steady-state process (issue #158)", () => {
    const s = snap({
      memory: {
        heapUsed: 45 * MB,
        heapTotal: 46 * MB,
        heapLimit: 4096 * MB,
        rss: 120 * MB,
        external: 0,
      },
    });
    const { status, alerts, notes } = evaluateHealth(s);
    expect(status).toBe("healthy");
    expect(alerts.find((a) => a.startsWith("memory_"))).toBeUndefined();
    expect(notes.find((n) => n.startsWith("memory_heap_tight_"))).toBeUndefined();
  });

  it("stays healthy on the prod false-positive: heapUsed/heapTotal 93% with gigabytes of limit headroom", () => {
    // The 2026-08-19 readout: heapUsed 649MB, heapTotal 695MB, rss 1073MB on a
    // 16GB VM. Against heapTotal that is 93% and alerted forever; against the
    // real 4GB ceiling it is ~16%.
    const s = snap({
      memory: {
        heapUsed: 649 * MB,
        heapTotal: 695 * MB,
        heapLimit: 4096 * MB,
        rss: 1073 * MB,
        external: 0,
      },
    });
    const { status, alerts, notes } = evaluateHealth(s);
    expect(status).toBe("healthy");
    expect(alerts.find((a) => a.startsWith("memory_"))).toBeUndefined();
    expect(notes.find((n) => n.startsWith("memory_heap_tight_"))).toBeUndefined();
  });

  it("goes critical when heapUsed nears the limit AND RSS is above the floor", () => {
    const s = snap({
      memory: {
        heapUsed: 970 * MB,
        heapTotal: 1000 * MB,
        heapLimit: 1000 * MB,
        rss: 1100 * MB,
        external: 0,
      },
    });
    const { status, alerts } = evaluateHealth(s);
    expect(status).toBe("critical");
    expect(alerts.some((a) => a.startsWith("memory_critical_"))).toBe(true);
  });

  it("records heap_tight in the warn band when RSS is below the floor", () => {
    const s = snap({
      memory: {
        heapUsed: 85 * MB,
        heapTotal: 100 * MB,
        heapLimit: 100 * MB,
        rss: 50 * MB,
        external: 0,
      },
    });
    const { status, alerts, notes } = evaluateHealth(s);
    expect(status).toBe("healthy");
    expect(notes.some((n) => n.startsWith("memory_heap_tight_"))).toBe(true);
    expect(alerts.some((a) => a.startsWith("memory_"))).toBe(false);
  });

  it("goes degraded when heapUsed is above warn AND RSS is above the floor", () => {
    const s = snap({
      memory: {
        heapUsed: 850 * MB,
        heapTotal: 1000 * MB,
        heapLimit: 1000 * MB,
        rss: 900 * MB,
        external: 0,
      },
    });
    const { status, alerts } = evaluateHealth(s, { memoryRssFloorBytes: 800 * MB });
    expect(status).toBe("degraded");
    expect(alerts.some((a) => a.startsWith("memory_warn_"))).toBe(true);
  });

  it("respects caller-supplied memoryRssFloorBytes", () => {
    const s = snap({
      memory: {
        heapUsed: 98,
        heapTotal: 100,
        heapLimit: 100,
        rss: 50 * MB,
        external: 0,
      },
    });
    const loose = evaluateHealth(s, { memoryRssFloorBytes: 10 * MB });
    expect(loose.status).toBe("critical");
    const strict = evaluateHealth(s, { memoryRssFloorBytes: 1024 * MB });
    expect(strict.status).toBe("healthy");
  });

  it("treats a missing heapLimit as no memory signal (snapshot from an old build)", () => {
    const s = snap();
    // a stored snapshot that predates the heapLimit field
    s.memory.heapLimit = undefined;
    s.memory.heapUsed = 970 * MB;
    s.memory.heapTotal = 1000 * MB;
    s.memory.rss = 1100 * MB;
    const { status, alerts } = evaluateHealth(s);
    expect(status).toBe("healthy");
    expect(alerts.some((a) => a.startsWith("memory_"))).toBe(false);
  });
});
