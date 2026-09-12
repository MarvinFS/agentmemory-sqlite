import { describe, it, expect, afterEach, vi } from "vitest";
import { logger } from "../src/logger.js";

function capture(): string[] {
  const out: string[] = [];
  vi.spyOn(process.stderr, "write").mockImplementation(((chunk: unknown) => {
    out.push(String(chunk));
    return true;
  }) as never);
  return out;
}

describe("AGENTMEMORY_LOG_LEVEL", () => {
  const saved = process.env.AGENTMEMORY_LOG_LEVEL;
  afterEach(() => {
    if (saved === undefined) delete process.env.AGENTMEMORY_LOG_LEVEL;
    else process.env.AGENTMEMORY_LOG_LEVEL = saved;
    vi.restoreAllMocks();
  });

  it("unset writes every level", () => {
    delete process.env.AGENTMEMORY_LOG_LEVEL;
    const out = capture();
    logger.info("i");
    logger.warn("w");
    logger.error("e");
    expect(out.map((l) => l.trim())).toEqual(["[agentmemory] info i", "[agentmemory] warn w", "[agentmemory] error e"]);
  });

  it("warn drops info and keeps warn, error and audit", () => {
    process.env.AGENTMEMORY_LOG_LEVEL = "warn";
    const out = capture();
    logger.info("i", { a: 1 });
    logger.warn("w");
    logger.error("e");
    logger.audit("forget", { targetIds: ["mem_1"] });
    expect(out.join("")).not.toContain("info i");
    expect(out.map((l) => l.trim())).toEqual([
      "[agentmemory] warn w",
      "[agentmemory] error e",
      '[agentmemory] audit forget {"targetIds":["mem_1"]}',
    ]);
  });

  it("an unknown value, including an inherited property name, means info", () => {
    const out = capture();
    for (const v of ["nonsense", "constructor", "__proto__"]) {
      process.env.AGENTMEMORY_LOG_LEVEL = v;
      logger.info("i");
    }
    expect(out).toHaveLength(3);
  });

  it("off silences everything except audit", () => {
    process.env.AGENTMEMORY_LOG_LEVEL = "off";
    const out = capture();
    logger.error("e");
    logger.audit("delete");
    expect(out.map((l) => l.trim())).toEqual(["[agentmemory] audit delete"]);
  });
});
