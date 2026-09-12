import type { ISdk } from "iii-sdk";
import type { MemoryProvider, QueryExpansion } from "../types.js";
import { logger } from "../logger.js";

const QUERY_EXPANSION_SYSTEM = `You are a query expansion engine for a memory retrieval system. Given a user query, generate diverse reformulations to maximize recall.

Output EXACTLY this XML:
<expansion>
  <reformulations>
    <query>semantically diverse rephrasing 1</query>
    <query>semantically diverse rephrasing 2</query>
    <query>semantically diverse rephrasing 3</query>
  </reformulations>
  <temporal>
    <query>time-concretized version if applicable</query>
  </temporal>
  <entities>
    <entity>extracted entity name 1</entity>
    <entity>extracted entity name 2</entity>
  </entities>
</expansion>

Rules:
- Generate 3-5 reformulations capturing different interpretations
- Include paraphrases, domain-specific restatements, and abstract/concrete variants
- Extract any named entities (people, files, projects, libraries, concepts)
- If the query mentions time ("last week", "recently"), generate temporal concretizations
- Each reformulation should capture a distinct facet of intent
- Keep reformulations concise (under 100 chars each)`;

function parseExpansionXml(xml: string): QueryExpansion | null {
  const reformulations: string[] = [];
  const queryRegex =
    /<reformulations>[\s\S]*?<\/reformulations>/;
  const reformBlock = xml.match(queryRegex);
  if (reformBlock) {
    const qRegex = /<query>([^<]+)<\/query>/g;
    let match;
    while ((match = qRegex.exec(reformBlock[0])) !== null) {
      reformulations.push(match[1].trim());
    }
  }

  const temporalConcretizations: string[] = [];
  const tempBlock = xml.match(/<temporal>[\s\S]*?<\/temporal>/);
  if (tempBlock) {
    const qRegex = /<query>([^<]+)<\/query>/g;
    let match;
    while ((match = qRegex.exec(tempBlock[0])) !== null) {
      temporalConcretizations.push(match[1].trim());
    }
  }

  const entityExtractions: string[] = [];
  const entityRegex = /<entity>([^<]+)<\/entity>/g;
  let match;
  while ((match = entityRegex.exec(xml)) !== null) {
    entityExtractions.push(match[1].trim());
  }

  return {
    original: "",
    reformulations,
    temporalConcretizations,
    entityExtractions,
  };
}

function emptyExpansion(query: string): QueryExpansion {
  return {
    original: query,
    reformulations: [],
    temporalConcretizations: [],
    entityExtractions: [],
  };
}

/**
 * One LLM round-trip that turns a query into reformulations plus entities.
 *
 * Extracted from the `mem::expand-query` registration below so the retrieval
 * path can call it directly instead of round-tripping through the SDK trigger.
 * Never throws: a provider failure degrades to an EMPTY expansion, which makes
 * `searchWithExpansion` behave exactly like a plain search. The failure is
 * logged at error level rather than swallowed, because a provider that is
 * always failing here should be visible in the journal.
 */
export async function expandQuery(
  provider: MemoryProvider,
  query: string,
  maxReformulations = 5,
): Promise<QueryExpansion> {
  try {
    const response = await provider.compress(
      QUERY_EXPANSION_SYSTEM,
      `Expand this query for memory retrieval:\n\n"${query}"`,
    );

    const parsed = parseExpansionXml(response);
    if (!parsed) {
      logger.warn("Failed to parse query expansion");
      return emptyExpansion(query);
    }

    parsed.original = query;
    parsed.reformulations = parsed.reformulations.slice(0, maxReformulations);

    logger.info("Query expanded", {
      original: query,
      reformulations: parsed.reformulations.length,
      entities: parsed.entityExtractions.length,
    });

    return parsed;
  } catch (err) {
    const msg = err instanceof Error ? err.message : String(err);
    logger.error("Query expansion failed", { error: msg });
    return emptyExpansion(query);
  }
}

export function registerQueryExpansionFunction(
  sdk: ISdk,
  provider: MemoryProvider,
): void {
  sdk.registerFunction("mem::expand-query",
    async (data: { query: string; maxReformulations?: number } | undefined) => {
      if (!data || typeof data.query !== "string" || !data.query.trim()) {
        logger.warn("Invalid expand-query payload");
        return { success: false, error: "query must be a non-empty string" };
      }
      const rawMaxR = Number(data.maxReformulations);
      const maxR = Number.isFinite(rawMaxR)
        ? Math.max(1, Math.min(10, Math.floor(rawMaxR)))
        : 5;

      return {
        success: true,
        expansion: await expandQuery(provider, data.query.trim(), maxR),
      };
    },
  );
}

export function extractEntitiesFromQuery(query: string): string[] {
  const entities: string[] = [];
  const quoted = query.match(/"([^"]+)"/g);
  if (quoted) {
    for (const q of quoted) {
      entities.push(q.replace(/"/g, ""));
    }
  }
  const capitalized = query.match(/\b[A-Z][a-zA-Z0-9_.-]+\b/g);
  if (capitalized) {
    const stopWords = new Set([
      "The",
      "This",
      "That",
      "What",
      "When",
      "Where",
      "How",
      "Why",
      "Who",
      "Which",
      "Did",
      "Does",
      "Do",
      "Is",
      "Are",
      "Was",
      "Were",
      "Has",
      "Have",
      "Had",
      "Can",
      "Could",
      "Would",
      "Should",
      "Will",
      "May",
      "Might",
      "If",
      "And",
      "But",
      "Or",
      "Not",
      "For",
      "From",
      "With",
      "About",
      "After",
      "Before",
      "Between",
    ]);
    for (const c of capitalized) {
      if (!stopWords.has(c)) entities.push(c);
    }
  }
  return [...new Set(entities)];
}
