import { normalizeDocuments, normalizeK } from "./document.js";
import { resolvePreset } from "./presets.js";
import { createCrossEncoderStrategy } from "./strategies/cross-encoder.js";
import { LateInteractionStrategy } from "./strategies/late-interaction.js";
import type {
  RankOptions,
  RerankerConfig,
  RerankerCreateOptions,
  RerankerDefinition,
  RerankDocument,
  RerankResult,
  ScoringStrategy,
} from "./types.js";

export class Reranker {
  private constructor(private readonly strategy: ScoringStrategy) {}

  static async create(
    definition: RerankerDefinition = "bge",
    options: RerankerCreateOptions = {},
  ): Promise<Reranker> {
    const config = typeof definition === "string" ? resolvePreset(definition) : { ...definition };
    const strategy = await (options.strategyFactory ?? createDefaultStrategy)(config);

    return new Reranker(strategy);
  }

  async rank<TDocument extends RerankDocument>(
    query: string,
    documents: readonly TDocument[],
    options?: RankOptions,
  ): Promise<Array<RerankResult<TDocument>>> {
    const normalized = normalizeDocuments(documents);
    const k = normalizeK(options, normalized.length);

    if (normalized.length === 0) {
      return [];
    }

    const results = await this.strategy.score(query, normalized);
    return [...results].sort((left, right) => right.score - left.score).slice(0, k);
  }
}

function createDefaultStrategy(config: RerankerConfig): Promise<ScoringStrategy> {
  if (config.strategy === "cross-encoder") {
    return Promise.resolve(createCrossEncoderStrategy(config));
  }

  return Promise.resolve(
    new LateInteractionStrategy(() => {
      throw new Error(
        "Late-interaction models need a token-vector embedder. Provide a custom strategyFactory for this model.",
      );
    }),
  );
}
