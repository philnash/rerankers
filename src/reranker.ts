import { normalizeDocuments, normalizeTopK } from "./document.js";
import { UnsupportedStrategyError } from "./errors.js";
import { createCrossEncoderStrategy } from "./strategies/cross-encoder.js";
import type {
  NormalizedRerankerConfig,
  RankOptions,
  RerankerConfig,
  RerankerCreateOptions,
  RerankDocument,
  RerankResult,
  ScoringStrategy,
} from "./types.js";

export class Reranker {
  private constructor(private readonly strategy: ScoringStrategy) {}

  static async create(
    config: RerankerConfig,
    options: RerankerCreateOptions = {},
  ): Promise<Reranker> {
    const strategy = await (options.strategyFactory ?? createDefaultStrategy)(
      normalizeConfig(config),
    );

    return new Reranker(strategy);
  }

  async rank<TDocument extends RerankDocument>(
    query: string,
    documents: readonly TDocument[],
    options?: RankOptions,
  ): Promise<Array<RerankResult<TDocument>>> {
    const normalized = normalizeDocuments(documents);
    if (normalized.length === 0) {
      return [];
    }

    const topK = normalizeTopK(options, normalized.length);
    const results = await this.strategy.score(query, normalized);
    return [...results].sort((left, right) => right.score - left.score).slice(0, topK);
  }
}

function normalizeConfig(config: RerankerConfig): NormalizedRerankerConfig {
  return {
    ...config,
    strategy: config.strategy ?? "cross-encoder",
  };
}

function createDefaultStrategy(config: NormalizedRerankerConfig): Promise<ScoringStrategy> {
  if (config.strategy === "cross-encoder") {
    return Promise.resolve(createCrossEncoderStrategy(config));
  }

  throw new UnsupportedStrategyError(config.strategy);
}
