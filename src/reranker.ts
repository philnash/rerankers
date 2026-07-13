import { normalizeDocuments, normalizeTopK } from "./document.js";
import { RerankerDisposedError, UnsupportedStrategyError } from "./errors.js";
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

export class Reranker implements AsyncDisposable {
  private readonly activeRanks = new Set<Promise<unknown>>();
  private disposePromise: Promise<void> | undefined;
  private disposed = false;

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
    if (this.disposed) {
      throw new RerankerDisposedError();
    }

    const operation = this.rankDocuments(query, documents, options);
    this.activeRanks.add(operation);

    try {
      return await operation;
    } finally {
      this.activeRanks.delete(operation);
    }
  }

  dispose(): Promise<void> {
    this.disposePromise ??= this.disposeStrategy();
    return this.disposePromise;
  }

  [Symbol.asyncDispose](): Promise<void> {
    return this.dispose();
  }

  private async rankDocuments<TDocument extends RerankDocument>(
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

  private async disposeStrategy(): Promise<void> {
    this.disposed = true;
    await Promise.allSettled([...this.activeRanks]);
    await this.strategy.dispose?.();
  }
}

function normalizeConfig(config: RerankerConfig): NormalizedRerankerConfig {
  return {
    ...config,
    strategy: config.strategy ?? "cross-encoder",
    transformerOptions: {
      dtype: "auto",
      ...config.transformerOptions,
    },
  };
}

function createDefaultStrategy(config: NormalizedRerankerConfig): Promise<ScoringStrategy> {
  if (config.strategy === "cross-encoder") {
    return Promise.resolve(createCrossEncoderStrategy(config));
  }

  throw new UnsupportedStrategyError(config.strategy);
}
