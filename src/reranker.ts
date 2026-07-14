import { RerankerDisposedError, RerankerInputError, UnsupportedStrategyError } from "./errors.js";
import { CrossEncoderStrategy } from "./strategies/cross-encoder.js";
import type {
  NormalizedDocument,
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

  private constructor(private readonly strategy: ScoringStrategy) {}

  static async create(
    config: RerankerConfig,
    options: RerankerCreateOptions = {},
  ): Promise<Reranker> {
    const normalizedConfig: NormalizedRerankerConfig = {
      model: config.model,
      strategy: config.strategy ?? "cross-encoder",
      transformerOptions: {
        dtype: "auto",
        ...config.transformerOptions,
      },
    };
    const strategy = await (options.strategyFactory ?? createDefaultStrategy)(normalizedConfig);

    return new Reranker(strategy);
  }

  async rank<TDocument extends RerankDocument>(
    query: string,
    documents: readonly TDocument[],
    options?: RankOptions,
  ): Promise<Array<RerankResult<TDocument>>> {
    if (this.disposePromise !== undefined) {
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

    const topK = normalizeTopK(options?.topK, normalized.length);
    const results = await this.strategy.score(query, normalized);
    return [...results].sort((left, right) => right.score - left.score).slice(0, topK);
  }

  private async disposeStrategy(): Promise<void> {
    await Promise.allSettled([...this.activeRanks]);
    await this.strategy.dispose?.();
  }
}

function normalizeDocuments<TDocument extends RerankDocument>(
  documents: readonly TDocument[],
): Array<NormalizedDocument<TDocument>> {
  return documents.map((document, index) => {
    if (typeof document === "string") {
      return { document, index, text: document };
    }

    if (!document || typeof document.text !== "string") {
      throw new RerankerInputError(
        "Each document must be a string or an object with a text field.",
      );
    }

    return { document, index, text: document.text };
  });
}

function normalizeTopK(topK: number | undefined, documentCount: number): number {
  if (topK === undefined) {
    return documentCount;
  }

  if (!Number.isInteger(topK) || topK < 1) {
    throw new RerankerInputError("rank option topK must be a positive integer.");
  }

  return Math.min(topK, documentCount);
}

function createDefaultStrategy(config: NormalizedRerankerConfig): ScoringStrategy {
  if (config.strategy !== "cross-encoder") {
    throw new UnsupportedStrategyError(config.strategy);
  }

  return new CrossEncoderStrategy(config);
}
