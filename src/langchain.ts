import type { DocumentInterface } from "@langchain/core/documents";
import { BaseDocumentCompressor } from "@langchain/core/retrievers/document_compressors";

import { Reranker } from "./reranker.js";
import type {
  RerankerConfig,
  RerankerStrategyName,
  StrategyFactory,
  TransformerOptions,
} from "./types.js";

export type LocalRerankerInput = DocumentInterface | string | { pageContent: string };

export type LocalRerankerResult = {
  index: number;
  relevanceScore: number;
};

export type LocalRerankerRerankOptions = {
  topK?: number;
};

export type LocalRerankerArgs =
  | {
      model: string;
      strategy?: RerankerStrategyName;
      transformerOptions?: TransformerOptions;
      strategyFactory?: StrategyFactory;
      reranker?: never;
      topK?: number;
    }
  | {
      reranker: Reranker;
      model?: never;
      strategy?: never;
      transformerOptions?: never;
      strategyFactory?: never;
      topK?: number;
    };

export class LocalReranker extends BaseDocumentCompressor implements AsyncDisposable {
  private readonly reranker: Promise<Reranker>;
  private readonly topK: number | undefined;

  constructor(fields: LocalRerankerArgs) {
    super();

    if (
      fields.reranker &&
      (fields.model !== undefined ||
        fields.strategy !== undefined ||
        fields.transformerOptions !== undefined ||
        fields.strategyFactory !== undefined)
    ) {
      throw new Error(
        "LocalReranker accepts either model options/strategyFactory or reranker, not both.",
      );
    }

    this.topK = fields.topK;
    if (fields.reranker) {
      this.reranker = Promise.resolve(fields.reranker);
      return;
    }

    const createOptions =
      fields.strategyFactory === undefined
        ? undefined
        : { strategyFactory: fields.strategyFactory };
    this.reranker = Reranker.create(toRerankerConfig(fields), createOptions);
  }

  async compressDocuments(
    documents: DocumentInterface[],
    query: string,
  ): Promise<DocumentInterface[]> {
    const results = await this.rerank(documents, query);

    return results.map(({ index, relevanceScore }) => {
      const document = documents[index];
      if (!document) {
        throw new Error(`Reranker returned an out-of-range document index: ${index}.`);
      }
      document.metadata.relevanceScore = relevanceScore;
      return document;
    });
  }

  async rerank<TInput extends LocalRerankerInput>(
    documents: readonly TInput[],
    query: string,
    options: LocalRerankerRerankOptions = {},
  ): Promise<LocalRerankerResult[]> {
    if (documents.length === 0) {
      return [];
    }

    const reranker = await this.reranker;
    const topK = options.topK ?? this.topK;
    const rerankDocuments = documents.map((document) => ({
      text: getPageContent(document),
      metadata: document,
    }));
    const rankOptions = topK === undefined ? undefined : { topK };
    const results = await reranker.rank(query, rerankDocuments, rankOptions);

    return results.map(({ index, score }) => ({ index, relevanceScore: score }));
  }

  async dispose(): Promise<void> {
    const reranker = await this.reranker;
    await reranker.dispose();
  }

  [Symbol.asyncDispose](): Promise<void> {
    return this.dispose();
  }
}

function toRerankerConfig(fields: Extract<LocalRerankerArgs, { model: string }>): RerankerConfig {
  const config: RerankerConfig = { model: fields.model };

  if (fields.strategy !== undefined) {
    config.strategy = fields.strategy;
  }

  if (fields.transformerOptions !== undefined) {
    config.transformerOptions = fields.transformerOptions;
  }

  return config;
}

function getPageContent(document: LocalRerankerInput): string {
  if (typeof document === "string") {
    return document;
  }

  return document.pageContent;
}
