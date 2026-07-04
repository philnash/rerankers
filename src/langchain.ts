import type { DocumentInterface } from "@langchain/core/documents";
import { BaseDocumentCompressor } from "@langchain/core/retrievers/document_compressors";

import { Reranker } from "./reranker.js";
import type {
  RankOptions,
  RerankerCreateOptions,
  RerankerDefinition,
  RerankDocument,
  RerankResult,
} from "./types.js";

export type LocalRerankerCore = {
  rank<TDocument extends RerankDocument>(
    query: string,
    documents: readonly TDocument[],
    options?: RankOptions,
  ): Promise<Array<RerankResult<TDocument>>>;
};

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
      model?: RerankerDefinition;
      createOptions?: RerankerCreateOptions;
      reranker?: never;
      topK?: number;
    }
  | {
      reranker: LocalRerankerCore;
      model?: never;
      createOptions?: never;
      topK?: number;
    };

export class LocalReranker extends BaseDocumentCompressor {
  private readonly reranker: Promise<LocalRerankerCore>;
  private readonly topK: number | undefined;

  constructor(fields: LocalRerankerArgs = {}) {
    super();

    if (fields.reranker && (fields.model !== undefined || fields.createOptions !== undefined)) {
      throw new Error("LocalReranker accepts either model/createOptions or reranker, not both.");
    }

    this.topK = fields.topK;
    this.reranker = fields.reranker
      ? Promise.resolve(fields.reranker)
      : Reranker.create(fields.model, fields.createOptions);
  }

  async compressDocuments(
    documents: DocumentInterface[],
    query: string,
  ): Promise<DocumentInterface[]> {
    if (documents.length === 0) {
      return [];
    }

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
}

function getPageContent(document: LocalRerankerInput): string {
  if (typeof document === "string") {
    return document;
  }

  return document.pageContent;
}
