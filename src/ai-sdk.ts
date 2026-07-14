import type { RerankingModel } from "ai";

import { RerankerDisposedError } from "./errors.js";
import { Reranker } from "./reranker.js";
import type { RerankerConfig, RerankerCreateOptions } from "./types.js";

export type AISDKRerankingModel = Extract<RerankingModel, { readonly specificationVersion: "v4" }> &
  AsyncDisposable & {
    dispose(): Promise<void>;
  };
export type AISDKRerankingModelCallOptions = Parameters<AISDKRerankingModel["doRerank"]>[0];
export type AISDKJSONObject = Extract<
  AISDKRerankingModelCallOptions["documents"],
  { type: "object" }
>["values"][number];

export type AISDKRerankingModelOptions = RerankerCreateOptions & {
  provider?: string;
  documentText?: (document: AISDKJSONObject) => string;
};

export function createAISDKRerankingModel(
  config: RerankerConfig,
  options: AISDKRerankingModelOptions = {},
): AISDKRerankingModel {
  const { documentText = defaultDocumentText, provider = "rerankers", ...createOptions } = options;
  const modelId = config.model;
  let rerankerPromise: Promise<Reranker> | undefined;
  let disposed = false;

  const dispose = async (): Promise<void> => {
    disposed = true;

    const currentRerankerPromise = rerankerPromise;
    if (currentRerankerPromise === undefined) {
      return;
    }

    try {
      const reranker = await currentRerankerPromise;
      await reranker.dispose();
    } finally {
      if (rerankerPromise === currentRerankerPromise) {
        rerankerPromise = undefined;
      }
    }
  };

  return {
    specificationVersion: "v4",
    provider,
    modelId,
    dispose,
    [Symbol.asyncDispose]: dispose,
    async doRerank({ documents, query, topN, abortSignal }) {
      if (disposed) {
        throw new RerankerDisposedError();
      }

      abortSignal?.throwIfAborted();

      rerankerPromise ??= Reranker.create(config, createOptions);

      let reranker: Reranker;
      try {
        reranker = await rerankerPromise;
      } catch (error) {
        rerankerPromise = undefined;
        throw error;
      }
      abortSignal?.throwIfAborted();

      const rankOptions = topN === undefined ? undefined : { topK: topN };
      const results =
        documents.type === "text"
          ? await reranker.rank(query, documents.values, rankOptions)
          : await reranker.rank(
              query,
              documents.values.map((document) => ({
                text: documentText(document),
                metadata: document,
              })),
              rankOptions,
            );
      abortSignal?.throwIfAborted();

      return {
        ranking: results.map(({ index, score }) => ({
          index,
          relevanceScore: score,
        })),
        response: {
          modelId,
        },
      };
    },
  };
}

export const rerankers = {
  rerankingModel: createAISDKRerankingModel,
} as const;

function defaultDocumentText(document: AISDKJSONObject): string {
  if (typeof document.text === "string") {
    return document.text;
  }

  return JSON.stringify(document);
}
