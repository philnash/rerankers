import type { RerankingModel } from "ai";

import { Reranker } from "./reranker.js";
import type { RerankerConfig, RerankerCreateOptions } from "./types.js";

export type AISDKRerankingModel = Extract<RerankingModel, { readonly specificationVersion: "v4" }>;
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

  return {
    specificationVersion: "v4",
    provider,
    modelId,
    async doRerank({ documents, query, topN, abortSignal }) {
      throwIfAborted(abortSignal);

      rerankerPromise ??= Reranker.create(config, createOptions);

      let reranker: Reranker;
      try {
        reranker = await rerankerPromise;
      } catch (error) {
        rerankerPromise = undefined;
        throw error;
      }
      throwIfAborted(abortSignal);

      const results =
        documents.type === "text"
          ? await reranker.rank(query, documents.values, toRankOptions(topN))
          : await reranker.rank(
              query,
              documents.values.map((document) => ({
                text: documentText(document),
                metadata: document,
              })),
              toRankOptions(topN),
            );
      throwIfAborted(abortSignal);

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

function toRankOptions(topN: number | undefined): { topK: number } | undefined {
  return topN === undefined ? undefined : { topK: topN };
}

function defaultDocumentText(document: AISDKJSONObject): string {
  if (typeof document.text === "string") {
    return document.text;
  }

  return JSON.stringify(document);
}

function throwIfAborted(abortSignal: AbortSignal | undefined): void {
  if (!abortSignal?.aborted) {
    return;
  }

  abortSignal.throwIfAborted();
}
