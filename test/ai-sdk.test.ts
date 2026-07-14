import { rerank, type RerankingModel } from "ai";
import { describe, expect, it, vi } from "vitest";

import { createAISDKRerankingModel, rerankers, type AISDKRerankingModel } from "../src/ai-sdk.js";
import {
  RerankerDisposedError,
  type NormalizedDocument,
  type RerankDocument,
  type RerankResult,
  type ScoringStrategy,
} from "../src/index.js";

type FakeStrategy = ScoringStrategy & {
  callCount: () => number;
  disposeCount: () => number;
  seenDocuments: () => Array<NormalizedDocument<RerankDocument>>;
};

function strategyFromScores(scores: number[]): FakeStrategy {
  let calls = 0;
  let disposals = 0;
  let seenDocuments: Array<NormalizedDocument<RerankDocument>> = [];

  const score: ScoringStrategy["score"] = <TDocument extends RerankDocument>(
    _query: string,
    documents: Array<NormalizedDocument<TDocument>>,
  ): Promise<Array<RerankResult<TDocument>>> => {
    calls += 1;
    seenDocuments = documents;
    return Promise.resolve(
      documents.map(({ document, index }) => ({
        document,
        index,
        score: scores[index] ?? 0,
      })),
    );
  };

  return {
    score,
    callCount: () => calls,
    dispose: () => {
      disposals += 1;
      return Promise.resolve();
    },
    disposeCount: () => disposals,
    seenDocuments: () => seenDocuments,
  };
}

function requireString(value: unknown): string {
  if (typeof value !== "string") {
    throw new TypeError("Expected a string field.");
  }

  return value;
}

function deferred<T>(): {
  promise: Promise<T>;
  resolve: (value: T) => void;
  reject: (reason?: unknown) => void;
} {
  let resolve: (value: T) => void = () => undefined;
  let reject: (reason?: unknown) => void = () => undefined;
  const promise = new Promise<T>((promiseResolve, promiseReject) => {
    resolve = promiseResolve;
    reject = promiseReject;
  });

  return { promise, resolve, reject };
}

describe("AI SDK adapter", () => {
  it("works with the AI SDK rerank helper", async () => {
    const model = createAISDKRerankingModel(
      { model: "mixedbread-ai/mxbai-rerank-base-v1" },
      {
        strategyFactory: () => Promise.resolve(strategyFromScores([0.2, 0.9, 0.4])),
      },
    );
    const compatibleModel: RerankingModel = model;

    const result = await rerank({
      model: compatibleModel,
      query: "red planet",
      documents: ["Venus", "Mars", "Jupiter"],
      topN: 2,
      maxRetries: 0,
    });

    expect(result.ranking).toEqual([
      { originalIndex: 1, score: 0.9, document: "Mars" },
      { originalIndex: 2, score: 0.4, document: "Jupiter" },
    ]);
    expect(result.rerankedDocuments).toEqual(["Mars", "Jupiter"]);
  });

  it("uses a document text extractor for AI SDK object documents", async () => {
    const strategy = strategyFromScores([0.1, 0.8]);
    const model = rerankers.rerankingModel(
      { model: "Xenova/bge-reranker-base" },
      {
        documentText: (document) =>
          `${requireString(document.title)}: ${requireString(document.body)}`,
        strategyFactory: () => Promise.resolve(strategy),
      },
    );

    await expect(
      model.doRerank({
        query: "red planet",
        documents: {
          type: "object",
          values: [
            { title: "Venus", body: "Hot planet." },
            { title: "Mars", body: "Red planet." },
          ],
        },
      }),
    ).resolves.toEqual({
      ranking: [
        { index: 1, relevanceScore: 0.8 },
        { index: 0, relevanceScore: 0.1 },
      ],
      response: {
        modelId: "Xenova/bge-reranker-base",
      },
    });

    expect(strategy.seenDocuments().map(({ text }) => text)).toEqual([
      "Venus: Hot planet.",
      "Mars: Red planet.",
    ]);
  });

  it("does not create a reranker when the call is already aborted", async () => {
    const strategy = strategyFromScores([1]);
    const abortController = new AbortController();
    abortController.abort(new Error("cancelled"));
    const model = createAISDKRerankingModel(
      { model: "Xenova/ms-marco-MiniLM-L-6-v2" },
      {
        strategyFactory: () => Promise.resolve(strategy),
      },
    );

    await expect(
      model.doRerank({
        query: "query",
        documents: { type: "text", values: ["document"] },
        abortSignal: abortController.signal,
      }),
    ).rejects.toThrow("cancelled");
    expect(strategy.callCount()).toBe(0);
  });

  it("shares the in-flight reranker load across concurrent first calls", async () => {
    const strategy = strategyFromScores([1, 0.5]);
    const load = deferred<ScoringStrategy>();
    let loadCount = 0;
    const model = createAISDKRerankingModel(
      { model: "Xenova/ms-marco-MiniLM-L-6-v2" },
      {
        strategyFactory: () => {
          loadCount += 1;
          return load.promise;
        },
      },
    );

    const first = model.doRerank({
      query: "query",
      documents: { type: "text", values: ["first", "second"] },
    });
    const second = model.doRerank({
      query: "query",
      documents: { type: "text", values: ["first", "second"] },
    });

    expect(loadCount).toBe(1);
    load.resolve(strategy);

    await expect(Promise.all([first, second])).resolves.toEqual([
      {
        ranking: [
          { index: 0, relevanceScore: 1 },
          { index: 1, relevanceScore: 0.5 },
        ],
        response: {
          modelId: "Xenova/ms-marco-MiniLM-L-6-v2",
        },
      },
      {
        ranking: [
          { index: 0, relevanceScore: 1 },
          { index: 1, relevanceScore: 0.5 },
        ],
        response: {
          modelId: "Xenova/ms-marco-MiniLM-L-6-v2",
        },
      },
    ]);
  });

  it("disposes a loaded reranker once through dispose and Symbol.asyncDispose", async () => {
    const strategy = strategyFromScores([1]);
    const model = createAISDKRerankingModel(
      { model: "Xenova/ms-marco-MiniLM-L-6-v2" },
      { strategyFactory: () => Promise.resolve(strategy) },
    );
    await model.doRerank({
      query: "query",
      documents: { type: "text", values: ["document"] },
    });

    await model.dispose();
    await model.dispose();
    await model[Symbol.asyncDispose]();

    expect(strategy.disposeCount()).toBe(1);
  });

  it("does not create a reranker when disposed before first use", async () => {
    const strategyFactory = vi.fn(() => Promise.resolve(strategyFromScores([1])));
    const model = createAISDKRerankingModel(
      { model: "Xenova/ms-marco-MiniLM-L-6-v2" },
      { strategyFactory },
    );

    await model.dispose();

    expect(strategyFactory).not.toHaveBeenCalled();
    await expect(
      model.doRerank({
        query: "query",
        documents: { type: "text", values: ["document"] },
      }),
    ).rejects.toBeInstanceOf(RerankerDisposedError);
  });

  it("waits for an active rerank before disposing", async () => {
    const scoringStarted = deferred<void>();
    const finishScoring = deferred<void>();
    const dispose = vi.fn(() => Promise.resolve());
    const score: ScoringStrategy["score"] = async <TDocument extends RerankDocument>(
      _query: string,
      documents: Array<NormalizedDocument<TDocument>>,
    ): Promise<Array<RerankResult<TDocument>>> => {
      scoringStarted.resolve();
      await finishScoring.promise;
      return documents.map(({ document, index }) => ({ document, index, score: 1 }));
    };
    const model = createAISDKRerankingModel(
      { model: "Xenova/ms-marco-MiniLM-L-6-v2" },
      { strategyFactory: () => Promise.resolve({ score, dispose }) },
    );
    const rerankPromise = model.doRerank({
      query: "query",
      documents: { type: "text", values: ["document"] },
    });
    await scoringStarted.promise;

    const disposePromise = model.dispose();
    await Promise.resolve();

    expect(dispose).not.toHaveBeenCalled();
    finishScoring.resolve();
    await rerankPromise;
    await disposePromise;
    expect(dispose).toHaveBeenCalledOnce();
  });
});

const aiSdkModel: RerankingModel = createAISDKRerankingModel({
  model: "mixedbread-ai/mxbai-rerank-base-v1",
});
const localModel: AISDKRerankingModel = rerankers.rerankingModel({
  model: "Xenova/bge-reranker-base",
});

void aiSdkModel;
void localModel;
