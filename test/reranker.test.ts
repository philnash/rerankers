import { describe, expect, it, vi } from "vitest";

import {
  Reranker,
  RerankerDisposedError,
  RerankerInputError,
  UnsupportedStrategyError,
  type NormalizedDocument,
  type RerankerConfig,
  type RerankDocument,
  type RerankResult,
  type ScoringStrategy,
} from "../src/index.js";

type FakeStrategy = ScoringStrategy & {
  callCount: () => number;
};

function strategyFromScores(scores: number[]): FakeStrategy {
  let calls = 0;
  const score: ScoringStrategy["score"] = <TDocument extends RerankDocument>(
    _query: string,
    documents: Array<NormalizedDocument<TDocument>>,
  ): Promise<Array<RerankResult<TDocument>>> => {
    calls += 1;
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
  };
}

describe("Reranker", () => {
  it("creates a cross-encoder reranker and returns the topK results in score order", async () => {
    const strategy = strategyFromScores([0.2, 0.9, 0.4]);
    const reranker = await Reranker.create(
      { model: "mixedbread-ai/mxbai-rerank-base-v1" },
      {
        strategyFactory: (config) => {
          expect(config).toEqual({
            model: "mixedbread-ai/mxbai-rerank-base-v1",
            strategy: "cross-encoder",
            transformerOptions: {
              dtype: "auto",
            },
          });
          return Promise.resolve(strategy);
        },
      },
    );

    const results = await reranker.rank("red planet", ["Venus", "Mars", "Jupiter"], { topK: 2 });

    expect(results).toEqual([
      { document: "Mars", index: 1, score: 0.9 },
      { document: "Jupiter", index: 2, score: 0.4 },
    ]);
  });

  it("preserves object documents and metadata", async () => {
    const strategy = strategyFromScores([0.1, 0.8]);
    const reranker = await Reranker.create(
      { model: "Xenova/bge-reranker-base" },
      {
        strategyFactory: () => Promise.resolve(strategy),
      },
    );
    const documents = [
      { id: "a", text: "first", metadata: { source: "alpha" } },
      { id: "b", text: "second", metadata: { source: "beta" } },
    ];

    const results = await reranker.rank("query", documents);

    expect(results[0]).toEqual({
      document: documents[1],
      index: 1,
      score: 0.8,
    });
  });

  it("returns an empty list without scoring when documents are empty", async () => {
    const strategy = strategyFromScores([]);
    const reranker = await Reranker.create(
      { model: "Xenova/ms-marco-MiniLM-L-6-v2" },
      {
        strategyFactory: () => Promise.resolve(strategy),
      },
    );

    await expect(reranker.rank("query", [])).resolves.toEqual([]);
    await expect(reranker.rank("query", [], { topK: -1 })).resolves.toEqual([]);
    expect(strategy.callCount()).toBe(0);
  });

  it("supports a custom model definition", async () => {
    const customConfig: RerankerConfig = {
      model: "mixedbread-ai/mxbai-rerank-large-v1",
      strategy: "cross-encoder",
      transformerOptions: {
        device: "wasm",
        dtype: "q8",
      },
    };
    const reranker = await Reranker.create(customConfig, {
      strategyFactory: (config) => {
        expect(config).toEqual({
          ...customConfig,
          strategy: "cross-encoder",
        });
        return Promise.resolve(strategyFromScores([1]));
      },
    });

    await expect(reranker.rank("query", ["doc"])).resolves.toEqual([
      { document: "doc", index: 0, score: 1 },
    ]);
  });

  it("passes explicit custom strategy names to a custom strategy factory", async () => {
    const customConfig: RerankerConfig = {
      model: "example/custom-reranker",
      strategy: "custom-strategy",
    };
    const reranker = await Reranker.create(customConfig, {
      strategyFactory: (config) => {
        expect(config).toEqual({
          ...customConfig,
          transformerOptions: {
            dtype: "auto",
          },
        });
        return Promise.resolve(strategyFromScores([1]));
      },
    });

    await expect(reranker.rank("query", ["doc"])).resolves.toEqual([
      { document: "doc", index: 0, score: 1 },
    ]);
  });

  it("throws a clear error when the default factory receives an unsupported strategy", async () => {
    const customConfig: RerankerConfig = {
      model: "example/custom-reranker",
      strategy: "custom-strategy",
    };

    await expect(Reranker.create(customConfig)).rejects.toBeInstanceOf(UnsupportedStrategyError);
  });

  it("throws when topK is not a positive integer", async () => {
    const reranker = await Reranker.create(
      { model: "Xenova/bge-reranker-base" },
      {
        strategyFactory: () => Promise.resolve(strategyFromScores([1])),
      },
    );

    await expect(reranker.rank("query", ["doc"], { topK: 0 })).rejects.toBeInstanceOf(
      RerankerInputError,
    );
  });

  it("disposes its strategy once and rejects subsequent ranks", async () => {
    const strategy = strategyFromScores([1]);
    const dispose = vi.fn(() => Promise.resolve());
    strategy.dispose = dispose;
    const reranker = await Reranker.create(
      { model: "Xenova/bge-reranker-base" },
      { strategyFactory: () => Promise.resolve(strategy) },
    );

    await reranker.dispose();
    await reranker.dispose();
    await reranker[Symbol.asyncDispose]();

    expect(dispose).toHaveBeenCalledOnce();
    await expect(reranker.rank("query", ["doc"])).rejects.toBeInstanceOf(RerankerDisposedError);
    expect(strategy.callCount()).toBe(0);
  });

  it("waits for an active rank before disposing its strategy", async () => {
    let finishScoring: (() => void) | undefined;
    const scoringFinished = new Promise<void>((resolve) => {
      finishScoring = resolve;
    });
    const score: ScoringStrategy["score"] = async <TDocument extends RerankDocument>(
      _query: string,
      documents: Array<NormalizedDocument<TDocument>>,
    ): Promise<Array<RerankResult<TDocument>>> => {
      await scoringFinished;
      return documents.map(({ document, index }) => ({ document, index, score: 1 }));
    };
    const dispose = vi.fn(() => Promise.resolve());
    const reranker = await Reranker.create(
      { model: "Xenova/bge-reranker-base" },
      { strategyFactory: () => Promise.resolve({ score, dispose }) },
    );

    const rankPromise = reranker.rank("query", ["doc"]);
    const disposePromise = reranker.dispose();
    await Promise.resolve();

    expect(dispose).not.toHaveBeenCalled();
    finishScoring?.();
    await rankPromise;
    await disposePromise;
    expect(dispose).toHaveBeenCalledOnce();
  });
});
