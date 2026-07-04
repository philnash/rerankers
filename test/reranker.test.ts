import { describe, expect, it } from "vitest";

import {
  Reranker,
  RerankerInputError,
  UnknownPresetError,
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
  it("creates a mixedbread preset and returns the topK results in score order", async () => {
    const strategy = strategyFromScores([0.2, 0.9, 0.4]);
    const reranker = await Reranker.create("mixedbread-base", {
      strategyFactory: (config) => {
        expect(config).toMatchObject({
          model: "mixedbread-ai/mxbai-rerank-base-v1",
          task: "text-ranking",
          strategy: "cross-encoder",
        });
        return Promise.resolve(strategy);
      },
    });

    const results = await reranker.rank("red planet", ["Venus", "Mars", "Jupiter"], { topK: 2 });

    expect(results).toEqual([
      { document: "Mars", index: 1, score: 0.9 },
      { document: "Jupiter", index: 2, score: 0.4 },
    ]);
  });

  it("preserves object documents and metadata", async () => {
    const strategy = strategyFromScores([0.1, 0.8]);
    const reranker = await Reranker.create("bge", {
      strategyFactory: () => Promise.resolve(strategy),
    });
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
    const reranker = await Reranker.create("minilm", {
      strategyFactory: () => Promise.resolve(strategy),
    });

    await expect(reranker.rank("query", [])).resolves.toEqual([]);
    await expect(reranker.rank("query", [], { topK: -1 })).resolves.toEqual([]);
    expect(strategy.callCount()).toBe(0);
  });

  it("supports a custom model definition", async () => {
    const customConfig: RerankerConfig = {
      model: "mixedbread-ai/mxbai-rerank-large-v1",
      strategy: "cross-encoder",
      task: "text-ranking",
      transformerOptions: {
        device: "wasm",
        dtype: "q8",
      },
    };
    const reranker = await Reranker.create(customConfig, {
      strategyFactory: (config) => {
        expect(config).toEqual(customConfig);
        return Promise.resolve(strategyFromScores([1]));
      },
    });

    await expect(reranker.rank("query", ["doc"])).resolves.toEqual([
      { document: "doc", index: 0, score: 1 },
    ]);
  });

  it("throws a clear error for an unknown preset", async () => {
    await expect(Reranker.create("not-a-model")).rejects.toBeInstanceOf(UnknownPresetError);
  });

  it("throws when topK is not a positive integer", async () => {
    const reranker = await Reranker.create("bge", {
      strategyFactory: () => Promise.resolve(strategyFromScores([1])),
    });

    await expect(reranker.rank("query", ["doc"], { topK: 0 })).rejects.toBeInstanceOf(
      RerankerInputError,
    );
  });
});
