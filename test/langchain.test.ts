import { Document, type DocumentInterface } from "@langchain/core/documents";
import { describe, expect, it, vi } from "vitest";

import { LocalReranker, type LocalRerankerInput } from "../src/langchain.js";
import { Reranker, RerankerDisposedError } from "../src/index.js";

async function fakeReranker(scores: number[]) {
  const dispose = vi.fn(() => Promise.resolve());
  const reranker = await Reranker.create(
    { model: "test/reranker" },
    {
      strategyFactory: () =>
        Promise.resolve({
          dispose,
          score: (_query, documents) =>
            Promise.resolve(
              documents.map(({ document, index }) => ({
                document,
                index,
                score: scores[index] ?? 0,
              })),
            ),
        }),
    },
  );
  const rank = vi.spyOn(reranker, "rank");

  return { dispose, rank, reranker };
}

describe("LocalReranker", () => {
  it("passes a top-level strategyFactory through when constructing the core reranker", async () => {
    const local = new LocalReranker({
      model: "mixedbread-ai/mxbai-rerank-base-v1",
      strategyFactory: (config) => {
        expect(config).toEqual({
          model: "mixedbread-ai/mxbai-rerank-base-v1",
          strategy: "cross-encoder",
          transformerOptions: {
            dtype: "auto",
          },
        });

        return Promise.resolve({
          score: (query, documents) => {
            expect(query).toBe("planet");
            return Promise.resolve(
              documents.map(({ document, index }) => ({
                document,
                index,
                score: index,
              })),
            );
          },
        });
      },
      topK: 1,
    });

    await expect(local.rerank(["Venus", "Mars"], "planet")).resolves.toEqual([
      { index: 1, relevanceScore: 1 },
    ]);
  });

  it("compresses LangChain documents in relevance order and writes relevanceScore metadata", async () => {
    const { rank, reranker } = await fakeReranker([0.2, 0.9, 0.4]);
    const local = new LocalReranker({ reranker, topK: 2 });
    const documents = [
      new Document({ pageContent: "Venus is hot.", metadata: { source: "a" } }),
      new Document({ pageContent: "Mars is red.", metadata: { source: "b" } }),
      new Document({ pageContent: "Jupiter is large.", metadata: { source: "c" } }),
    ];

    const compressed = await local.compressDocuments(documents, "red planet");

    expect(compressed).toEqual([documents[1], documents[2]]);
    expect(compressed[0]).toBe(documents[1]);
    expect(compressed[1]).toBe(documents[2]);
    expect(documents[1]?.metadata).toMatchObject({ source: "b", relevanceScore: 0.9 });
    expect(documents[2]?.metadata).toMatchObject({ source: "c", relevanceScore: 0.4 });
    expect(rank).toHaveBeenCalledWith("red planet", expect.any(Array), { topK: 2 });
    expect(rank.mock.calls[0]?.[1]).toEqual([
      { text: "Venus is hot.", metadata: documents[0] },
      { text: "Mars is red.", metadata: documents[1] },
      { text: "Jupiter is large.", metadata: documents[2] },
    ]);
  });

  it("returns raw rerank results and lets method topK override constructor topK", async () => {
    const { rank, reranker } = await fakeReranker([0.2, 0.9, 0.4]);
    const local = new LocalReranker({ reranker, topK: 1 });
    const documents = [
      new Document({ pageContent: "Venus" }),
      new Document({ pageContent: "Mars" }),
      new Document({ pageContent: "Jupiter" }),
    ];

    const results = await local.rerank(documents, "planet", { topK: 3 });

    expect(results).toEqual([
      { index: 1, relevanceScore: 0.9 },
      { index: 2, relevanceScore: 0.4 },
      { index: 0, relevanceScore: 0.2 },
    ]);
    expect(rank.mock.calls[0]?.[2]).toEqual({ topK: 3 });
  });

  it("supports string and pageContent object inputs in rerank", async () => {
    const { rank, reranker } = await fakeReranker([0.1, 0.7]);
    const local = new LocalReranker({ reranker });
    const documents: LocalRerankerInput[] = ["plain text", { pageContent: "object text" }];

    await expect(local.rerank(documents, "query")).resolves.toEqual([
      { index: 1, relevanceScore: 0.7 },
      { index: 0, relevanceScore: 0.1 },
    ]);
    expect(rank.mock.calls[0]?.[1]).toEqual([
      { text: "plain text", metadata: "plain text" },
      { text: "object text", metadata: documents[1] },
    ]);
  });

  it("returns an empty list without calling the core reranker", async () => {
    const { rank, reranker } = await fakeReranker([]);
    const local = new LocalReranker({ reranker, topK: 2 });

    await expect(local.compressDocuments([], "query")).resolves.toEqual([]);
    await expect(local.rerank([], "query", { topK: 1 })).resolves.toEqual([]);
    expect(rank).not.toHaveBeenCalled();
  });

  it("rejects JavaScript callers that provide both model and reranker", async () => {
    const { reranker } = await fakeReranker([]);

    expect(
      () =>
        new LocalReranker({
          model: "mixedbread-ai/mxbai-rerank-base-v1",
          reranker,
        } as unknown as ConstructorParameters<typeof LocalReranker>[0]),
    ).toThrow("LocalReranker accepts either model options/strategyFactory or reranker, not both.");
  });

  it("accepts DocumentInterface inputs without requiring concrete Document instances", async () => {
    const { reranker } = await fakeReranker([1]);
    const local = new LocalReranker({ reranker });
    const document: DocumentInterface = {
      pageContent: "interface document",
      metadata: {},
    };

    const compressed = await local.compressDocuments([document], "query");

    expect(compressed).toEqual([document]);
    expect(document.metadata.relevanceScore).toBe(1);
  });

  it("disposes a core reranker that it creates and rejects subsequent reranks", async () => {
    const dispose = vi.fn(() => Promise.resolve());
    const local = new LocalReranker({
      model: "mixedbread-ai/mxbai-rerank-base-v1",
      strategyFactory: () =>
        Promise.resolve({
          score: () => Promise.resolve([]),
          dispose,
        }),
    });

    await local.dispose();
    await local.dispose();
    await local[Symbol.asyncDispose]();

    expect(dispose).toHaveBeenCalledOnce();
    await expect(local.rerank(["document"], "query")).rejects.toBeInstanceOf(RerankerDisposedError);
  });

  it("disposes an injected core reranker", async () => {
    const { dispose, reranker } = await fakeReranker([1]);
    const local = new LocalReranker({ reranker });

    await local.dispose();

    expect(dispose).toHaveBeenCalledOnce();
  });
});
