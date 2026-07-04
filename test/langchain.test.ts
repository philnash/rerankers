import { Document, type DocumentInterface } from "@langchain/core/documents";
import { describe, expect, it } from "vitest";

import {
  LocalReranker,
  type LocalRerankerCore,
  type LocalRerankerInput,
} from "../src/langchain.js";
import type { RankOptions, RerankDocument, RerankResult } from "../src/index.js";

type RankCall = {
  query: string;
  documents: readonly RerankDocument[];
  options: RankOptions | undefined;
};

function fakeReranker(scores: number[]) {
  const calls: RankCall[] = [];
  const reranker: LocalRerankerCore = {
    rank<TDocument extends RerankDocument>(
      query: string,
      documents: readonly TDocument[],
      options?: RankOptions,
    ): Promise<Array<RerankResult<TDocument>>> {
      calls.push({ query, documents, options });

      const ranked = documents.map((document, index) => ({
        document,
        index,
        score: scores[index] ?? 0,
      }));

      return Promise.resolve(
        ranked.sort((left, right) => right.score - left.score).slice(0, options?.topK),
      );
    },
  };

  return { calls, reranker };
}

describe("LocalReranker", () => {
  it("compresses LangChain documents in relevance order and writes relevanceScore metadata", async () => {
    const { calls, reranker } = fakeReranker([0.2, 0.9, 0.4]);
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
    expect(calls[0]).toMatchObject({
      query: "red planet",
      options: { topK: 2 },
    });
    expect(calls[0]?.documents).toEqual([
      { text: "Venus is hot.", metadata: documents[0] },
      { text: "Mars is red.", metadata: documents[1] },
      { text: "Jupiter is large.", metadata: documents[2] },
    ]);
  });

  it("returns raw rerank results and lets method topK override constructor topK", async () => {
    const { calls, reranker } = fakeReranker([0.2, 0.9, 0.4]);
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
    expect(calls[0]?.options).toEqual({ topK: 3 });
  });

  it("supports string and pageContent object inputs in rerank", async () => {
    const { calls, reranker } = fakeReranker([0.1, 0.7]);
    const local = new LocalReranker({ reranker });
    const documents: LocalRerankerInput[] = ["plain text", { pageContent: "object text" }];

    await expect(local.rerank(documents, "query")).resolves.toEqual([
      { index: 1, relevanceScore: 0.7 },
      { index: 0, relevanceScore: 0.1 },
    ]);
    expect(calls[0]?.documents).toEqual([
      { text: "plain text", metadata: "plain text" },
      { text: "object text", metadata: documents[1] },
    ]);
  });

  it("returns an empty list without calling the core reranker", async () => {
    const { calls, reranker } = fakeReranker([]);
    const local = new LocalReranker({ reranker, topK: 2 });

    await expect(local.compressDocuments([], "query")).resolves.toEqual([]);
    await expect(local.rerank([], "query", { topK: 1 })).resolves.toEqual([]);
    expect(calls).toEqual([]);
  });

  it("rejects JavaScript callers that provide both model and reranker", () => {
    const { reranker } = fakeReranker([]);

    expect(
      () =>
        new LocalReranker({
          model: "mixedbread-base",
          reranker,
        } as unknown as ConstructorParameters<typeof LocalReranker>[0]),
    ).toThrow("LocalReranker accepts either model/createOptions or reranker, not both.");
  });

  it("accepts DocumentInterface inputs without requiring concrete Document instances", async () => {
    const { reranker } = fakeReranker([1]);
    const local = new LocalReranker({ reranker });
    const document: DocumentInterface = {
      pageContent: "interface document",
      metadata: {},
    };

    const compressed = await local.compressDocuments([document], "query");

    expect(compressed).toEqual([document]);
    expect(document.metadata.relevanceScore).toBe(1);
  });
});
