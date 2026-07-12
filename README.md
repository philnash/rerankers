# rerankers

Browser-compatible TypeScript reranking over open Hugging Face models through Transformers.js.

```ts
import { Reranker } from "rerankers";

const reranker = await Reranker.create({ model: "mixedbread-ai/mxbai-rerank-base-v1" });
const results = await reranker.rank("Who wrote To Kill a Mockingbird?", documents, { topK: 5 });
```

## Install

```sh
npm install rerankers
```

This package uses `@huggingface/transformers`. Models are downloaded and cached by Transformers.js.

## Models

Pass a Hugging Face model ID. The strategy defaults to `cross-encoder`.

```ts
const reranker = await Reranker.create({
  model: "mixedbread-ai/mxbai-rerank-large-v1",
  transformerOptions: {
    device: "wasm",
    dtype: "q8",
  },
});
```

## Documents

Documents can be strings or objects. Object documents are returned unchanged.

```ts
const results = await reranker.rank("red planet", [
  { id: "venus", text: "Venus is hot.", metadata: { source: "encyclopedia" } },
  { id: "mars", text: "Mars is called the Red Planet.", metadata: { source: "encyclopedia" } },
]);
```

Each result includes:

```ts
type RerankResult<TDocument> = {
  document: TDocument;
  index: number;
  score: number;
};
```

## LangChain

Install LangChain core alongside this package:

```sh
npm install rerankers @langchain/core
```

Use the `LocalReranker` adapter from the `rerankers/langchain` subpath anywhere LangChain expects a document compressor.

```ts
import { LocalReranker } from "rerankers/langchain";

const reranker = new LocalReranker({
  model: "mixedbread-ai/mxbai-rerank-base-v1",
  topK: 5,
});

const documents = await reranker.compressDocuments(retrievedDocuments, "red planet");
```

Returned LangChain documents are the original document objects in reranked order. Each returned document receives `metadata.relevanceScore`.

Pass core creation options through `createOptions`:

```ts
const reranker = new LocalReranker({
  model: "mixedbread-ai/mxbai-rerank-base-v1",
  createOptions: {
    strategyFactory,
  },
  topK: 5,
});
```

For advanced lifecycle control, pass an already-created reranker:

```ts
import { Reranker } from "rerankers";

const coreReranker = await Reranker.create({ model: "mixedbread-ai/mxbai-rerank-base-v1" });
const reranker = new LocalReranker({ reranker: coreReranker, topK: 5 });
```

Use `rerank()` when you want raw scores without mutating LangChain document metadata:

```ts
const results = await reranker.rerank(documents, "red planet", { topK: 5 });
// [{ index: 1, relevanceScore: 0.92 }, ...]
```

## Browser And Node

The package is ESM and keeps the runtime path compatible with browsers and Node.js. Use Transformers.js options such as `device` and `dtype` to tune runtime behavior.

## Vercel AI SDK

Install the AI SDK alongside this package, then use the `rerankers/ai-sdk` adapter anywhere AI SDK expects a reranking model.

```sh
npm install rerankers ai
```

```ts
import { rerank } from "ai";
import { rerankers } from "rerankers/ai-sdk";

const { ranking, rerankedDocuments } = await rerank({
  model: rerankers.rerankingModel({ model: "mixedbread-ai/mxbai-rerank-base-v1" }),
  query: "Which document mentions Mars?",
  documents: [
    "Venus has a thick atmosphere.",
    "Mars is called the Red Planet.",
    "Jupiter is the largest planet.",
  ],
  topN: 2,
});
```

AI SDK object documents are supported with a text extractor. If you do not provide one, the adapter uses a string `text` property when present and falls back to `JSON.stringify(document)`.

```ts
const model = rerankers.rerankingModel(
  { model: "Xenova/bge-reranker-base" },
  {
    documentText: (document) => `${document.title}: ${document.body}`,
  },
);
```

## Strategy Extension

The built-in strategy uses cross-encoder models. The `strategyFactory` option is the extension point for future reranking strategies or project-specific model integrations.
