# rerankers

Browser-compatible TypeScript reranking over open Hugging Face models through Transformers.js.

```ts
import { Reranker } from "rerankers";

const reranker = await Reranker.create("mixedbread-base");
const results = await reranker.rank("Who wrote To Kill a Mockingbird?", documents, { topK: 5 });
```

## Install

```sh
npm install rerankers
```

This package uses `@huggingface/transformers`. Models are downloaded and cached by Transformers.js.

## Presets

| Preset              | Model                                   | Strategy                       |
| ------------------- | --------------------------------------- | ------------------------------ |
| `bge`               | `Xenova/bge-reranker-base`              | cross-encoder                  |
| `minilm`            | `Xenova/ms-marco-MiniLM-L-6-v2`         | cross-encoder                  |
| `mixedbread-xsmall` | `mixedbread-ai/mxbai-rerank-xsmall-v1`  | cross-encoder                  |
| `mixedbread-base`   | `mixedbread-ai/mxbai-rerank-base-v1`    | cross-encoder                  |
| `mixedbread-large`  | `mixedbread-ai/mxbai-rerank-large-v1`   | cross-encoder                  |
| `colbert-small`     | `answerdotai/answerai-colbert-small-v1` | late-interaction, experimental |

## Custom Models

```ts
const reranker = await Reranker.create({
  model: "mixedbread-ai/mxbai-rerank-large-v1",
  task: "text-ranking",
  strategy: "cross-encoder",
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
  model: "mixedbread-base",
  topK: 5,
});

const documents = await reranker.compressDocuments(retrievedDocuments, "red planet");
```

Returned LangChain documents are the original document objects in reranked order. Each returned document receives `metadata.relevanceScore`.

Pass core creation options through `createOptions`:

```ts
const reranker = new LocalReranker({
  model: "mixedbread-base",
  createOptions: {
    strategyFactory,
  },
  topK: 5,
});
```

For advanced lifecycle control, pass an already-created reranker:

```ts
import { Reranker } from "rerankers";

const coreReranker = await Reranker.create("mixedbread-base");
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
  model: rerankers.rerankingModel("mixedbread-base"),
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
const model = rerankers.rerankingModel("bge", {
  documentText: (document) => `${document.title}: ${document.body}`,
});
```

## Late Interaction

ColBERT-style models score token vectors with late interaction instead of returning one cross-encoder score. The library includes the MaxSim scoring primitive and an experimental late-interaction strategy boundary, but model-specific vector extraction depends on what the selected Transformers.js-compatible model exposes.

For production ColBERT use, provide a custom `strategyFactory` until the chosen model's token-vector path is verified.
