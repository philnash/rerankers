<h1 style="text-align: center">🔀 rerankers 🔀</h1>

Run local reranking models directly in your JavaScript/TypeScript application.

- [Why rerankers?](#why-rerankers)
  - [Demo](#demo)
- [How it works](#how-it-works)
  - [Install](#install)
  - [Using the library](#using-the-library)
  - [Model config](#model-config)
  - [Models](#models)
  - [Documents](#documents)
  - [Releasing resources](#releasing-resources)
- [Ecosystem plugins](#ecosystem-plugins)
  - [LangChain](#langchain)
  - [Vercel AI SDK](#vercel-ai-sdk)
- [Browser And Node](#browser-and-node)
- [License](#license)

## Why rerankers?

In RAG and other retrieval workflows, returning accurate results is a trade off between search capabilities and speed. Embedding models and vector indexes are good at returning semantically similar results fast. Cross-encoding rerankers are much more accurate, but slower, because they compare the query with each document at ranking time.

This library uses [`@huggingface/transformers`](https://huggingface.co/docs/transformers.js/en/index) to make it easy to rerank documents against queries using local models.

### Demo

You can see this library in action in a [live reranking demo](https://philnash.github.io/rerankers/).

## How it works

### Install

Install the library from npm.

```sh
npm install rerankers
```

### Using the library

This package uses `@huggingface/transformers`. Models are downloaded on first use and cached by Transformers.js.

```ts
import { Reranker } from "rerankers";

const query = "How can I reduce the initial load time of a JavaScript web application?";

const documents = [
  {
    id: "doc-1",
    text: "Code splitting lets a JavaScript application load only the code needed for the current page. Dynamic imports and route-based chunks can significantly reduce the initial bundle size.",
  },
  {
    id: "doc-2",
    text: "JavaScript is a programming language commonly used to add interactive behaviour to websites and build server-side applications with Node.js.",
  },
  {
    id: "doc-3",
    text: "Compressing images, serving modern formats such as WebP or AVIF, and lazy-loading below-the-fold media can improve page load performance.",
  },
  {
    id: "doc-4",
    text: "Tree shaking removes unused exports from a production bundle. It works best with ES modules and can reduce the amount of JavaScript downloaded during the initial page load.",
  },
  {
    id: "doc-5",
    text: "A service worker can cache application assets after the first visit, making repeat visits faster and allowing some functionality to work offline.",
  },
];

const reranker = await Reranker.create({ model: "mixedbread-ai/mxbai-rerank-base-v1" });
const results = await reranker.rank(query, documents, { topK: 3 });

console.log(results);
// => [
//   {
//     document: {
//       id: "doc-1",
//       text: "Code splitting lets a JavaScript application load only the code needed for the current page. Dynamic imports and route-based chunks can significantly reduce the initial bundle size."
//     },
//     index: 0,
//     score: 0.9616439938545227
//   },
//   {
//     document: {
//       id: "doc-4",
//       text: "Tree shaking removes unused exports from a production bundle. It works best with ES modules and can reduce the amount of JavaScript downloaded during the initial page load."
//     },
//     index: 3,
//     score: 0.9472419619560242
//   },
//   {
//     document: {
//       id: "doc-5",
//       text: "A service worker can cache application assets after the first visit, making repeat visits faster and allowing some functionality to work offline."
//     },
//     index: 4,
//     score: 0.4266827702522278
//   }
// ]
```

### Model config

You can just pass a Hugging Face model ID and default options will be used to load the model. Specifically, the `dtype` will be set to `auto`.

```ts
const reranker = await Reranker.create({
  model: "mixedbread-ai/mxbai-rerank-large-v1",
});
```

If you want to pass more options, such as a specific dtype or device, you can pass a `transformerOptions` property like this:

```ts
const reranker = await Reranker.create({
  model: "mixedbread-ai/mxbai-rerank-large-v1",
  transformerOptions: {
    device: "wasm",
    dtype: "q8",
  },
});
```

For more on [dtypes](https://huggingface.co/docs/transformers.js/en/guides/dtypes) or [device options](https://huggingface.co/docs/transformers.js/en/guides/webgpu) check the [Transformers.js documentation](https://huggingface.co/docs/transformers.js/en/index).

### Models

These models have been tested and work with rerankers. Other reranking models with ONNX weights should also work.

| Model                                                                                                     | Parameters | Languages           | Context |
| --------------------------------------------------------------------------------------------------------- | ---------: | ------------------- | ------: |
| [`Xenova/ms-marco-TinyBERT-L-2-v2`](https://huggingface.co/Xenova/ms-marco-TinyBERT-L-2-v2)               |       4.4M | English             |     512 |
| [`Xenova/ms-marco-MiniLM-L-2-v2`](https://huggingface.co/Xenova/ms-marco-MiniLM-L-2-v2)                   |      15.6M | English             |     512 |
| [`Xenova/ms-marco-MiniLM-L-4-v2`](https://huggingface.co/Xenova/ms-marco-MiniLM-L-4-v2)                   |      19.2M | English             |     512 |
| [`Xenova/ms-marco-MiniLM-L-6-v2`](https://huggingface.co/Xenova/ms-marco-MiniLM-L-6-v2)                   |      22.7M | English             |     512 |
| [`Xenova/ms-marco-MiniLM-L-12-v2`](https://huggingface.co/Xenova/ms-marco-MiniLM-L-12-v2)                 |      33.4M | English             |     512 |
| [`jinaai/jina-reranker-v1-tiny-en`](https://huggingface.co/jinaai/jina-reranker-v1-tiny-en)               |      33.0M | English             |   8,192 |
| [`jinaai/jina-reranker-v1-turbo-en`](https://huggingface.co/jinaai/jina-reranker-v1-turbo-en)             |      37.8M | English             |   8,192 |
| [`mixedbread-ai/mxbai-rerank-xsmall-v1`](https://huggingface.co/mixedbread-ai/mxbai-rerank-xsmall-v1)     |      70.8M | English             |     512 |
| [`mixedbread-ai/mxbai-rerank-base-v1`](https://huggingface.co/mixedbread-ai/mxbai-rerank-base-v1)         |       184M | English             |     512 |
| [`Xenova/bge-reranker-base`](https://huggingface.co/Xenova/bge-reranker-base)                             |      ~278M | Chinese and English |     512 |
| [`mixedbread-ai/mxbai-rerank-large-v1`](https://huggingface.co/mixedbread-ai/mxbai-rerank-large-v1)       |       435M | English             |     512 |
| [`Xenova/bge-reranker-large`](https://huggingface.co/Xenova/bge-reranker-large)                           |      ~560M | Chinese and English |     512 |
| [`onnx-community/bge-reranker-v2-m3-ONNX`](https://huggingface.co/onnx-community/bge-reranker-v2-m3-ONNX) |       568M | Multilingual        |   8,192 |

### Documents

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

### Releasing resources

A reranker keeps its model loaded so it can be reused across calls. Models can be large and occupy a significant amount of memory. When you no longer need it, call `dispose()` to release the model's inference sessions and regain the memory:

```ts
const reranker = await Reranker.create({
  model: "mixedbread-ai/mxbai-rerank-base-v1",
});

try {
  const results = await reranker.rank(query, documents);
} finally {
  await reranker.dispose();
}
```

`Reranker` also implements `AsyncDisposable`, so runtimes that support explicit resource management can dispose it automatically:

```ts
await using reranker = await Reranker.create({
  model: "mixedbread-ai/mxbai-rerank-base-v1",
});

const results = await reranker.rank(query, documents);
```

Disposal waits for active ranking calls to finish and is safe to call more than once. A disposed reranker cannot be used again, trying to do so will throw a `RerankerDisposedError`.

## Ecosystem plugins

### LangChain

Install LangChain core alongside this package:

```sh
npm install rerankers @langchain/core
```

Use the `LocalReranker` adapter from the `rerankers/langchain` subpath anywhere LangChain expects a document compressor.

```ts
import { LocalReranker } from "rerankers/langchain";
import { Document } from "@langchain/core/documents";

// Using the documents from the first example
const lcDocs = documents.map(
  (doc) =>
    new Document({
      pageContent: doc.text,
      metadata: { id: doc.id },
    }),
);

const reranker = new LocalReranker({
  model: "mixedbread-ai/mxbai-rerank-base-v1",
  topK: 5,
});

try {
  const documents = await reranker.compressDocuments(lcDocs, query);
} finally {
  await reranker.dispose();
}
```

Returned LangChain documents are the original document objects in reranked order. Each returned document receives `metadata.relevanceScore`.

You can pass `transformerOptions` in the same initializer object

```ts
const reranker = new LocalReranker({
  model: "mixedbread-ai/mxbai-rerank-base-v1",
  transformerOptions: {
    dtype: "q8",
    device: "gpu",
  },
  topK: 5,
});
```

For full lifecycle control, pass an already-created reranker:

```ts
import { Reranker } from "rerankers";

const coreReranker = await Reranker.create({ model: "mixedbread-ai/mxbai-rerank-base-v1" });
const reranker = new LocalReranker({ reranker: coreReranker, topK: 5 });
```

`LocalReranker.dispose()` delegates to its core reranker, including when the core reranker was injected. `LocalReranker` also supports `await using` through `Symbol.asyncDispose`.

Use `rerank()` when you want raw scores without mutating LangChain document metadata:

```ts
const results = await reranker.rerank(documents, "red planet", { topK: 5 });
// [{ index: 1, relevanceScore: 0.92 }, ...]
```

### Vercel AI SDK

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

## Browser And Node

The package is ESM and keeps the runtime path compatible with browsers and Node.js. Use Transformers.js options such as `device` and `dtype` to tune runtime behavior.

## License

MIT
