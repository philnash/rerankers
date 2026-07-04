# rerankers

Browser-compatible TypeScript reranking over open Hugging Face models through Transformers.js.

```ts
import { Reranker } from "rerankers";

const reranker = await Reranker.create("mixedbread-base");
const results = await reranker.rank("Who wrote To Kill a Mockingbird?", documents, { k: 5 });
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

## Browser And Node

The package is ESM and keeps the runtime path compatible with browsers and Node.js. Use Transformers.js options such as `device` and `dtype` to tune runtime behavior.

## Late Interaction

ColBERT-style models score token vectors with late interaction instead of returning one cross-encoder score. The library includes the MaxSim scoring primitive and an experimental late-interaction strategy boundary, but model-specific vector extraction depends on what the selected Transformers.js-compatible model exposes.

For production ColBERT use, provide a custom `strategyFactory` until the chosen model's token-vector path is verified.
