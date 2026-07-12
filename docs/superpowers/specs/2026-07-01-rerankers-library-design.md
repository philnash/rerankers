# Rerankers TypeScript Library Design

## Goal

Create a browser-compatible TypeScript library named `rerankers` that hides Transformers.js setup and exposes a small API for reranking candidate documents against a query.

The library should be easy for the common case:

```ts
import { Reranker } from "rerankers";

const reranker = await Reranker.create("mixedbread-base");
const results = await reranker.rank("Who wrote To Kill a Mockingbird?", documents, { k: 5 });
```

It should also allow advanced users to define their own Hugging Face model, task, scoring strategy, and Transformers.js options.

## Runtime And Tooling

- TypeScript ESM package.
- Browser-compatible first, with Node.js support through the same public API.
- Transformers.js dependency: `@huggingface/transformers`.
- Test runner: Vitest.
- Code quality: ESLint and Prettier.
- Build output: bundled JavaScript and `.d.ts` types.

## Public API

The main API is `Reranker.create()` plus an async `rank()` method.

```ts
const reranker = await Reranker.create("bge");
const results = await reranker.rank(query, documents, { k: 3 });
```

Documents may be strings or objects:

```ts
type RerankDocument<TMetadata = unknown> =
  | string
  | {
      text: string;
      id?: string;
      metadata?: TMetadata;
    };
```

Results preserve the original document and include its original index:

```ts
type RerankResult<TDocument> = {
  document: TDocument;
  index: number;
  score: number;
};
```

Custom model configuration should be explicit:

```ts
const reranker = await Reranker.create({
  model: "mixedbread-ai/mxbai-rerank-large-v1",
  task: "text-ranking",
  strategy: "cross-encoder",
  transformerOptions: {
    dtype: "q8",
    device: "wasm",
  },
});
```

## Presets

Named presets should map memorable names to open Hugging Face model ids and strategies:

- `bge`: `Xenova/bge-reranker-base`, cross-encoder/text-ranking.
- `minilm`: `Xenova/ms-marco-MiniLM-L-6-v2`, cross-encoder/text-classification.
- `mixedbread-xsmall`: `mixedbread-ai/mxbai-rerank-xsmall-v1`, cross-encoder/text-ranking.
- `mixedbread-base`: `mixedbread-ai/mxbai-rerank-base-v1`, cross-encoder/text-ranking.
- `mixedbread-large`: `mixedbread-ai/mxbai-rerank-large-v1`, cross-encoder/text-ranking.

ModernBERT should be supported through custom model configuration in the first version. A named ModernBERT preset should only be added after identifying a specific Transformers.js-compatible model and scoring path.

## Strategies

### Cross-Encoder

This is the default and most reliable initial strategy. It scores each `(query, document)` pair with a model such as BGE, MiniLM, or Mixedbread rerankers.

The implementation should accept Transformers.js output shapes seen from pipelines and sequence-classification models:

- A number.
- A single object with `score`.
- A single object with `logits`.
- An array of objects with `score`.
- Tensor-like output with `sigmoid()` and `tolist()`.

The first implementation should use the high-level Transformers.js `pipeline()` API where possible, while keeping a loader boundary injectable for tests.

### Strategy Extension

The first version should only ship the cross-encoder implementation. The `strategy` field and injectable `strategyFactory` should remain as the extension point for future reranking strategies or project-specific model integrations.

## Error Handling

- Unknown preset names should throw a clear `UnknownPresetError`.
- Invalid documents should throw a `RerankerInputError`.
- `k` must be a positive integer when provided.
- Empty document lists should return an empty array without loading or scoring.
- Model-loading errors should retain the original cause and include model id and task.

## Testing

Implementation must be test-driven.

Unit tests should avoid network downloads by injecting fake Transformers.js loaders and fake scoring functions. Tests should cover:

- Named preset resolution.
- Custom model configuration.
- Ranking sorted by descending score.
- `k` limiting.
- String documents.
- Object documents with `id` and `metadata`.
- Empty document list avoiding model calls.
- Invalid `k`.
- Unknown preset.
- Cross-encoder score normalization from representative output shapes.
- Custom strategy names passed to an injected strategy factory.
- Unsupported strategy names throwing clearly when no custom strategy factory is provided.

Integration tests that download real Hugging Face models are optional and should be opt-in, not part of the default test suite.

## Documentation

The README should include:

- Quick start.
- Browser usage notes.
- Node usage notes.
- Preset table.
- Custom model configuration.
- Explanation that models are downloaded/cached by Transformers.js.
- Explanation of the custom strategy extension point.

## References Checked

- Transformers.js supports browser execution, pipeline creation, model ids, `device`, and `dtype` options.
- `Xenova/bge-reranker-base` advertises Transformers.js `text-ranking` usage.
- `Xenova/ms-marco-MiniLM-L-6-v2` advertises Transformers.js sequence-classification usage.
- `mixedbread-ai/mxbai-rerank-xsmall-v1`, `mixedbread-ai/mxbai-rerank-base-v1`, and `mixedbread-ai/mxbai-rerank-large-v1` advertise Transformers.js `text-ranking` usage.

## Self-Review

- No placeholder requirements remain.
- Browser compatibility is explicit.
- Mixedbread models are first-class presets.
- Strategy extension remains possible without including additional strategy implementations in the first version.
- ModernBERT is supported by the custom configuration path until a concrete Transformers.js-compatible reranking model is selected.
