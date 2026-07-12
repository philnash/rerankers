# LangChain LocalReranker Adapter Design

## Goal

Add a LangChain JS adapter for this package that plugs into LangChain's document compression interface while preserving the library's local, Transformers.js-powered reranking model. The adapter should feel like the Cohere and MixedBread LangChain rerankers, but use this package's first-release API shape.

## Core API Rename

The core `Reranker.rank()` option currently named `k` will be renamed to `topK` throughout the codebase:

```ts
const results = await reranker.rank(query, documents, { topK: 5 });
```

`RankOptions` will become:

```ts
type RankOptions = {
  topK?: number;
};
```

There will be no `k` compatibility alias. The package has not been released yet, so the first release should use the clearest API rather than carrying a deprecated option. Validation and error messages should refer to `topK`.

Existing tests, README examples, and the local-only demo script should be updated to use `topK`.

## LangChain Export

The LangChain adapter will be exposed from a subpath export:

```ts
import { LocalReranker } from "rerankers/langchain";
```

The root `rerankers` export should not import LangChain. This keeps the core browser-compatible library independent from LangChain for users who do not need the adapter.

Packaging changes:

- add `@langchain/core` as an optional peer dependency
- add `@langchain/core` as a dev dependency for typechecking and tests
- add a `./langchain` export in `package.json`
- build both `src/index.ts` and `src/langchain.ts`
- keep `@langchain/core` external in the bundle

## Public Adapter API

The adapter class will be named `LocalReranker`. It extends LangChain's `BaseDocumentCompressor`, but the class name should not include `DocumentCompressor`.

Default construction creates the underlying core reranker:

```ts
const compressor = new LocalReranker({
  model: "mixedbread-base",
  strategyFactory,
  topK: 5,
});
```

Advanced construction accepts an already-created core reranker:

```ts
const compressor = new LocalReranker({
  reranker,
  topK: 5,
});
```

The two construction modes should be mutually exclusive. `model` is a `RerankerDefinition`; `strategyFactory` is passed through to `Reranker.create(model, { strategyFactory })`. If `model` is omitted, the adapter should use the core default model.

The advanced `reranker` option exists for lifecycle control, sharing an already-warmed model instance, and tests that supply a fake rank implementation.

Suggested type shape:

```ts
type LocalRerankerArgs =
  | {
      model?: RerankerDefinition;
      strategyFactory?: StrategyFactory;
      reranker?: never;
      topK?: number;
    }
  | {
      reranker: Reranker;
      model?: never;
      strategyFactory?: never;
      topK?: number;
    };
```

## Adapter Behavior

`LocalReranker` will implement:

```ts
compressDocuments(
  documents: DocumentInterface[],
  query: string,
  callbacks?: Callbacks,
): Promise<DocumentInterface[]>;
```

Behavior:

- return `[]` for empty input
- extract each document's text from `document.pageContent`
- rerank through the core `Reranker.rank()` API
- return the original LangChain document objects in reranked order
- write `document.metadata.relevanceScore = score` on each returned document
- use constructor `topK` unless a method-level override is supplied through `rerank(..., { topK })`

`LocalReranker` will also expose a helper:

```ts
rerank(
  documents: Array<DocumentInterface | string | { pageContent: string }>,
  query: string,
  options?: { topK?: number },
): Promise<Array<{ index: number; relevanceScore: number }>>;
```

The helper returns raw ranked results with original input indexes and scores normalized to the `relevanceScore` name used by the LangChain Cohere and MixedBread adapters.

For `compressDocuments`, the result metadata key will be exactly `relevanceScore`.

## Error Handling

The adapter should rely on the core reranker's existing model and input validation where possible.

Adapter-specific validation should be small:

- reject invalid `topK` via the core `Reranker.rank()` path
- do not silently accept both `model` and `reranker` at runtime if JavaScript users bypass TypeScript
- preserve clear errors from `Reranker.create()`

## Tests

Core tests:

- update existing `k` tests and fixtures to `topK`
- verify invalid `topK` raises `RerankerInputError`

LangChain adapter tests:

- use a fake reranker or fake strategy so tests do not download models
- verify `compressDocuments()` preserves original document objects
- verify documents are returned in reranked order
- verify existing metadata is preserved
- verify `metadata.relevanceScore` is written
- verify `rerank()` returns `{ index, relevanceScore }[]`
- verify constructor `topK` is used
- verify method-level `topK` override is used by `rerank()`
- verify empty input returns `[]`

Verification commands:

```sh
npm test
npm run typecheck
npm run lint
npm run build
```

## Documentation

The README should:

- update core examples from `{ k: ... }` to `{ topK: ... }`
- add a LangChain section using `LocalReranker`
- show `strategyFactory` passed through to `Reranker.create()`
- mention the warmed-instance `reranker` option as advanced usage
- document that returned LangChain documents receive `metadata.relevanceScore`
