# LangChain LocalReranker Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a LangChain JS `LocalReranker` adapter and rename the core rank limit option from `k` to `topK` for the first release API.

**Architecture:** Keep the core package export independent from LangChain and expose the adapter through `rerankers/langchain`. The adapter converts LangChain documents to core reranker documents, preserves original document objects, writes `metadata.relevanceScore`, and provides a raw `rerank()` helper.

**Tech Stack:** TypeScript, Vitest, tsup, `@langchain/core`, Transformers.js-backed core reranker.

---

## File Structure

- `src/types.ts`: rename `RankOptions.k` to `RankOptions.topK`.
- `src/document.ts`: rename `normalizeK()` to `normalizeTopK()` and update validation text.
- `src/reranker.ts`: call `normalizeTopK()` and slice by `topK`.
- `src/langchain.ts`: new subpath adapter export containing `LocalReranker` and related public types.
- `src/index.ts`: unchanged for LangChain imports; still exports core types.
- `test/reranker.test.ts`: update core tests to use `topK`.
- `test/langchain.test.ts`: new adapter tests using fake core rerankers so no model download is needed.
- `test/types.test.ts`: add type-level coverage for `topK` and reject `k`.
- `package.json`: add `./langchain` export and `@langchain/core` optional peer/dev dependency.
- `package-lock.json`: update through `npm install @langchain/core --save-dev`.
- `tsup.config.ts`: build `src/index.ts` and `src/langchain.ts`; externalize `@langchain/core`.
- `README.md`: update core examples and add LangChain usage.
- `demo.local.mjs`: update local-only demo to use `topK`.

## Task 1: Rename Core Rank Limit To topK

**Files:**

- Modify: `src/types.ts`
- Modify: `src/document.ts`
- Modify: `src/reranker.ts`
- Modify: `test/reranker.test.ts`
- Modify: `test/types.test.ts`

- [ ] **Step 1: Write failing tests for `topK`**

Update `test/reranker.test.ts` so the top-result test calls:

```ts
const results = await reranker.rank("red planet", ["Venus", "Mars", "Jupiter"], { topK: 2 });
```

Update the invalid option test to call:

```ts
await expect(reranker.rank("query", ["doc"], { topK: 0 })).rejects.toBeInstanceOf(
  RerankerInputError,
);
```

Add a compile-time rejection in `test/types.test.ts`:

```ts
// @ts-expect-error rank options should use topK, not k.
void ({ k: 1 } satisfies RankOptions);
```

- [ ] **Step 2: Run the focused tests and verify failure**

Run:

```sh
npm test -- test/reranker.test.ts test/types.test.ts
```

Expected: TypeScript/Vitest fails because `topK` is not in `RankOptions` yet, or because the `@ts-expect-error` is unused.

- [ ] **Step 3: Implement the minimal core rename**

Change `src/types.ts`:

```ts
export type RankOptions = {
  topK?: number;
};
```

Change `src/document.ts` to export `normalizeTopK()`:

```ts
export function normalizeTopK(options: RankOptions | undefined, documentCount: number): number {
  if (options?.topK === undefined) {
    return documentCount;
  }

  if (!Number.isInteger(options.topK) || options.topK < 1) {
    throw new RerankerInputError("rank option topK must be a positive integer.");
  }

  return Math.min(options.topK, documentCount);
}
```

Change `src/reranker.ts` to import and call `normalizeTopK()`.

- [ ] **Step 4: Run the focused tests and verify pass**

Run:

```sh
npm test -- test/reranker.test.ts test/types.test.ts
```

Expected: tests pass.

## Task 2: Add LangChain Adapter Tests

**Files:**

- Create: `test/langchain.test.ts`

- [ ] **Step 1: Install LangChain core for tests**

Run:

```sh
npm install @langchain/core --save-dev
```

Expected: `package.json` and `package-lock.json` include `@langchain/core`.

- [ ] **Step 2: Write failing adapter tests**

Create `test/langchain.test.ts` with tests that import `LocalReranker` from `../src/langchain.js`, construct it with a fake reranker object, and verify:

```ts
expect(compressed).toEqual([documents[1], documents[2]]);
expect(documents[1].metadata).toMatchObject({ source: "b", relevanceScore: 0.9 });
expect(await local.rerank(documents, "query")).toEqual([
  { index: 1, relevanceScore: 0.9 },
  { index: 2, relevanceScore: 0.4 },
]);
```

Also verify constructor `topK`, method override `topK`, empty arrays, string inputs, and runtime rejection when both `model` and `reranker` are supplied.

- [ ] **Step 3: Run the adapter tests and verify failure**

Run:

```sh
npm test -- test/langchain.test.ts
```

Expected: tests fail because `src/langchain.ts` does not exist.

## Task 3: Implement LocalReranker

**Files:**

- Create: `src/langchain.ts`

- [ ] **Step 1: Add the adapter implementation**

Create `src/langchain.ts` with:

```ts
import type { DocumentInterface } from "@langchain/core/documents";
import { BaseDocumentCompressor } from "@langchain/core/retrievers/document_compressors";

import { Reranker } from "./reranker.js";
import type { RankOptions, RerankerCreateOptions, RerankerDefinition } from "./types.js";
```

Define `LocalRerankerArgs`, `LocalRerankerRerankOptions`, and `LocalRerankerResult`. Store either a `Promise<Reranker>` created from `Reranker.create(model, createOptions)` or a resolved warmed reranker. Reject runtime args that include both `model` and `reranker`.

`compressDocuments()` should call `this.rerank(documents, query)`, write `metadata.relevanceScore` on each selected original document, and return those original objects.

`rerank()` should normalize inputs into `{ text, metadata: original }`, call core `rank(query, normalized, { topK })`, and map results to `{ index, relevanceScore: score }`.

- [ ] **Step 2: Run the adapter tests and verify pass**

Run:

```sh
npm test -- test/langchain.test.ts
```

Expected: tests pass.

## Task 4: Package And Documentation

**Files:**

- Modify: `package.json`
- Modify: `tsup.config.ts`
- Modify: `README.md`
- Modify: `demo.local.mjs`

- [ ] **Step 1: Update package exports and build config**

Add to `package.json` exports:

```json
"./langchain": {
  "types": "./dist/langchain.d.ts",
  "import": "./dist/langchain.js"
}
```

Add optional peer dependency:

```json
"peerDependencies": {
  "@langchain/core": "^1.2.1"
},
"peerDependenciesMeta": {
  "@langchain/core": {
    "optional": true
  }
}
```

Update `tsup.config.ts`:

```ts
export default defineConfig({
  clean: true,
  dts: true,
  entry: ["src/index.ts", "src/langchain.ts"],
  external: ["@langchain/core"],
  format: ["esm"],
  sourcemap: true,
  splitting: false,
});
```

- [ ] **Step 2: Update docs and demo**

Update README core examples to `{ topK: 5 }`. Add a LangChain section showing:

```ts
import { LocalReranker } from "rerankers/langchain";

const reranker = new LocalReranker({
  model: "mixedbread-base",
  topK: 5,
});

const documents = await reranker.compressDocuments(results, query);
```

Mention `metadata.relevanceScore`, `createOptions`, and warmed `reranker` advanced usage.

Update `demo.local.mjs` to call rank with `{ topK }`.

## Task 5: Full Verification

**Files:**

- All touched files

- [ ] **Step 1: Run full test suite**

Run:

```sh
npm test
```

Expected: all tests pass.

- [ ] **Step 2: Run typecheck**

Run:

```sh
npm run typecheck
```

Expected: no TypeScript errors.

- [ ] **Step 3: Run lint**

Run:

```sh
npm run lint
```

Expected: no ESLint errors.

- [ ] **Step 4: Run build**

Run:

```sh
npm run build
```

Expected: `dist/index.js`, `dist/index.d.ts`, `dist/langchain.js`, and `dist/langchain.d.ts` are generated.
