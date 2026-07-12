# Rerankers TypeScript Library Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build a browser-compatible TypeScript reranking library over Transformers.js with simple presets, custom model configuration, and test-driven coverage.

**Architecture:** The public `Reranker` class resolves a preset or custom config, lazy-loads a scoring strategy, normalizes documents, scores them, and returns sorted results. Strategy implementations are isolated behind a shared interface so additional reranking strategies can be added later without changing the public ranking API.

**Tech Stack:** TypeScript, ESM, `@huggingface/transformers`, Vitest, ESLint, Prettier, tsup.

---

## File Structure

- `package.json`: package metadata, exports, scripts, dependencies, dev dependencies.
- `tsconfig.json`: strict TypeScript settings.
- `tsup.config.ts`: ESM and declaration build.
- `eslint.config.js`: ESLint flat config for TypeScript.
- `.prettierrc.json`: Prettier rules.
- `src/index.ts`: public exports.
- `src/types.ts`: public and internal types.
- `src/errors.ts`: typed library errors.
- `src/presets.ts`: named preset definitions and resolver.
- `src/document.ts`: document normalization and `k` validation.
- `src/scoring.ts`: score extraction helpers.
- `src/strategies/cross-encoder.ts`: cross-encoder strategy.
- `src/reranker.ts`: main `Reranker` class.
- `test/*.test.ts`: unit tests written before implementation.
- `README.md`: usage documentation and preset table.

## Tasks

### Task 1: Tooling And Package Skeleton

- [ ] Create TypeScript, ESLint, Prettier, Vitest, and tsup config files.
- [ ] Add scripts: `test`, `lint`, `format`, `typecheck`, `build`.
- [ ] Install dependencies.
- [ ] Run `npm test` and expect it to fail initially because no tests exist or no implementation exists.

### Task 2: Public API Tests

- [ ] Write failing tests for `Reranker.create("mixedbread-base")`, `rank(query, documents, { k })`, sorted scores, metadata preservation, empty documents, invalid `k`, and unknown presets.
- [ ] Use an injected fake strategy factory so tests do not download models.
- [ ] Run `npm test` and verify failures are caused by missing implementation.

### Task 3: Presets, Types, And Reranker

- [ ] Implement public types, errors, preset resolver, document normalization, and the main `Reranker` class.
- [ ] Run `npm test` and verify Task 2 tests pass.

### Task 4: Cross-Encoder Strategy

- [ ] Write failing tests for score normalization from pipeline/object/tensor-like outputs.
- [ ] Implement cross-encoder loading through `@huggingface/transformers` behind an injectable loader.
- [ ] Run `npm test` and verify cross-encoder tests pass.

### Task 5: README And Final Verification

- [ ] Document quick start, browser usage, Node usage, presets, custom configuration, and the custom strategy extension point.
- [ ] Run `npm run format`.
- [ ] Run `npm run lint`.
- [ ] Run `npm run typecheck`.
- [ ] Run `npm test`.
- [ ] Run `npm run build`.

## Self-Review

- The plan covers the approved spec including mixedbread presets and keeps the strategy boundary open for future implementations.
- Default tests avoid live Hugging Face downloads.
- There are no placeholders or deferred implementation sections.
- The workspace is not a git repository, so commit steps are intentionally omitted.
