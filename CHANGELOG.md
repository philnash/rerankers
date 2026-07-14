# Changelog

All notable changes to this project will be documented in this file.

The format is based on [Keep a Changelog](https://keepachangelog.com/en/1.1.0/),
and this project adheres to [Semantic Versioning](https://semver.org/spec/v2.0.0.html).

## [0.1.0] - 2026-07-14

### Added

- Browser- and Node.js-compatible local reranking powered by Transformers.js and open Hugging Face cross-encoder models.
- The `Reranker` API for ranking strings or structured documents while preserving each original document, index, and relevance score.
- Explicit model configuration with support for Transformers.js options such as `device` and `dtype`; `dtype` defaults to `auto`.
- Result limiting with the `topK` option.
- Custom reranking strategies through `strategyFactory`.
- Typed errors for invalid input, unsupported strategies, model-loading failures, and attempts to use a disposed reranker.
- Resource cleanup through `dispose()` and `Symbol.asyncDispose`, including safe handling of active ranking calls and repeated disposal.
- A LangChain `BaseDocumentCompressor` adapter exported from `rerankers/langchain`, with `compressDocuments()` and raw `rerank()` support.
- A Vercel AI SDK reranking model exported from `rerankers/ai-sdk`, supporting text and object documents.
- TypeScript declarations for the core library and both optional ecosystem adapters.
- Documentation for tested English, multilingual, and Chinese-English reranking models.
- A browser demo showing local document reranking.

[0.1.0]: https://github.com/philnash/rerankers/releases/tag/v0.1.0
