import { fileURLToPath } from "node:url";

export const modelCacheDirectory = fileURLToPath(
  new URL("../.cache/rerankers-model-benchmark", import.meta.url),
);
export const benchmarkReportPath = fileURLToPath(
  new URL("../model-benchmark-results.html", import.meta.url),
);

export const modelIds = [
  "Xenova/ms-marco-TinyBERT-L-2-v2",
  "Xenova/ms-marco-MiniLM-L-2-v2",
  "Xenova/ms-marco-MiniLM-L-4-v2",
  "Xenova/ms-marco-MiniLM-L-6-v2",
  "Xenova/ms-marco-MiniLM-L-12-v2",
  "jinaai/jina-reranker-v1-tiny-en",
  "jinaai/jina-reranker-v1-turbo-en",
  "mixedbread-ai/mxbai-rerank-xsmall-v1",
  "mixedbread-ai/mxbai-rerank-base-v1",
  "jinaai/jina-reranker-v2-base-multilingual",
  "Xenova/bge-reranker-base",
  "onnx-community/gte-multilingual-reranker-base",
  "mixedbread-ai/mxbai-rerank-large-v1",
  "Xenova/bge-reranker-large",
  "onnx-community/bge-reranker-v2-m3-ONNX",
] as const;
