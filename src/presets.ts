import { UnknownPresetError } from "./errors.js";
import type { RerankerConfig, RerankerPreset } from "./types.js";

const presets = {
  bge: {
    model: "Xenova/bge-reranker-base",
    strategy: "cross-encoder",
    task: "text-ranking",
  },
  minilm: {
    model: "Xenova/ms-marco-MiniLM-L-6-v2",
    strategy: "cross-encoder",
    task: "text-classification",
  },
  "mixedbread-xsmall": {
    model: "mixedbread-ai/mxbai-rerank-xsmall-v1",
    strategy: "cross-encoder",
    task: "text-ranking",
  },
  "mixedbread-base": {
    model: "mixedbread-ai/mxbai-rerank-base-v1",
    strategy: "cross-encoder",
    task: "text-ranking",
  },
  "mixedbread-large": {
    model: "mixedbread-ai/mxbai-rerank-large-v1",
    strategy: "cross-encoder",
    task: "text-ranking",
  },
  "colbert-small": {
    model: "answerdotai/answerai-colbert-small-v1",
    strategy: "late-interaction",
    experimental: true,
  },
} satisfies Record<RerankerPreset, RerankerConfig>;

export function resolvePreset(preset: string): RerankerConfig {
  if (preset in presets) {
    return { ...presets[preset as RerankerPreset] };
  }

  throw new UnknownPresetError(preset);
}

export const presetNames = Object.keys(presets) as RerankerPreset[];
