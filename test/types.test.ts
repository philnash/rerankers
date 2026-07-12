import { describe, expect, it } from "vitest";

import type { SequenceClassifierLoader } from "../src/strategies/cross-encoder.js";
import type { RankOptions, RerankerConfig } from "../src/index.js";

const validTransformersOptionsConfig: RerankerConfig = {
  model: "mixedbread-ai/mxbai-rerank-base-v1",
  strategy: "cross-encoder",
  transformerOptions: {
    device: "wasm",
    dtype: "q8",
    local_files_only: true,
  },
};

const invalidTransformersOptionsConfig: RerankerConfig = {
  model: "mixedbread-ai/mxbai-rerank-base-v1",
  strategy: "cross-encoder",
  transformerOptions: {
    // @ts-expect-error dtype should come from Transformers.js DataType.
    dtype: "definitely-not-a-transformers-dtype",
  },
};

const invalidSequenceClassifierPair: Awaited<ReturnType<SequenceClassifierLoader>> = {
  // @ts-expect-error model must be callable.
  model: {},
  tokenizer: () => ({
    attention_mask: [[1]],
    input_ids: [[101]],
  }),
};

const validRankOptions: RankOptions = {
  topK: 3,
};

const customStrategyConfig: RerankerConfig = {
  model: "example/custom-reranker",
  strategy: "custom-strategy",
};

// @ts-expect-error rank options should use topK, not k.
void ({ k: 1 } satisfies RankOptions);

describe("public Transformers.js option types", () => {
  it("accepts valid Transformers.js pretrained model options", () => {
    expect(validTransformersOptionsConfig.transformerOptions).toMatchObject({
      device: "wasm",
      dtype: "q8",
      local_files_only: true,
    });
  });
});

void invalidTransformersOptionsConfig;
void invalidSequenceClassifierPair;
void validRankOptions;
void customStrategyConfig;
