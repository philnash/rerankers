import { describe, expect, it, vi } from "vitest";

import {
  CrossEncoderStrategy,
  type SequenceClassifierLoader,
} from "../src/strategies/cross-encoder.js";
import type { RerankerConfig } from "../src/index.js";

describe("CrossEncoderStrategy", () => {
  it("loads the configured sequence classifier once and scores query-document pairs", async () => {
    const tokenizedInputs = {
      attention_mask: [[1], [1]],
      input_ids: [[101], [101]],
    };
    const tokenizer = vi.fn(() => tokenizedInputs);
    const model = vi.fn(() =>
      Promise.resolve({
        logits: {
          sigmoid: () => ({
            tolist: () => [[0.2], [0.9]],
          }),
        },
      }),
    );
    const loader: SequenceClassifierLoader = vi.fn(() => Promise.resolve({ model, tokenizer }));
    const config: RerankerConfig = {
      model: "mixedbread-ai/mxbai-rerank-xsmall-v1",
      strategy: "cross-encoder",
      task: "text-ranking",
      transformerOptions: { dtype: "q8" },
    };
    const strategy = new CrossEncoderStrategy(config, loader);

    const results = await strategy.score("query", [
      { document: "first", index: 0, text: "first" },
      { document: "second", index: 1, text: "second" },
    ]);

    expect(loader).toHaveBeenCalledOnce();
    expect(loader).toHaveBeenCalledWith(config.model, { dtype: "q8" });
    expect(tokenizer).toHaveBeenCalledWith(["query", "query"], {
      padding: true,
      text_pair: ["first", "second"],
      truncation: true,
    });
    expect(model).toHaveBeenCalledWith(tokenizedInputs);
    expect(results).toEqual([
      { document: "first", index: 0, score: 0.2 },
      { document: "second", index: 1, score: 0.9 },
    ]);
  });
});
