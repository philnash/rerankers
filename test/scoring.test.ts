import { describe, expect, it } from "vitest";

import { extractScore, extractScores } from "../src/scoring.js";

describe("extractScore", () => {
  it("extracts a numeric score", () => {
    expect(extractScore(0.42)).toBe(0.42);
  });

  it("extracts score from pipeline objects", () => {
    expect(extractScore({ label: "LABEL_0", score: 0.7 })).toBe(0.7);
  });

  it("extracts score from the first array item", () => {
    expect(extractScore([{ label: "LABEL_0", score: 0.8 }])).toBe(0.8);
  });

  it("extracts score from a tensor-like sigmoid output", () => {
    const tensorLike = {
      sigmoid: () => ({
        tolist: () => [[0.91]],
      }),
    };

    expect(extractScore({ logits: tensorLike })).toBe(0.91);
  });

  it("extracts one score per row from tensor-like logits", () => {
    const tensorLike = {
      sigmoid: () => ({
        tolist: () => [[0.3], [0.7]],
      }),
    };

    expect(extractScores({ logits: tensorLike }, 2)).toEqual([0.3, 0.7]);
  });

  it("extracts scores directly from raw multi-row tensor-like output", () => {
    const tensorLike = {
      sigmoid: () => ({
        tolist: () => [[0.3], [0.7]],
      }),
    };

    expect(extractScores(tensorLike, 2)).toEqual([0.3, 0.7]);
  });
});
