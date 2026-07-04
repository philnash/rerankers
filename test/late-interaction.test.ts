import { describe, expect, it } from "vitest";

import { maxSimScore } from "../src/strategies/late-interaction.js";

describe("maxSimScore", () => {
  it("sums each query token's best document-token dot product", () => {
    const query = [
      [1, 0],
      [0, 1],
    ];
    const document = [
      [0.5, 0.5],
      [0.2, 0.9],
    ];

    expect(maxSimScore(query, document)).toBeCloseTo(1.4);
  });
});
