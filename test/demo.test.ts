import { describe, expect, it } from "vitest";

import { nonEmptyDocuments, validateRankingInput } from "../demo/src/ranking.js";

describe("demo ranking helpers", () => {
  it("requires a query", () => {
    expect(validateRankingInput("  ", [{ id: "one", text: "A document" }])).toBe(
      "Enter a query before ranking.",
    );
  });

  it("requires at least one non-empty document", () => {
    expect(validateRankingInput("query", [{ id: "one", text: "  " }])).toBe(
      "Add at least one document with text before ranking.",
    );
  });

  it("filters empty documents and trims rankable text", () => {
    expect(
      nonEmptyDocuments([
        { id: "one", text: "  first  " },
        { id: "two", text: "" },
        { id: "three", text: "third" },
      ]),
    ).toEqual([
      { id: "one", text: "first" },
      { id: "three", text: "third" },
    ]);
  });
});
