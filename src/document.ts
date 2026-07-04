import { RerankerInputError } from "./errors.js";
import type { NormalizedDocument, RankOptions, RerankDocument } from "./types.js";

export function normalizeDocuments<TDocument extends RerankDocument>(
  documents: readonly TDocument[],
): Array<NormalizedDocument<TDocument>> {
  return documents.map((document, index) => {
    if (typeof document === "string") {
      return { document, index, text: document };
    }

    if (!document || typeof document.text !== "string") {
      throw new RerankerInputError(
        "Each document must be a string or an object with a text field.",
      );
    }

    return { document, index, text: document.text };
  });
}

export function normalizeTopK(options: RankOptions | undefined, documentCount: number): number {
  if (options?.topK === undefined) {
    return documentCount;
  }

  if (!Number.isInteger(options.topK) || options.topK < 1) {
    throw new RerankerInputError("rank option topK must be a positive integer.");
  }

  return Math.min(options.topK, documentCount);
}
