import type {
  NormalizedDocument,
  RerankDocument,
  RerankResult,
  ScoringStrategy,
} from "../types.js";

export type TokenVector = readonly number[];
export type TokenMatrix = readonly TokenVector[];

export type LateInteractionEmbedder = (
  text: string,
  role: "query" | "document",
) => Promise<TokenMatrix>;

export class LateInteractionStrategy implements ScoringStrategy {
  constructor(private readonly embed: LateInteractionEmbedder) {}

  async score<TDocument extends RerankDocument>(
    query: string,
    documents: Array<NormalizedDocument<TDocument>>,
  ): Promise<Array<RerankResult<TDocument>>> {
    const queryVectors = await this.embed(query, "query");
    const results = await Promise.all(
      documents.map(async ({ document, index, text }) => ({
        document,
        index,
        score: maxSimScore(queryVectors, await this.embed(text, "document")),
      })),
    );

    return results;
  }
}

export function maxSimScore(query: TokenMatrix, document: TokenMatrix): number {
  return query.reduce((total, queryToken) => {
    const best = document.reduce(
      (max, documentToken) => Math.max(max, dotProduct(queryToken, documentToken)),
      Number.NEGATIVE_INFINITY,
    );

    return total + (Number.isFinite(best) ? best : 0);
  }, 0);
}

function dotProduct(left: TokenVector, right: TokenVector): number {
  const length = Math.min(left.length, right.length);
  let total = 0;

  for (let index = 0; index < length; index += 1) {
    total += (left[index] ?? 0) * (right[index] ?? 0);
  }

  return total;
}
