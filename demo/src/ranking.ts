import { Reranker, type RerankResult } from "../../src/index.js";

export type DemoDocument = {
  id: string;
  text: string;
};

export type ModelOption = {
  id: string;
  label: string;
  note: string;
  tag: string;
};

export const MODEL_OPTIONS: readonly ModelOption[] = [
  {
    id: "mixedbread-ai/mxbai-rerank-xsmall-v1",
    label: "Mixedbread XSmall",
    note: "The quickest starting point for an in-browser experiment.",
    tag: "Lightweight",
  },
  {
    id: "mixedbread-ai/mxbai-rerank-base-v1",
    label: "Mixedbread Base",
    note: "A larger general-purpose model with a slower first load.",
    tag: "Balanced",
  },
  {
    id: "Xenova/bge-reranker-base",
    label: "BGE Reranker Base",
    note: "An alternative cross-encoder model converted for Transformers.js.",
    tag: "Alternative",
  },
] as const;

export function validateRankingInput(
  query: string,
  documents: readonly DemoDocument[],
): string | null {
  if (query.trim().length === 0) {
    return "Enter a query before ranking.";
  }

  if (!documents.some(({ text }) => text.trim().length > 0)) {
    return "Add at least one document with text before ranking.";
  }

  return null;
}

export function nonEmptyDocuments(documents: readonly DemoDocument[]): DemoDocument[] {
  return documents
    .filter(({ text }) => text.trim().length > 0)
    .map((document) => ({ ...document, text: document.text.trim() }));
}

export class RerankerCache {
  private readonly instances = new Map<string, Promise<Reranker>>();

  get(model: string): Promise<Reranker> {
    const existing = this.instances.get(model);
    if (existing) {
      return existing;
    }

    const instance = Reranker.create({
      model,
      transformerOptions: { dtype: "q8" },
    }).catch((error: unknown) => {
      this.instances.delete(model);
      throw error;
    });
    this.instances.set(model, instance);
    return instance;
  }
}

export type DemoRankingResult = RerankResult<DemoDocument>;
