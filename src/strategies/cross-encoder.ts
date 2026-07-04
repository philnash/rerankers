import { AutoModelForSequenceClassification, AutoTokenizer } from "@huggingface/transformers";

import { RerankerModelLoadError } from "../errors.js";
import { extractScores } from "../scoring.js";
import type {
  NormalizedDocument,
  RerankerConfig,
  RerankDocument,
  RerankResult,
  ScoringStrategy,
} from "../types.js";

type Tokenizer = (
  text: string | string[],
  options?: {
    padding?: boolean;
    text_pair?: string | string[];
    truncation?: boolean;
  },
) => unknown;

type SequenceClassifier = (inputs: unknown) => Promise<unknown>;

export type SequenceClassifierLoader = (
  model: string,
  options?: Record<string, unknown>,
) => Promise<{
  model: SequenceClassifier;
  tokenizer: Tokenizer;
}>;

export class CrossEncoderStrategy implements ScoringStrategy {
  private classifier: Promise<{
    model: SequenceClassifier;
    tokenizer: Tokenizer;
  }> | undefined;

  constructor(
    private readonly config: RerankerConfig,
    private readonly loadClassifier: SequenceClassifierLoader = defaultSequenceClassifierLoader,
  ) {}

  async score<TDocument extends RerankDocument>(
    query: string,
    documents: Array<NormalizedDocument<TDocument>>,
  ): Promise<Array<RerankResult<TDocument>>> {
    const { model, tokenizer } = await this.getClassifier();
    const inputs = tokenizer(new Array<string>(documents.length).fill(query), {
      padding: true,
      text_pair: documents.map(({ text }) => text),
      truncation: true,
    });
    const scores = extractScores(await model(inputs), documents.length);

    return documents.map(({ document, index }, resultIndex) => ({
      document,
      index,
      score: scores[resultIndex] ?? 0,
    }));
  }

  private getClassifier(): Promise<{
    model: SequenceClassifier;
    tokenizer: Tokenizer;
  }> {
    this.classifier ??= this.load().catch((error: unknown) => {
      this.classifier = undefined;
      throw error;
    });

    return this.classifier;
  }

  private async load(): Promise<{
    model: SequenceClassifier;
    tokenizer: Tokenizer;
  }> {
    try {
      return await this.loadClassifier(this.config.model, this.config.transformerOptions);
    } catch (error) {
      throw new RerankerModelLoadError(this.config.model, this.config.task, error);
    }
  }
}

export function createCrossEncoderStrategy(config: RerankerConfig): ScoringStrategy {
  return new CrossEncoderStrategy(config);
}

const defaultSequenceClassifierLoader: SequenceClassifierLoader = async (modelId, options) => {
  const [model, tokenizer] = await Promise.all([
    AutoModelForSequenceClassification.from_pretrained(modelId, options),
    AutoTokenizer.from_pretrained(modelId, options),
  ]);

  return {
    model: model as SequenceClassifier,
    tokenizer: tokenizer as Tokenizer,
  };
};
