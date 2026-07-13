import { AutoModelForSequenceClassification, AutoTokenizer } from "@huggingface/transformers";
import type { PreTrainedTokenizer } from "@huggingface/transformers";

import { RerankerModelLoadError } from "../errors.js";
import { extractScores } from "../scoring.js";
import type {
  NormalizedDocument,
  NormalizedRerankerConfig,
  RerankDocument,
  RerankResult,
  ScoringStrategy,
} from "../types.js";

type Tokenizer = (
  ...args: Parameters<PreTrainedTokenizer["_call"]>
) => ReturnType<PreTrainedTokenizer["_call"]>;

type TokenizerOutput = ReturnType<Tokenizer>;

type SequenceClassifier = (inputs: TokenizerOutput) => Promise<unknown>;

type SequenceClassifierPair = {
  model: SequenceClassifier;
  tokenizer: Tokenizer;
  dispose(): Promise<void>;
};

export type SequenceClassifierLoader = (
  model: string,
  options?: NormalizedRerankerConfig["transformerOptions"],
) => Promise<SequenceClassifierPair>;

export class CrossEncoderStrategy implements ScoringStrategy {
  private classifier: Promise<SequenceClassifierPair> | undefined;
  private disposePromise: Promise<void> | undefined;

  constructor(
    private readonly config: NormalizedRerankerConfig,
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

  dispose(): Promise<void> {
    this.disposePromise ??= this.disposeClassifier();
    return this.disposePromise;
  }

  private getClassifier(): Promise<SequenceClassifierPair> {
    this.classifier ??= this.load().catch((error: unknown) => {
      this.classifier = undefined;
      throw error;
    });

    return this.classifier;
  }

  private async load(): Promise<SequenceClassifierPair> {
    try {
      return await this.loadClassifier(this.config.model, this.config.transformerOptions);
    } catch (error) {
      throw new RerankerModelLoadError(this.config.model, error);
    }
  }

  private async disposeClassifier(): Promise<void> {
    const classifierPromise = this.classifier;
    this.classifier = undefined;

    if (classifierPromise === undefined) {
      return;
    }

    const classifier = await classifierPromise;
    await classifier.dispose();
  }
}

export function createCrossEncoderStrategy(config: NormalizedRerankerConfig): ScoringStrategy {
  return new CrossEncoderStrategy(config);
}

const defaultSequenceClassifierLoader: SequenceClassifierLoader = async (modelId, options) => {
  const [model, tokenizer] = await Promise.all([
    AutoModelForSequenceClassification.from_pretrained(modelId, options),
    AutoTokenizer.from_pretrained(modelId, options),
  ]);

  return {
    model: (inputs) => Promise.resolve(model(inputs)),
    tokenizer: (...args) => tokenizer._call(...args),
    dispose: async () => {
      await model.dispose();
    },
  };
};
