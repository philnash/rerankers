import type { PretrainedModelOptions } from "@huggingface/transformers";

export type RerankerStrategyName = "cross-encoder" | (string & {});

export type RerankDocument<TMetadata = unknown> =
  | string
  | {
      text: string;
      id?: string;
      metadata?: TMetadata;
    };

export type NormalizedDocument<TDocument extends RerankDocument = RerankDocument> = {
  document: TDocument;
  index: number;
  text: string;
};

export type RerankResult<TDocument = RerankDocument> = {
  document: TDocument;
  index: number;
  score: number;
};

export type TransformerOptions = PretrainedModelOptions;

export type RerankerConfig = {
  model: string;
  strategy?: RerankerStrategyName;
  transformerOptions?: TransformerOptions;
};

export type NormalizedRerankerConfig = RerankerConfig & {
  strategy: RerankerStrategyName;
  transformerOptions: TransformerOptions;
};

export type RankOptions = {
  topK?: number;
};

export type ScoringStrategy = {
  score<TDocument extends RerankDocument>(
    query: string,
    documents: Array<NormalizedDocument<TDocument>>,
  ): Promise<Array<RerankResult<TDocument>>>;
  dispose?(): Promise<void>;
};

export type StrategyFactory = (
  config: NormalizedRerankerConfig,
) => ScoringStrategy | Promise<ScoringStrategy>;

export type RerankerCreateOptions = {
  strategyFactory?: StrategyFactory;
};
