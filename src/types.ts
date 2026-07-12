import type { PretrainedModelOptions } from "@huggingface/transformers";

export type BuiltInRerankerStrategyName = "cross-encoder";
export type RerankerStrategyName = BuiltInRerankerStrategyName | (string & {});

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
};

export type RankOptions = {
  topK?: number;
};

export type ScoringStrategy = {
  score<TDocument extends RerankDocument>(
    query: string,
    documents: Array<NormalizedDocument<TDocument>>,
  ): Promise<Array<RerankResult<TDocument>>>;
};

export type StrategyFactory = (config: NormalizedRerankerConfig) => Promise<ScoringStrategy>;

export type RerankerCreateOptions = {
  strategyFactory?: StrategyFactory;
};
