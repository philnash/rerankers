export type RerankerPreset =
  "bge" | "minilm" | "mixedbread-xsmall" | "mixedbread-base" | "mixedbread-large" | "colbert-small";

export type RerankerStrategyName = "cross-encoder" | "late-interaction";

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

export type TransformerOptions = Record<string, unknown>;

export type RerankerConfig = {
  model: string;
  strategy: RerankerStrategyName;
  task?: string;
  transformerOptions?: TransformerOptions;
  experimental?: boolean;
};

export type RankOptions = {
  k?: number;
};

export type ScoringStrategy = {
  score<TDocument extends RerankDocument>(
    query: string,
    documents: Array<NormalizedDocument<TDocument>>,
  ): Promise<Array<RerankResult<TDocument>>>;
};

export type StrategyFactory = (config: RerankerConfig) => Promise<ScoringStrategy>;

export type RerankerCreateOptions = {
  strategyFactory?: StrategyFactory;
};

export type RerankerDefinition = string | RerankerConfig;
