export class RerankerError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = new.target.name;
  }
}

export class RerankerInputError extends RerankerError {}

export class RerankerDisposedError extends RerankerError {
  constructor() {
    super("Cannot rank with a disposed reranker.");
  }
}

export class UnsupportedStrategyError extends RerankerError {
  constructor(strategy: string) {
    super(`Unsupported reranker strategy: ${strategy}`);
  }
}

export class RerankerModelLoadError extends RerankerError {
  constructor(model: string, cause: unknown) {
    super(`Failed to load reranker model "${model}".`, {
      cause,
    });
  }
}
