export class RerankerError extends Error {
  constructor(message: string, options?: ErrorOptions) {
    super(message, options);
    this.name = new.target.name;
  }
}

export class UnknownPresetError extends RerankerError {
  constructor(preset: string) {
    super(`Unknown reranker preset: ${preset}`);
  }
}

export class RerankerInputError extends RerankerError {}

export class UnsupportedStrategyError extends RerankerError {
  constructor(strategy: string) {
    super(`Unsupported reranker strategy: ${strategy}`);
  }
}

export class RerankerModelLoadError extends RerankerError {
  constructor(model: string, task: string | undefined, cause: unknown) {
    super(`Failed to load reranker model "${model}"${task ? ` for task "${task}"` : ""}.`, {
      cause,
    });
  }
}
