type TensorLike = {
  sigmoid?: () => {
    tolist?: () => unknown;
  };
  tolist?: () => unknown;
};

export function extractScore(output: unknown): number {
  if (typeof output === "number") {
    return output;
  }

  if (Array.isArray(output)) {
    if (output.length === 0) {
      throw new TypeError("Cannot extract a score from an empty output array.");
    }

    return extractScore(output[0]);
  }

  if (isRecord(output)) {
    if (typeof output.score === "number") {
      return output.score;
    }

    if ("logits" in output) {
      return extractScoreFromTensor(output.logits);
    }
  }

  return extractScoreFromTensor(output);
}

export function extractScores(output: unknown, expectedCount: number): number[] {
  if (expectedCount === 0) {
    return [];
  }

  if (Array.isArray(output) && output.length === expectedCount) {
    return output.map((item) => extractScore(item));
  }

  if (isRecord(output) && "logits" in output) {
    const tensorScores = extractScoresFromTensor(output.logits);
    if (tensorScores.length >= expectedCount) {
      return tensorScores.slice(0, expectedCount);
    }
  }

  if (isTensorLike(output)) {
    const tensorScores = extractScoresFromTensor(output);
    if (tensorScores.length >= expectedCount) {
      return tensorScores.slice(0, expectedCount);
    }
  }

  if (expectedCount === 1) {
    return [extractScore(output)];
  }

  throw new TypeError(`Could not extract ${expectedCount} reranking scores from model output.`);
}

function getTensorValues(output: TensorLike): unknown {
  return output.sigmoid?.().tolist?.() ?? output.tolist?.();
}

function extractScoreFromTensor(output: unknown): number {
  if (!isTensorLike(output)) {
    throw new TypeError("Could not extract a numeric reranking score from model output.");
  }

  const values = getTensorValues(output);
  const score = findFirstNestedNumber(values);
  if (score === undefined) {
    throw new TypeError("Could not extract a numeric reranking score from tensor output.");
  }

  return score;
}

function extractScoresFromTensor(output: unknown): number[] {
  if (!isTensorLike(output)) {
    return [];
  }

  const values = getTensorValues(output);

  if (!Array.isArray(values)) {
    const score = findFirstNestedNumber(values);
    return score === undefined ? [] : [score];
  }

  return values
    .map((row) => findFirstNestedNumber(row))
    .filter((score): score is number => score !== undefined);
}

function findFirstNestedNumber(value: unknown): number | undefined {
  if (typeof value === "number") {
    return value;
  }

  if (Array.isArray(value)) {
    for (const item of value) {
      const score = findFirstNestedNumber(item);
      if (score !== undefined) {
        return score;
      }
    }
  }

  return undefined;
}

function isRecord(value: unknown): value is Record<string, unknown> {
  return typeof value === "object" && value !== null;
}

function isTensorLike(value: unknown): value is TensorLike {
  return (
    isRecord(value) && (typeof value.sigmoid === "function" || typeof value.tolist === "function")
  );
}
