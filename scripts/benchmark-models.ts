import { spawn } from "node:child_process";
import { writeFile } from "node:fs/promises";
import { performance } from "node:perf_hooks";
import { fileURLToPath } from "node:url";

import { env } from "@huggingface/transformers";

import { benchmarkReportPath, modelCacheDirectory, modelIds } from "./model-benchmark-config.ts";

const distEntryPoint = "../dist/index.js";
const { Reranker } = (await import(distEntryPoint)) as typeof import("../src/index.js");

type Success = {
  model: string;
  status: "ok";
  loadAndWarmupSeconds: number;
  rerankSeconds: number;
  topDocument: string;
  scores: string;
};

type Failure = {
  model: string;
  status: "error";
  elapsedSeconds: number;
  reason: string;
  error: string;
};

type BenchmarkResult = Success | Failure;

const resultMarker = "__RERANKERS_BENCHMARK_RESULT__";
const scriptPath = fileURLToPath(import.meta.url);
const workerModel = readWorkerModel(process.argv);

if (workerModel !== undefined) {
  await runWorker(workerModel);
} else {
  await runCoordinator();
}

async function runCoordinator(): Promise<void> {
  console.log(`Testing ${modelIds.length} models sequentially.`);
  console.log(`Model cache: ${modelCacheDirectory}`);
  console.log(
    "Each model runs in a fresh process so its memory is released before the next model.\n",
  );

  const results: BenchmarkResult[] = [];

  for (const [index, model] of modelIds.entries()) {
    console.log(`[${index + 1}/${modelIds.length}] ${model}`);
    const result = await runModelProcess(model);
    results.push(result);
    console.log(
      result.status === "ok"
        ? `  ok: load + warm-up ${formatSeconds(result.loadAndWarmupSeconds)}, rerank ${formatSeconds(result.rerankSeconds)}`
        : `  error in ${formatSeconds(result.elapsedSeconds)}: ${result.reason}`,
    );
  }

  console.log("\nResults\n");
  console.table(
    results.map((result) =>
      result.status === "ok"
        ? {
            model: result.model,
            status: result.status,
            "load + warm-up (s)": result.loadAndWarmupSeconds.toFixed(1),
            "rerank (s)": result.rerankSeconds.toFixed(3),
            topDocument: result.topDocument,
            scores: result.scores,
            reason: "",
            error: "",
          }
        : {
            model: result.model,
            status: result.status,
            "load + warm-up (s)": result.elapsedSeconds.toFixed(1),
            "rerank (s)": "",
            topDocument: "",
            scores: "",
            reason: result.reason,
            error: result.error,
          },
    ),
  );

  await writeHtmlReport(results);
  console.log(`\nHTML report: ${benchmarkReportPath}`);

  const failures = results.filter((result) => result.status === "error");
  if (failures.length > 0) {
    console.log("\nFailure details\n");
    for (const failure of failures) {
      console.log(`${failure.model}\n  Reason: ${failure.reason}\n  Error: ${failure.error}\n`);
    }
    process.exitCode = 1;
  }
}

function runModelProcess(model: string): Promise<BenchmarkResult> {
  return new Promise((resolveResult) => {
    const startedAt = performance.now();
    const child = spawn(process.execPath, [scriptPath, "--worker", model], {
      env: process.env,
      stdio: ["ignore", "pipe", "pipe"],
    });

    let stdout = "";
    let stderr = "";

    child.stdout.setEncoding("utf8");
    child.stderr.setEncoding("utf8");
    child.stdout.on("data", (chunk: string) => {
      stdout += chunk;
    });
    child.stderr.on("data", (chunk: string) => {
      stderr += chunk;
    });

    child.on("error", (error) => {
      resolveResult(makeFailure(model, startedAt, error));
    });

    child.on("close", (code, signal) => {
      const resultLine = [...stdout.split("\n")]
        .reverse()
        .find((line) => line.startsWith(resultMarker));

      if (resultLine !== undefined) {
        try {
          const parsed: unknown = JSON.parse(resultLine.slice(resultMarker.length));
          if (!isBenchmarkResult(parsed)) {
            throw new TypeError("Worker returned an invalid benchmark result");
          }
          resolveResult(
            parsed.status === "error" ? appendWorkerOutput(parsed, stdout, stderr) : parsed,
          );
          return;
        } catch (error) {
          resolveResult(makeFailure(model, startedAt, error));
          return;
        }
      }

      const detail = stderr.trim() || stdout.trim() || `Worker exited with code ${String(code)}`;
      const suffix = signal === null ? "" : ` (signal ${signal})`;
      resolveResult(
        makeFailure(model, startedAt, new Error(`${detail}${suffix}`), "Worker process failed"),
      );
    });
  });
}

async function runWorker(model: string): Promise<void> {
  env.cacheDir = modelCacheDirectory;
  const startedAt = performance.now();

  try {
    const reranker = await Reranker.create({
      model,
      transformerOptions: { dtype: "q8" },
    });
    await reranker.rank("Which planet is known as the Red Planet?", [
      "Mars is known as the Red Planet.",
    ]);
    const loadedAt = performance.now();
    const results = await reranker.rank("Which planet is known as the Red Planet?", [
      "Venus has a dense atmosphere and is the hottest planet.",
      "Mars is known as the Red Planet because iron minerals in its soil oxidize.",
      "Jupiter is the largest planet and has a persistent Great Red Spot.",
      "Saturn is recognizable by its extensive ring system.",
    ]);
    const rankedAt = performance.now();

    const result: Success = {
      model,
      status: "ok",
      loadAndWarmupSeconds: (loadedAt - startedAt) / 1_000,
      rerankSeconds: (rankedAt - loadedAt) / 1_000,
      topDocument: String(results[0]?.document ?? ""),
      scores: results.map(({ index, score }) => `${index}:${score.toFixed(5)}`).join(" "),
    };
    emitResult(result);
  } catch (error) {
    emitResult(makeFailure(model, startedAt, error));
  }
}

function makeFailure(
  model: string,
  startedAt: number,
  error: unknown,
  reason = diagnoseError(error),
): Failure {
  return {
    model,
    status: "error",
    elapsedSeconds: elapsedSeconds(startedAt),
    reason,
    error: formatFullError(error),
  };
}

function diagnoseError(error: unknown): string {
  const message = formatErrorChain(error).toLowerCase();

  if (message.includes("unauthorized") || message.includes("401") || message.includes("gated")) {
    return "Model access requires authentication or license acceptance";
  }
  if (
    message.includes("404") ||
    message.includes("not found") ||
    message.includes("no available file")
  ) {
    return "Model repository or required ONNX/tokenizer file was not found";
  }
  if (message.includes("unsupported model") || message.includes("unsupported architecture")) {
    return "Model architecture is not supported by Transformers.js";
  }
  if (message.includes("sequenceclassification") || message.includes("sequence classification")) {
    return "Model is incompatible with rerankers' sequence-classification strategy";
  }
  if (message.includes("onnx") || message.includes("protobuf") || message.includes("opset")) {
    return "ONNX model could not be loaded by the current runtime";
  }
  if (message.includes("out of memory") || message.includes("allocation failed")) {
    return "Insufficient memory for this model and dtype";
  }
  if (
    message.includes("fetch failed") ||
    message.includes("network") ||
    message.includes("enotfound") ||
    message.includes("timeout")
  ) {
    return "Network or model-download failure";
  }
  if (message.includes("score") || message.includes("logits") || message.includes("tensor")) {
    return "Model output is incompatible with rerankers' score extraction";
  }

  return "Unknown load or inference failure; inspect the reported error";
}

function formatErrorChain(error: unknown): string {
  const messages: string[] = [];
  let current: unknown = error;

  while (current !== undefined && current !== null) {
    if (current instanceof Error) {
      messages.push(`${current.name}: ${current.message}`);
      current = current.cause;
    } else {
      messages.push(
        typeof current === "string" ||
          typeof current === "number" ||
          typeof current === "boolean" ||
          typeof current === "bigint"
          ? String(current)
          : JSON.stringify(current),
      );
      break;
    }
  }

  return messages.join(" <- ");
}

function formatFullError(error: unknown): string {
  const errors: string[] = [];
  let current: unknown = error;
  let causeNumber = 0;

  while (current !== undefined && current !== null) {
    const heading = causeNumber === 0 ? "" : `Caused by (${causeNumber}):\n`;

    if (current instanceof Error) {
      errors.push(`${heading}${current.stack ?? `${current.name}: ${current.message}`}`);
      current = current.cause;
    } else {
      errors.push(`${heading}${formatUnknownError(current)}`);
      break;
    }

    causeNumber += 1;
  }

  return errors.join("\n\n");
}

function formatUnknownError(error: unknown): string {
  if (
    typeof error === "string" ||
    typeof error === "number" ||
    typeof error === "boolean" ||
    typeof error === "bigint"
  ) {
    return String(error);
  }

  try {
    return JSON.stringify(error, null, 2);
  } catch {
    return Object.prototype.toString.call(error);
  }
}

function appendWorkerOutput(failure: Failure, stdout: string, stderr: string): Failure {
  const workerStdout = stdout
    .split("\n")
    .filter((line) => !line.startsWith(resultMarker))
    .join("\n")
    .trim();
  const workerStderr = stderr.trim();
  const output = [
    failure.error,
    workerStderr === "" ? "" : `Worker stderr:\n${workerStderr}`,
    workerStdout === "" ? "" : `Worker stdout:\n${workerStdout}`,
  ].filter((part) => part !== "");

  return { ...failure, error: output.join("\n\n") };
}

function emitResult(result: BenchmarkResult): void {
  console.log(`${resultMarker}${JSON.stringify(result)}`);
}

function readWorkerModel(args: string[]): string | undefined {
  const workerIndex = args.indexOf("--worker");
  return workerIndex === -1 ? undefined : args[workerIndex + 1];
}

function elapsedSeconds(startedAt: number): number {
  return (performance.now() - startedAt) / 1_000;
}

function formatSeconds(seconds: number): string {
  return `${seconds.toFixed(1)}s`;
}

function isBenchmarkResult(value: unknown): value is BenchmarkResult {
  if (typeof value !== "object" || value === null) {
    return false;
  }

  const candidate = value as Record<string, unknown>;
  return (
    typeof candidate.model === "string" &&
    ((candidate.status === "ok" &&
      typeof candidate.loadAndWarmupSeconds === "number" &&
      typeof candidate.rerankSeconds === "number") ||
      (candidate.status === "error" && typeof candidate.elapsedSeconds === "number"))
  );
}

async function writeHtmlReport(results: BenchmarkResult[]): Promise<void> {
  const successes = results.filter((result) => result.status === "ok").length;
  const failures = results.length - successes;
  const generatedAt = new Date();
  const rows = results
    .map((result) => {
      if (result.status === "ok") {
        return `<tr>
          <td><code>${escapeHtml(result.model)}</code></td>
          <td><span class="status status-ok">OK</span></td>
          <td class="number">${result.loadAndWarmupSeconds.toFixed(1)}</td>
          <td class="number">${result.rerankSeconds.toFixed(3)}</td>
          <td>${escapeHtml(result.topDocument)}</td>
          <td><code>${escapeHtml(result.scores)}</code></td>
          <td></td>
        </tr>`;
      }

      return `<tr>
        <td><code>${escapeHtml(result.model)}</code></td>
        <td><span class="status status-error">Error</span></td>
        <td class="number">${result.elapsedSeconds.toFixed(1)}</td>
        <td class="number">—</td>
        <td>—</td>
        <td>—</td>
        <td>${escapeHtml(result.reason)}</td>
      </tr>`;
    })
    .join("\n");
  const errorDetails = results
    .filter((result): result is Failure => result.status === "error")
    .map(
      (failure) => `<details>
        <summary><code>${escapeHtml(failure.model)}</code> — ${escapeHtml(failure.reason)}</summary>
        <pre>${escapeHtml(failure.error)}</pre>
      </details>`,
    )
    .join("\n");

  const html = `<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8">
  <meta name="viewport" content="width=device-width, initial-scale=1">
  <title>Rerankers model benchmark</title>
  <style>
    :root { color-scheme: light; --ink: #18201d; --muted: #64706a; --paper: #f5f2e9; --line: #c9cec7; --ok: #17633a; --error: #a12d22; }
    * { box-sizing: border-box; }
    body { margin: 0; background: var(--paper); color: var(--ink); font: 15px/1.5 Georgia, "Times New Roman", serif; }
    main { width: min(1500px, calc(100% - 32px)); margin: 48px auto; }
    header { display: flex; align-items: end; justify-content: space-between; gap: 24px; border-bottom: 3px solid var(--ink); padding-bottom: 18px; }
    h1, h2 { margin: 0; font-weight: 600; letter-spacing: -0.025em; }
    h1 { font-size: clamp(2rem, 5vw, 4.5rem); line-height: 0.95; }
    h2 { margin-top: 48px; font-size: 1.6rem; }
    .meta { color: var(--muted); text-align: right; }
    .totals { display: flex; gap: 20px; margin: 20px 0; font-size: 1.1rem; }
    .totals strong { font-size: 1.5rem; }
    .table-wrap { overflow-x: auto; border: 1px solid var(--line); background: #fffdf7; }
    table { width: 100%; border-collapse: collapse; min-width: 1080px; }
    th, td { padding: 11px 12px; border-bottom: 1px solid var(--line); text-align: left; vertical-align: top; }
    th { position: sticky; top: 0; background: var(--ink); color: white; font: 700 0.72rem/1.2 ui-monospace, monospace; letter-spacing: 0.08em; text-transform: uppercase; }
    tbody tr:hover { background: #edf0e7; }
    code, pre { font: 0.82rem/1.45 ui-monospace, SFMono-Regular, Consolas, monospace; }
    .number { text-align: right; font-variant-numeric: tabular-nums; }
    .status { display: inline-block; border: 1px solid currentColor; padding: 2px 7px; font: 700 0.7rem/1.3 ui-monospace, monospace; text-transform: uppercase; }
    .status-ok { color: var(--ok); }
    .status-error { color: var(--error); }
    details { margin-top: 10px; border: 1px solid var(--line); background: #fffdf7; }
    summary { cursor: pointer; padding: 14px; }
    pre { margin: 0; padding: 16px; border-top: 1px solid var(--line); overflow: auto; white-space: pre-wrap; overflow-wrap: anywhere; background: #171c1a; color: #f1eee4; }
    .empty { color: var(--muted); }
    @media (max-width: 700px) { main { margin-top: 24px; } header { display: block; } .meta { margin-top: 16px; text-align: left; } }
    @media print { body { background: white; } main { width: 100%; margin: 0; } th { position: static; } details { break-inside: avoid; } }
  </style>
</head>
<body>
  <main>
    <header>
      <h1>Rerankers<br>model benchmark</h1>
      <div class="meta">Generated ${escapeHtml(generatedAt.toLocaleString())}<br>${results.length} models tested</div>
    </header>
    <div class="totals"><span><strong>${successes}</strong> passed</span><span><strong>${failures}</strong> failed</span></div>
    <div class="table-wrap">
      <table>
        <thead><tr><th>Model</th><th>Status</th><th>Load + warm-up (s)</th><th>Rerank (s)</th><th>Top document</th><th>Scores</th><th>Reason</th></tr></thead>
        <tbody>${rows}</tbody>
      </table>
    </div>
    <h2>Full errors</h2>
    ${errorDetails === "" ? '<p class="empty">No errors were reported.</p>' : errorDetails}
  </main>
</body>
</html>
`;

  await writeFile(benchmarkReportPath, html, "utf8");
}

function escapeHtml(value: string): string {
  return value
    .replaceAll("&", "&amp;")
    .replaceAll("<", "&lt;")
    .replaceAll(">", "&gt;")
    .replaceAll('"', "&quot;")
    .replaceAll("'", "&#39;");
}
