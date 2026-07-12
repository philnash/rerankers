import {
  MODEL_OPTIONS,
  nonEmptyDocuments,
  RerankerCache,
  type DemoDocument,
  type DemoRankingResult,
  validateRankingInput,
} from "./ranking.js";

const DEFAULT_DOCUMENTS: DemoDocument[] = [
  {
    id: "document-1",
    text: "Code splitting lets a JavaScript application load only the code needed for the current page. Dynamic imports and route-based chunks can significantly reduce the initial bundle size.",
  },
  {
    id: "document-2",
    text: "JavaScript is a programming language commonly used to add interactive behaviour to websites and build server-side applications with Node.js.",
  },
  {
    id: "document-3",
    text: "Compressing images, serving modern formats such as WebP or AVIF, and lazy-loading below-the-fold media can improve page load performance.",
  },
];

const cache = new RerankerCache();
let documents = DEFAULT_DOCUMENTS;
let isRanking = false;

const form = getElement<HTMLFormElement>("#ranking-form");
const modelSelect = getElement<HTMLSelectElement>("#model");
const modelNote = getElement<HTMLElement>("#model-note");
const queryInput = getElement<HTMLTextAreaElement>("#query");
const documentList = getElement<HTMLElement>("#document-list");
const errorPanel = getElement<HTMLElement>("#form-error");
const rankButton = getElement<HTMLButtonElement>("#rank-button");
const status = getElement<HTMLElement>("#status");
const results = getElement<HTMLElement>("#results");

renderDocuments();

modelSelect.addEventListener("change", () => {
  modelNote.textContent = MODEL_OPTIONS.find(({ id }) => id === modelSelect.value)?.note ?? "";
});

getElement<HTMLButtonElement>("#add-document").addEventListener("click", () => {
  documents = [...documents, { id: crypto.randomUUID(), text: "" }];
  renderDocuments();
  documentList.querySelector<HTMLTextAreaElement>(".document-card:last-child textarea")?.focus();
});

form.addEventListener("submit", (event) => {
  event.preventDefault();
  void runRanking();
});

async function runRanking(): Promise<void> {
  if (isRanking) return;

  const validationError = validateRankingInput(queryInput.value, documents);
  if (validationError) {
    showError(validationError);
    return;
  }

  isRanking = true;
  showError(null);
  setBusy(true, "Loading model…");

  try {
    const reranker = await cache.get(modelSelect.value);
    const rankableDocuments = nonEmptyDocuments(documents);
    setBusy(
      true,
      `Ranking ${rankableDocuments.length} document${rankableDocuments.length === 1 ? "" : "s"}…`,
    );
    const ranked = await reranker.rank(queryInput.value.trim(), rankableDocuments);
    renderResults(ranked);
    status.textContent = `${ranked.length} document${ranked.length === 1 ? "" : "s"} ranked with ${selectedModelLabel()}.`;
  } catch (error: unknown) {
    showError(error instanceof Error ? error.message : "The ranking could not be completed.");
    status.textContent = "Ranking failed. Your previous result has been preserved.";
  } finally {
    isRanking = false;
    setBusy(false);
  }
}

function renderDocuments(): void {
  documentList.innerHTML = documents
    .map(
      ({ id, text }, index) => `
        <article class="document-card" data-id="${id}">
          <div class="document-meta">
            <label for="document-${id}">Document ${String(index + 1).padStart(2, "0")}</label>
            <button class="remove-button" type="button" aria-label="Remove document ${index + 1}" ${documents.length === 1 ? "disabled" : ""}>Remove</button>
          </div>
          <textarea id="document-${id}" rows="4">${escapeHtml(text)}</textarea>
        </article>
      `,
    )
    .join("");

  documentList.querySelectorAll<HTMLElement>(".document-card").forEach((card) => {
    const id = card.dataset.id;
    const textarea = card.querySelector<HTMLTextAreaElement>("textarea");
    const removeButton = card.querySelector<HTMLButtonElement>(".remove-button");
    if (!id || !textarea || !removeButton) return;

    textarea.addEventListener("input", () => {
      documents = documents.map((document) =>
        document.id === id ? { ...document, text: textarea.value } : document,
      );
    });
    removeButton.addEventListener("click", () => {
      documents = documents.filter((document) => document.id !== id);
      renderDocuments();
    });
  });
}

function renderResults(ranked: readonly DemoRankingResult[]): void {
  const maxScore = Math.max(...ranked.map(({ score }) => score), 0);
  const minScore = Math.min(...ranked.map(({ score }) => score), 0);
  const range = maxScore - minScore || 1;

  results.innerHTML = ranked
    .map(({ document, index, score }, rank) => {
      const normalizedScore = Math.max(0.04, (score - minScore) / range);
      return `
        <article class="result-card" style="--delay: ${rank * 70}ms">
          <div class="result-rank">${String(rank + 1).padStart(2, "0")}</div>
          <div class="result-content">
            <div class="result-meta">
              <span>Original document ${String(index + 1).padStart(2, "0")}</span>
              <strong>${formatScore(score)}</strong>
            </div>
            <p>${escapeHtml(document.text)}</p>
            <div class="score-track" aria-label="Relevance score ${formatScore(score)}">
              <span style="width: ${normalizedScore * 100}%"></span>
            </div>
          </div>
        </article>
      `;
    })
    .join("");
}

function setBusy(busy: boolean, message = ""): void {
  rankButton.disabled = busy;
  rankButton.classList.toggle("is-loading", busy);
  rankButton.querySelector<HTMLElement>(".rank-button-label")!.textContent = busy
    ? message
    : "Rank documents";
  form.setAttribute("aria-busy", String(busy));
  if (busy) status.textContent = message;
}

function showError(message: string | null): void {
  errorPanel.hidden = message === null;
  errorPanel.textContent = message ?? "";
}

function selectedModelLabel(): string {
  return MODEL_OPTIONS.find(({ id }) => id === modelSelect.value)?.label ?? modelSelect.value;
}

function formatScore(score: number): string {
  return score.toFixed(4);
}

function escapeHtml(value: string): string {
  const element = document.createElement("div");
  element.textContent = value;
  return element.innerHTML;
}

function getElement<TElement extends Element>(selector: string): TElement {
  const element = document.querySelector<TElement>(selector);
  if (!element) throw new Error(`Element not found: ${selector}`);
  return element;
}
