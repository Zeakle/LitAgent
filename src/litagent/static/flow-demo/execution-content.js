import { safePaperUrl } from "./evidence-model.js?v=15.10.6.431";

// Business projections are read-only. Raw payloads remain available separately.
const object = value => value && typeof value === "object" && !Array.isArray(value);
const pick = (value, keys) => Object.fromEntries(keys.filter(key => value?.[key] !== undefined).map(key => [key, value[key]]));
const COMMON = ["query", "source", "mode", "report_language", "quality_policy", "criteria", "threshold", "top_k", "candidate_k", "retrieval_mode", "max_representative_chunks", "max_sub_queries", "coverage_constraints", "named_methods"];
const INTERNAL = new Set(["writer_document", "budget_envelope", "targeted_repair_state", "native_wire_input_projection"]);

export function businessInput(execution, task) {
  let input = execution?.input ?? task?.input ?? task?.input_data;
  if (execution?.kind === "subspan" && object(input)) {
    if (Object.hasOwn(input, "input")) return input.input;
    input = Object.fromEntries(Object.entries(input).filter(([key]) => !["name", "round", "phase", "parent_task_id"].includes(key)));
    if (!Object.keys(input).length) return null;
  }
  return input;
}

export function businessOutput(execution) {
  const output = execution?.output;
  if (execution?.kind === "subspan" && object(output)
    && Object.keys(output).every(key => ["error", "error_type"].includes(key))) return null;
  return output;
}

export function draftText(output) {
  if (typeof output === "string") return output;
  for (const key of ["draft", "final_draft", "survey"]) if (typeof output?.[key] === "string") return output[key];
  return null;
}

export function contentProjection(input, output, role) {
  let shownInput = input, shownOutput = output;
  if (object(input)) {
    const fields = [...COMMON, "previous_draft", "draft", "survey", "source_report_sha256", "report_sha256", "review", "previous_review", "review_feedback", "targets", "evidence_selection", "fact_cards", "coverage_requirements", "extractions", "upstream_results", "reused", "quality"];
    const selected = pick(input, fields);
    if (Object.keys(selected).length) shownInput = selected;
  }
  if (role === "synthesis" && object(output)) shownOutput = pick(output, [
    "report_sha256", "writing_bindings", "evidence_selection", "coverage_matrix", "promoted", "status", "reason_code", "degradation_reasons",
  ]);
  if (role === "synthesis" && typeof output === "string") shownOutput = null;
  return { input: shownInput, output: shownOutput };
}

// Bounded line LCS; the large-document fallback still preserves every byte.
export function diffLines(before, after) {
  const lines = text => text.match(/[^\n]*\n|[^\n]+$/g) || [];
  const left = lines(before), right = lines(after), chunks = [];
  const add = (kind, text) => {
    if (!text) return;
    if (chunks.at(-1)?.kind === kind) chunks.at(-1).text += text;
    else chunks.push({ kind, text });
  };
  let start = 0, end = 0;
  while (start < left.length && start < right.length && left[start] === right[start]) add("same", left[start++]);
  while (end < left.length - start && end < right.length - start && left.at(-end - 1) === right.at(-end - 1)) end++;
  const a = left.slice(start, left.length - end), b = right.slice(start, right.length - end);
  if ((a.length + 1) * (b.length + 1) > 1000000) {
    add("removed", a.join("")); add("added", b.join(""));
  } else {
    const width = b.length + 1, grid = new Uint32Array((a.length + 1) * width);
    for (let i = a.length - 1; i >= 0; i--) for (let j = b.length - 1; j >= 0; j--) {
      grid[i * width + j] = a[i] === b[j] ? 1 + grid[(i + 1) * width + j + 1]
        : Math.max(grid[(i + 1) * width + j], grid[i * width + j + 1]);
    }
    let i = 0, j = 0;
    while (i < a.length || j < b.length) {
      if (i < a.length && j < b.length && a[i] === b[j]) { add("same", a[i++]); j++; }
      else if (i < a.length && (j === b.length || grid[(i + 1) * width + j] >= grid[i * width + j + 1])) add("removed", a[i++]);
      else add("added", b[j++]);
    }
  }
  if (end) add("same", left.slice(left.length - end).join(""));
  return chunks;
}

function label(key, t) {
  const translated = t(`io_${key}`);
  return translated === `io_${key}` ? key.replaceAll("_", " ") : translated;
}

function readable(container, value, t, onEvidence, depth = 0) {
  if (value == null) {
    const note = document.createElement("p"); note.textContent = t("notRecorded"); container.append(note); return;
  }
  if (Array.isArray(value)) {
    if (!value.length) { const p = document.createElement("p"); p.textContent = t("ioEmpty"); container.append(p); }
    value.forEach((item, index) => {
      const row = document.createElement(object(item) ? "details" : "div");
      row.className = "execution-entry";
      if (object(item)) {
        const title = document.createElement("summary");
        title.textContent = `${index + 1}. ${item.title || item.paper_title || item.text || item.claim || item.issue || item.evidence_id || item.paper_id || t("ioEntry")}`;
        row.append(title);
      }
      readable(row, item, t, onEvidence, depth + 1); container.append(row);
    });
  } else if (object(value)) {
    const entries = Object.entries(value).filter(([key]) => !INTERNAL.has(key));
    if (!entries.length) { const p = document.createElement("p"); p.textContent = t("ioEmpty"); container.append(p); }
    for (const [key, item] of entries) {
      const collapsible = ["upstream_results", "evidence_selection", "fact_cards", "writing_bindings", "extractions", "selected_items", "previous_draft", "draft", "survey"].includes(key);
      const block = document.createElement(collapsible ? "details" : "div"); block.className = "execution-field";
      const title = document.createElement(collapsible ? "summary" : "strong"); title.textContent = label(key, t); block.append(title);
      if ((key === "evidence_id" || key === "source_unit_id") && typeof item === "string" && onEvidence) {
        const button = document.createElement("button"); button.type = "button";
        button.className = "call-inspect-button"; button.textContent = item;
        button.addEventListener("click", () => onEvidence(item, button, value)); block.append(button);
      } else if (key === "url" && safePaperUrl(item)) {
        const link = document.createElement("a"); link.href = safePaperUrl(item);
        link.target = "_blank"; link.rel = "noopener noreferrer"; link.textContent = t("openPaper"); block.append(link);
      } else readable(block, item, t, onEvidence, depth + 1);
      container.append(block);
    }
  } else {
    const p = document.createElement("p"); p.className = "execution-text";
    p.textContent = typeof value === "boolean" ? t(value ? "ioYes" : "ioNo") : String(value);
    container.append(p);
  }
}

export function renderBusinessContent(container, { input, output, role, previousDraft, diffUnavailable, diffMode = false, onDiffMode, onEvidence }, t) {
  const projection = contentProjection(input, output, role);
  for (const [direction, data] of Object.entries(projection)) {
    const section = document.createElement("section"); section.className = "execution-business";
    section.dataset.direction = direction;
    const title = document.createElement("h3"); title.textContent = t(direction); section.append(title);
    if (direction === "output" && role === "synthesis") {
      const draft = draftText(output);
      if (draft != null) {
        const controls = document.createElement("div"); controls.className = "execution-draft-controls";
        const full = document.createElement("button"), changes = document.createElement("button");
        full.type = changes.type = "button";
        full.textContent = t("ioFullDraft"); changes.textContent = t("ioDiff");
        changes.disabled = previousDraft == null;
        const body = document.createElement("div"); body.className = "execution-document";
        const render = mode => {
          const changed = mode && previousDraft != null;
          full.setAttribute("aria-pressed", String(!changed)); changes.setAttribute("aria-pressed", String(changed));
          body.replaceChildren();
          if (changed) for (const chunk of diffLines(previousDraft, draft)) {
            const piece = document.createElement(chunk.kind === "added" ? "ins" : chunk.kind === "removed" ? "del" : "span");
            piece.textContent = chunk.text; body.append(piece);
          } else body.textContent = draft;
        };
        full.addEventListener("click", () => { onDiffMode?.(false); render(false); });
        changes.addEventListener("click", () => { onDiffMode?.(true); render(true); });
        render(diffMode); controls.append(full, changes); section.append(controls);
        const note = document.createElement("p"); note.className = "execution-diff-note";
        note.textContent = t(previousDraft == null ? diffUnavailable || "ioNoPreviousDraft" : "ioDiffHelp");
        section.append(note, body);
      } else readable(section, null, t);
    }
    if (data != null || role !== "synthesis" || direction !== "output") readable(section, data, t, onEvidence);
    container.append(section);
  }
}
