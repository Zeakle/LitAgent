import { buildExecutionPresentation, summarizeExecutionTask } from "./execution-model.js?v=15.10.6.431";
import { buildBusinessFlow, phaseRecords } from "./execution-flow.js?v=15.10.6.431";

import { businessInput, businessOutput, draftText, renderBusinessContent } from "./execution-content.js?v=15.10.6.431";
import { buildEvidenceLedger } from "./evidence-model.js?v=15.10.6.431";

function appendField(list, label, value) {
  if (value == null || value === "") return;
  const term = document.createElement("dt");
  term.textContent = label;
  const description = document.createElement("dd");
  description.textContent = String(value);
  list.append(term, description);
}

function appendJson(container, title, value, open = false) {
  const details = document.createElement("details");
  details.className = "inspector-json";
  details.open = open;
  const summary = document.createElement("summary");
  summary.textContent = title;
  const pre = document.createElement("pre");
  pre.textContent = JSON.stringify(value, null, 2);
  details.append(summary, pre);
  container.append(details);
}

export function relatedCalls(nodes, taskId) {
  return Object.entries(nodes || {}).filter(([, node]) => node && node.kind !== "worker"
    && ((node.task_id === taskId && node.kind !== "subspan") || node.input?.parent_task_id === taskId
      || node.metadata?.parent_task_id === taskId));
}

export function renderExecutionInspector(container, selection, t, { view, onSelect, onOpenReport } = {}) {
  if (!["node", "observation", "event"].includes(selection?.kind)) return false;
  let value = selection.value || {};
  if (value.role && view) value = buildBusinessFlow(buildExecutionPresentation(view), view.graph?.nodes)
    .nodes.find(node => node.id === selection.id) || value;
  const taskId = value.taskId || value.task_id || value.task?.task_id || value.execution?.task_id || value.id || selection.id;
  const nodes = view?.graph?.nodes || {};
  const recorded = nodes[selection.id] || nodes[`worker:${taskId}`];
  const execution = recorded || value.execution || (selection.kind === "event" ? value.data
    : value.input !== undefined ? value : value.raw) || value;
  const list = document.createElement("dl");
  appendField(list, t("role"), value.labelKey ? t(value.labelKey) : value.name || value.kind);
  appendField(list, t("flowRound"), value.round);
  appendField(list, t("status"), execution.status || value.status);
  const duration = execution.elapsed_ms ?? execution.duration_ms ?? execution.durationMs ?? value.durationMs;
  appendField(
    list,
    t("elapsed"),
    duration == null ? null : `${(Number(duration) / 1000).toFixed(1)}${t("secondsShort")}`,
  );
  appendField(list, "Task ID", execution.task_id || taskId);
  appendField(list, "Operation ID", value.operationId || execution.operation_id);
  appendField(list, t("error"), value.error || execution.error);
  container.append(list);

  if (value.role) {
    const note = document.createElement("p");
    note.textContent = t(value.execution ? "flowThisRound" : "flowStageSummaryOnly");
    container.append(note);
    for (const record of value.roundRecords) {
      const button = document.createElement("button"); button.type = "button"; button.className = "call-inspect-button";
      button.textContent = `${record.name} · ${record.status}`;
      button.addEventListener("click", () => onSelect?.({ kind: "observation", id: record.id, value: record }, button));
      container.append(button);
    }
    const parent = document.createElement("button"); parent.type = "button"; parent.className = "call-inspect-button";
    parent.textContent = t("flowOpenParent");
    parent.addEventListener("click", () => onSelect?.({ kind: "node", id: value.parentTaskId,
      value: { id: value.parentTaskId, execution: value.parentExecution } }, parent));
    container.append(parent);
  }

  if ((value.task?.observation?.kind === "delivery" || value.kind === "delivery") && view?.report?.survey && onOpenReport) {
    const report = document.createElement("button"); report.type = "button";
    report.className = "call-inspect-button"; report.textContent = t("openReport");
    report.addEventListener("click", onOpenReport); container.append(report);
  }
  const summaryFacts = recorded ? summarizeExecutionTask(value.task || {}, recorded) : value.summaryFacts;
  if (summaryFacts?.length) {
    const summary = document.createElement("section");
    summary.className = "inspector-summary";
    const heading = document.createElement("h3");
    heading.textContent = t("resultSummary");
    const facts = document.createElement("ul");
    for (const fact of summaryFacts) {
      const item = document.createElement("li");
      item.textContent = `${t(fact.labelKey)}: ${Number(fact.value).toLocaleString()}`;
      item.title = fact.sourcePath || "";
      facts.append(item);
    }
    summary.append(heading, facts);
    container.append(summary);
  }
  const sourceQuery = value.task?.observation?.sourceQuery || value.sourceQuery;
  if (sourceQuery) {
    const context = document.createElement("dl");
    appendField(context, t("flowOriginalQuery"), sourceQuery);
    container.append(context);
  }
  const input = businessInput(execution, value.task), output = businessOutput(execution);
  const role = value.role || value.task?.agent_type
    || (/^(synthesis|revision)(\.|$)/.test(execution.name || "") ? "synthesis" : null);
  const round = value.round || execution.input?.round || execution.metadata?.round;
  const parentTaskId = value.parentTaskId || execution.input?.parent_task_id || execution.metadata?.parent_task_id;
  let previousDraft = typeof input?.previous_draft === "string" ? input.previous_draft : null;
  if (previousDraft == null && role === "synthesis" && parentTaskId && round > 1 && view) {
    const previous = phaseRecords(buildExecutionPresentation(view), parentTaskId, "synthesis")
      .filter(record => Number(record.input?.round ?? record.raw?.metadata?.round ?? /\.r(\d+)$/.exec(record.name)?.[1]) === round - 1).at(-1);
    previousDraft = draftText(previous?.output);
  }
  renderBusinessContent(container, {
    input, output, role, previousDraft,
    diffUnavailable: round === 1 ? "ioFirstDraft" : "ioNoPreviousDraft",
    diffMode: selection.diffMode === true, onDiffMode: mode => { selection.diffMode = mode; },
    onEvidence: (id, button, source) => {
      const item = buildEvidenceLedger(view || {}).find(entry => String(entry.evidence_id).replace(/^E:/, "") === id.replace(/^E:/, ""));
      onSelect?.({ kind: "evidence", id, value: item || source || { evidence_id: id } }, button);
    },
  }, t);
  appendJson(container, t("ioRawInput"), input ?? t("notRecorded"));
  appendJson(container, t("ioRawOutput"), output ?? t("notRecorded"));
  if (selection.kind === "node" || execution.kind === "subspan") {
    const calls = relatedCalls(nodes, taskId);
    const details = document.createElement("details"); details.className = "inspector-json";
    const heading = document.createElement("summary"); heading.textContent = `${t("relatedCalls")} (${calls.length})`;
    details.append(heading);
    if (!calls.length) {
      const note = document.createElement("p"); note.textContent = t("noRelatedCalls"); details.append(note);
    }
    for (const [id, node] of calls) {
      const button = document.createElement("button"); button.type = "button"; button.className = "call-inspect-button";
      button.textContent = `${node.name || node.kind} · ${node.status || "unknown"} · ${node.operation_id || id}`;
      button.addEventListener("click", () => onSelect?.({ kind: "observation", id, value: node }, button));
      details.append(button);
    }
    container.append(details);
  }
  appendJson(container, t("rawRecord"), value.role
    ? value.execution?.raw ?? t("notRecorded") : recorded || execution.raw || execution);
  return true;
}
