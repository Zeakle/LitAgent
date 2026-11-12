function appendField(list, label, value) {
  if (value == null || value === "") return;
  const term = document.createElement("dt");
  term.textContent = label;
  const description = document.createElement("dd");
  description.textContent = String(value);
  list.append(term, description);
}

function appendJson(container, title, value) {
  if (value == null) return;
  const details = document.createElement("details");
  details.className = "inspector-json";
  const summary = document.createElement("summary");
  summary.textContent = title;
  const pre = document.createElement("pre");
  pre.textContent = JSON.stringify(value, null, 2);
  details.append(summary, pre);
  container.append(details);
}

export function renderExecutionInspector(container, selection, t) {
  if (!["node", "observation", "event"].includes(selection?.kind)) return false;
  const value = selection.value || {};
  const execution = value.execution || value.raw || value;
  const list = document.createElement("dl");
  appendField(list, t("role"), value.labelKey ? t(value.labelKey) : value.name || value.kind);
  appendField(list, t("status"), value.status || execution.status);
  const duration = value.durationMs ?? execution.elapsed_ms ?? execution.duration_ms;
  appendField(
    list,
    t("elapsed"),
    duration == null ? null : `${(Number(duration) / 1000).toFixed(1)}${t("secondsShort")}`,
  );
  appendField(list, "Task ID", value.id || value.taskId || execution.task_id);
  appendField(list, "Operation ID", value.operationId || execution.operation_id);
  appendField(list, t("error"), value.error || execution.error);
  container.append(list);

  if (value.summaryFacts?.length) {
    const summary = document.createElement("section");
    summary.className = "inspector-summary";
    const heading = document.createElement("h3");
    heading.textContent = t("resultSummary");
    const facts = document.createElement("ul");
    for (const fact of value.summaryFacts) {
      const item = document.createElement("li");
      item.textContent = `${t(fact.labelKey)}: ${Number(fact.value).toLocaleString()}`;
      item.title = fact.sourcePath || "";
      facts.append(item);
    }
    summary.append(heading, facts);
    container.append(summary);
  }
  appendJson(container, t("input"), execution.input);
  appendJson(container, t("output"), execution.output);
  appendJson(container, t("rawRecord"), selection.value);
  return true;
}
