function sourceLabel(item, t) {
  if (item.source === "interactive") return t("interactive");
  if (item.source === "sample") return t("sample");
  return t("runArtifact");
}

function readableTime(item, locale) {
  const value = item.completedAt || item.updatedAt || item.startedAt;
  if (!value) return "";
  const date = new Date(value);
  return Number.isNaN(date.valueOf()) ? "" : date.toLocaleString(locale, { month: "short", day: "numeric", hour: "2-digit", minute: "2-digit" });
}

export function renderHistory(container, runs, selectedKey, query, locale, t, onSelect) {
  container.replaceChildren();
  const needle = query.trim().toLocaleLowerCase();
  const visible = runs.filter((run) => !needle || `${run.query} ${run.runId}`.toLocaleLowerCase().includes(needle));
  for (const run of visible) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "run-item";
    button.dataset.active = String(run.key === selectedKey);
    button.dataset.status = run.status;
    const heading = document.createElement("strong");
    heading.textContent = run.query || run.runId;
    const meta = document.createElement("span");
    const source = document.createElement("b");
    source.textContent = sourceLabel(run, t);
    const time = document.createElement("time");
    time.textContent = readableTime(run, locale);
    meta.append(source, time);
    const status = document.createElement("small");
    status.textContent = run.status;
    button.append(heading, meta, status);
    button.addEventListener("click", () => onSelect(run.key));
    container.append(button);
  }
  if (!visible.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = t("noRun");
    container.append(empty);
  }
}
