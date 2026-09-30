import { classifyEvent } from "./execution-view.js";

function timestamp(value) {
  const parsed = Date.parse(value || "");
  return Number.isFinite(parsed) ? parsed : null;
}

function eventIdentity(event, index) {
  const source = event.source || "event";
  if (event.sequence != null) return `${source}:sequence:${event.sequence}`;
  if (event.seq != null) return `${source}:seq:${event.seq}`;
  return `${source}:index:${index}`;
}

function eventKind(name) {
  return String(name || "").replace(/\.(start|end|complete|failed|error|cancelled)$/, "");
}

function observationKey(item) {
  if (item.kind === "worker") return `worker:task:${item.taskId || ""}`;
  if (item.operationId) return `${item.kind}:operation:${item.operationId}`;
  return `${item.kind}:task:${item.taskId || ""}`;
}

function eventObservationKey(event) {
  const data = event.data || {};
  const kind = event.event === "worker.input" ? "worker" : eventKind(event.event);
  if (kind === "worker") return `worker:task:${data.task_id || ""}`;
  if (data.operation_id) return `${kind}:operation:${data.operation_id}`;
  return `${kind}:task:${data.task_id || ""}`;
}

function isPairedLifecycleEvent(event, observationKeys) {
  const name = String(event.event || "");
  if (name.endsWith(".retry")) return false;
  const lifecycle = name === "worker.input"
    || /\.(start|end|complete|failed|error|cancelled)$/.test(name);
  return lifecycle && observationKeys.has(eventObservationKey(event));
}

export function buildCallTimeline(
  presentation = {},
  { taskId = null, category = "all" } = {},
) {
  const observations = Array.isArray(presentation.observations)
    ? presentation.observations
    : [];
  const events = Array.isArray(presentation.events) ? presentation.events : [];
  const knownTaskIds = new Set(observations.map((item) => item.taskId).filter(Boolean));
  const observationKeys = new Set(observations.map(observationKey));
  const items = observations.map((item) => ({
    id: item.id,
    recordType: "observation",
    event: `${item.kind}.${item.status}`,
    category: classifyEvent(`${item.kind}.${item.status}`),
    taskId: item.taskId,
    operationId: item.operationId,
    attempt: item.attempt,
    startedAt: item.startedAt,
    completedAt: item.completedAt,
    durationMs: item.durationMs,
    status: item.status,
    orphan: false,
    value: item,
  }));
  const seen = new Set();
  events.forEach((event, index) => {
    const id = eventIdentity(event, index);
    if (seen.has(id)) return;
    seen.add(id);
    // Paired observations already carry the complete lifecycle and payload. Keep
    // retries and unmatched events so failures are visible without duplicate rows.
    if (isPairedLifecycleEvent(event, observationKeys)) return;
    const data = event.data || {};
    const eventTaskId = data.task_id || null;
    items.push({
      id,
      recordType: "event",
      event: event.event || "system",
      category: classifyEvent(event.event || ""),
      taskId: eventTaskId,
      operationId: data.operation_id || null,
      attempt: data.attempt ?? null,
      startedAt: event.at || event.created_at || null,
      completedAt: null,
      durationMs: null,
      status: /failed|error/.test(event.event || "") ? "failed" : null,
      orphan: Boolean(eventTaskId && !knownTaskIds.has(eventTaskId)),
      value: event,
    });
  });
  const filtered = items.filter((item) => {
    if (taskId && item.taskId !== taskId) return false;
    return category === "all" || item.category === category;
  });
  filtered.sort((left, right) => {
    const leftAt = timestamp(left.startedAt);
    const rightAt = timestamp(right.startedAt);
    if (leftAt == null && rightAt == null) return left.id.localeCompare(right.id);
    if (leftAt == null) return 1;
    if (rightAt == null) return -1;
    return leftAt - rightAt || left.id.localeCompare(right.id);
  });
  const starts = filtered.map((item) => timestamp(item.startedAt)).filter(Number.isFinite);
  const ends = filtered
    .flatMap((item) => [timestamp(item.startedAt), timestamp(item.completedAt)])
    .filter(Number.isFinite);
  return {
    items: filtered,
    total: filtered.length,
    range: {
      startedAt: starts.length ? Math.min(...starts) : null,
      completedAt: ends.length ? Math.max(...ends) : null,
    },
  };
}

export function renderCallTimeline(
  container,
  model,
  {
    locale,
    onSelect,
    emptyText,
    pageSize = 200,
    showAllText = "Show all",
    durationUnit = "s",
  } = {},
) {
  container.replaceChildren();
  if (!model.items.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = emptyText || "No events";
    container.append(empty);
    return;
  }
  const formatter = new Intl.DateTimeFormat(locale || "zh-CN", {
    hour: "2-digit",
    minute: "2-digit",
    second: "2-digit",
  });
  const renderItems = (items) => {
    for (const item of items) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "timeline-item";
    button.dataset.category = item.category;
    button.dataset.orphan = String(item.orphan);
    const time = document.createElement("time");
    const at = timestamp(item.startedAt);
    time.textContent = at == null ? "—" : formatter.format(at);
    const body = document.createElement("span");
    const name = document.createElement("strong");
    name.textContent = item.event;
    const meta = document.createElement("small");
    const details = [item.taskId, item.attempt == null ? null : `attempt ${item.attempt}`]
      .filter(Boolean);
    meta.textContent = details.join(" · ");
    body.append(name, meta);
    const duration = document.createElement("b");
    duration.textContent = item.durationMs == null
      ? ""
      : `${(item.durationMs / 1000).toFixed(1)}${durationUnit}`;
    button.append(time, body, duration);
    button.addEventListener("click", () => onSelect?.({
      kind: item.recordType,
      id: item.id,
      value: item.value,
    }, button));
      container.append(button);
    }
  };
  renderItems(model.items.slice(0, pageSize));
  if (model.items.length > pageSize) {
    const more = document.createElement("button");
    more.type = "button";
    more.className = "timeline-show-all";
    more.textContent = `${showAllText} (${model.items.length})`;
    more.addEventListener("click", () => {
      more.remove();
      renderItems(model.items.slice(pageSize));
    });
    container.append(more);
  }
}
