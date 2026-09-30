function statusSummary(state, t) {
  const parts = [];
  const labels = [
    ["running", "statusRunning"],
    ["failed", "statusFailed"],
    ["completed", "statusCompleted"],
    ["skipped", "statusSkipped"],
    ["cancelled", "statusCancelled"],
    ["unknown", "statusUnknown"],
  ];
  for (const [status, label] of labels) {
    if (state?.counts?.[status]) parts.push(`${t(label)} ${state.counts[status]}`);
  }
  if (state?.waitingHuman) parts.unshift(t("statusAwaitingHuman"));
  return parts.join(" · ") || t("statusUnknown");
}

function taskSummary(task, t) {
  const fact = task.summaryFacts?.[0];
  if (!fact) return t("outputNotRecorded");
  return `${t(fact.labelKey)} ${Number(fact.value).toLocaleString()}`;
}

function statusLabel(status, t) {
  const key = `status${String(status || "unknown").replace(
    /(^|_)([a-z])/g,
    (_match, _prefix, letter) => letter.toUpperCase(),
  )}`;
  return t(key);
}

function memberButton(member, t, onSelect) {
  const button = document.createElement("button");
  button.type = "button";
  button.className = "stage-member";
  button.dataset.status = member.status || "unknown";
  const title = document.createElement("strong");
  title.textContent = t(member.labelKey) || member.type || member.id;
  const summary = document.createElement("span");
  summary.textContent = taskSummary(member, t);
  const status = document.createElement("small");
  const elapsed = member.durationMs == null
    ? ""
    : ` · ${(member.durationMs / 1000).toFixed(1)}${t("secondsShort")}`;
  status.textContent = `${statusLabel(member.status, t)}${elapsed}`;
  button.append(title, summary, status);
  button.addEventListener("click", () => onSelect?.({
    kind: "node",
    id: member.id,
    value: member,
  }, button));
  return button;
}

export function renderStageOverview(
  container,
  presentation,
  { expandedStageIds = [], t, onToggle, onSelect } = {},
) {
  container.replaceChildren();
  const expanded = new Set(expandedStageIds || []);
  const tasks = new Map((presentation.tasks || []).map((task) => [task.id, task]));
  const observations = new Map(
    (presentation.observations || []).map((observation) => [observation.id, observation]),
  );
  if (!presentation.stages?.length) {
    const empty = document.createElement("p");
    empty.className = "empty-state";
    empty.textContent = t("emptyExecution");
    container.append(empty);
    return;
  }
  for (const [index, stage] of presentation.stages.entries()) {
    const article = document.createElement("article");
    article.className = "stage-row";
    article.dataset.stage = stage.id;
    const marker = document.createElement("span");
    marker.className = "stage-index";
    marker.textContent = String(index + 1).padStart(2, "0");
    const body = document.createElement("div");
    body.className = "stage-body";
    const toggle = document.createElement("button");
    toggle.type = "button";
    toggle.className = "stage-toggle";
    toggle.setAttribute("aria-expanded", String(expanded.has(stage.id)));
    const heading = document.createElement("span");
    const title = document.createElement("strong");
    title.textContent = t(stage.labelKey);
    const state = document.createElement("small");
    state.textContent = statusSummary(stage.state, t);
    heading.append(title, state);
    const count = document.createElement("b");
    const total = (stage.memberTaskIds?.length || 0) + (stage.memberObservationIds?.length || 0);
    count.textContent = `${total} ${t("records")}`;
    toggle.append(heading, count);
    toggle.addEventListener("click", () => onToggle?.(stage.id));
    body.append(toggle);
    if (expanded.has(stage.id)) {
      const members = document.createElement("div");
      members.className = "stage-members";
      for (const taskId of stage.memberTaskIds || []) {
        const task = tasks.get(taskId);
        if (task) members.append(memberButton(task, t, onSelect));
      }
      for (const observationId of stage.memberObservationIds || []) {
        const observation = observations.get(observationId);
        if (!observation) continue;
        const button = document.createElement("button");
        button.type = "button";
        button.className = "stage-member";
        button.dataset.status = observation.status;
        const title = document.createElement("strong");
        title.textContent = observation.name;
        const summary = document.createElement("span");
        summary.textContent = observation.kind;
        const status = document.createElement("small");
        status.textContent = statusLabel(observation.status, t);
        button.append(title, summary, status);
        button.addEventListener("click", () => onSelect?.({
          kind: "observation",
          id: observation.id,
          value: observation,
        }, button));
        members.append(button);
      }
      body.append(members);
    }
    article.append(marker, body);
    container.append(article);
  }
}
