const $ = (selector) => document.querySelector(selector);
const state = { context: null, view: null, selectedRunId: null, stream: null };

function formatJson(value) {
  return JSON.stringify(value ?? null, null, 2);
}

function empty(container, message) {
  const text = document.createElement("p");
  text.className = "empty-state";
  text.textContent = message;
  container.replaceChildren(text);
}

function showInspector(label, value) {
  $("#node-kind").textContent = label;
  const box = $("#inspector");
  box.classList.remove("empty");
  box.textContent = formatJson(value);
}

function statusLabel(source) {
  return { sample: "Offline sample", archive: "Local archive", live: "Live run" }[source] || source;
}

function renderSummary() {
  const summary = state.view?.summary;
  if (!summary) return;
  $("#query").textContent = summary.query || "No query recorded";
  $("#run-id").textContent = summary.run_id;
  $("#run-source").textContent = statusLabel(state.view.source);
  $("#run-status").textContent = summary.status;
  $("#total-tokens").textContent = Number(summary.total_tokens || 0).toLocaleString();
  $("#total-elapsed").textContent = summary.elapsed_ms == null ? "-" : `${(summary.elapsed_ms / 1000).toFixed(1)}s`;
  $("#quality").textContent = summary.quality_status;
  $("#delivery").textContent = summary.delivery_status;

  const alert = $("#delivery-alert");
  if (summary.publishable) {
    alert.hidden = true;
  } else {
    alert.hidden = false;
    alert.dataset.status = summary.delivery_status;
    const reasons = summary.reason_codes.length ? ` Reasons: ${summary.reason_codes.join(", ")}.` : "";
    alert.textContent = `Not publishable / 不可发布: ${summary.delivery_status}.${reasons}`;
  }

  const download = $("#download");
  download.href = `/flow-demo/runs/${encodeURIComponent(summary.run_id)}/download`;
  download.classList.remove("disabled");

  const trace = $("#trace-link");
  const traceUrl = state.view.trace?.trace_url;
  if (traceUrl) {
    trace.href = traceUrl;
    trace.classList.remove("disabled");
  } else {
    trace.removeAttribute("href");
    trace.classList.add("disabled");
  }

  const cancellable = state.view.source === "live" && ["queued", "running", "cancelling"].includes(summary.status);
  $("#cancel-run").disabled = !cancellable;
}

function taskDepth(taskId, dependencies, cache, active = new Set()) {
  if (cache.has(taskId)) return cache.get(taskId);
  if (active.has(taskId)) return 0;
  active.add(taskId);
  const parents = dependencies[taskId] || [];
  const depth = parents.length ? Math.max(...parents.map((parent) => taskDepth(parent, dependencies, cache, active))) + 1 : 0;
  active.delete(taskId);
  cache.set(taskId, depth);
  return depth;
}

function renderDag() {
  const dagData = state.view?.dag || { tasks: {}, dependencies: {}, nodes: {} };
  const tasks = Object.entries(dagData.tasks || {});
  const dag = $("#dag");
  dag.replaceChildren();
  $("#dag-count").textContent = `${tasks.length} tasks`;
  if (!tasks.length) {
    empty(dag, "No planner graph was recorded.");
    return;
  }

  const depths = new Map();
  const lanes = new Map();
  for (const [taskId, task] of tasks) {
    const depth = taskDepth(taskId, dagData.dependencies || {}, depths);
    if (!lanes.has(depth)) lanes.set(depth, []);
    lanes.get(depth).push([taskId, task]);
  }

  for (const [depth, laneTasks] of [...lanes.entries()].sort((a, b) => a[0] - b[0])) {
    const lane = document.createElement("section");
    lane.className = "dag-lane";
    const label = document.createElement("h3");
    label.textContent = `Stage ${depth + 1}`;
    lane.append(label);
    const group = document.createElement("div");
    group.className = "dag-nodes";
    for (const [taskId, task] of laneTasks) {
      const node = document.importNode($("#node-template").content, true).querySelector("button");
      const execution = dagData.nodes?.[`worker:${taskId}`] || {};
      const status = execution.status || task.status || "unknown";
      node.dataset.status = status;
      node.querySelector(".node-name").textContent = `${task.agent_type || "worker"} / ${taskId}`;
      const dependencies = dagData.dependencies?.[taskId] || [];
      node.querySelector("small").textContent = dependencies.length ? `after ${dependencies.join(", ")}` : "root";
      node.addEventListener("click", () => showInspector(`worker / ${taskId}`, { task, dependencies, execution }));
      group.append(node);
    }
    lane.append(group);
    dag.append(lane);
  }
}

function renderTimeline() {
  const events = state.view?.timeline || [];
  const timeline = $("#timeline");
  timeline.replaceChildren();
  $("#event-count").textContent = `${events.length} events`;
  if (!events.length) {
    empty(timeline, "No lifecycle events were recorded.");
    return;
  }
  for (const event of events) {
    const button = document.createElement("button");
    button.className = "event";
    const time = document.createElement("time");
    time.textContent = event.at ? new Date(event.at).toLocaleTimeString() : "-";
    const name = document.createElement("span");
    name.className = "event-name";
    name.textContent = event.event || "unknown";
    const task = document.createElement("span");
    task.className = "event-task";
    task.textContent = event.data?.task_id || event.data?.operation_id || "";
    button.append(time, name, task);
    button.addEventListener("click", () => showInspector(`event / ${event.event}`, event));
    timeline.append(button);
  }
}

function renderReport() {
  const report = state.view?.report || {};
  $("#report-text").textContent = report.survey || "No final report was recorded.";
  $("#review-rounds").textContent = `${(report.review_history || []).length} review rounds`;
}

function renderEvidence() {
  const items = state.view?.evidence || [];
  const list = $("#evidence-list");
  list.replaceChildren();
  $("#evidence-count").textContent = `${items.length} items`;
  if (!items.length) {
    empty(list, "No selected evidence is available for this run.");
    return;
  }
  for (const item of items) {
    const article = document.createElement("article");
    const heading = document.createElement("h3");
    heading.textContent = `${item.evidence_id} / ${item.paper_title || item.paper_id || "Unknown paper"}`;
    const meta = document.createElement("p");
    meta.className = "evidence-meta";
    const locator = typeof item.locator === "string" ? item.locator : formatJson(item.locator);
    meta.textContent = `${item.paper_id || "no paper id"} · ${item.content_scope} · ${locator || "no locator"}`;
    const text = document.createElement("p");
    text.textContent = item.text || "No evidence text recorded.";
    article.append(heading, meta, text);
    article.addEventListener("click", () => showInspector(`evidence / ${item.evidence_id}`, item));
    list.append(article);
  }
}

function renderGraph() {
  const graph = state.view?.graph || {};
  const papers = Array.isArray(graph.papers) ? graph.papers : [];
  $("#paper-count").textContent = `${papers.length} papers`;
  const summary = $("#graph-summary");
  summary.replaceChildren();
  const counts = graph.tier_counts || {};
  for (const [tier, count] of Object.entries(counts)) {
    const item = document.createElement("span");
    item.textContent = `${tier}: ${count}`;
    summary.append(item);
  }
  const list = $("#paper-list");
  list.replaceChildren();
  if (!papers.length) {
    empty(list, "No paper graph was produced.");
    return;
  }
  for (const paper of papers) {
    const row = document.createElement("button");
    row.className = "paper-row";
    row.textContent = `${paper.title || paper.paper_id || "Unknown paper"} · ${paper.tier || "unclassified"}`;
    row.addEventListener("click", () => showInspector("paper", paper));
    list.append(row);
  }
}

function renderEvaluation() {
  const evaluation = state.view?.evaluation || {};
  const grid = $("#evaluation-grid");
  grid.replaceChildren();
  if (!Object.keys(evaluation).length) {
    empty(grid, "Evaluation was not completed for this run.");
  } else {
    for (const [metric, result] of Object.entries(evaluation)) {
      const item = document.createElement("section");
      item.className = "metric";
      const name = document.createElement("h3");
      name.textContent = metric.replaceAll("_", " ");
      const score = document.createElement("strong");
      score.textContent = result?.score == null ? "unverified" : Number(result.score).toFixed(3);
      const status = document.createElement("span");
      status.textContent = result?.passed === true ? "passed" : result?.passed === false ? "failed" : "unverified";
      item.dataset.status = status.textContent;
      item.append(name, score, status);
      grid.append(item);
    }
  }
  $("#delivery-detail").textContent = formatJson({ quality: state.view?.quality || {}, delivery: state.view?.delivery || {} });
}

function renderBenchmarks() {
  const benchmarks = state.context?.benchmarks || {};
  const list = $("#benchmark-list");
  list.replaceChildren();
  for (const [name, benchmark] of Object.entries(benchmarks)) {
    const row = document.createElement("section");
    const heading = document.createElement("h3");
    heading.textContent = `${name.toUpperCase()} benchmark`;
    const status = document.createElement("strong");
    status.textContent = benchmark.status;
    const detail = document.createElement("p");
    detail.textContent = benchmark.result ? `Repository evidence: ${benchmark.result}` : "Formal live result is not available yet.";
    row.append(heading, status, detail);
    list.append(row);
  }
}

function render() {
  renderSummary();
  renderDag();
  renderTimeline();
  renderReport();
  renderEvidence();
  renderGraph();
  renderEvaluation();
  renderBenchmarks();
}

async function loadView(runId) {
  const response = await fetch(`/flow-demo/runs/${encodeURIComponent(runId)}/view`);
  if (!response.ok) return;
  state.view = await response.json();
  state.selectedRunId = runId;
  render();
}

function stream(runId) {
  state.stream?.close();
  const source = new EventSource(`/flow-demo/runs/${encodeURIComponent(runId)}/events`);
  state.stream = source;
  source.addEventListener("trace", () => loadView(runId));
  source.addEventListener("done", async () => {
    await loadView(runId);
    source.close();
    await loadRuns(runId);
  });
  source.onerror = () => source.close();
}

async function loadRuns(selectedId) {
  const response = await fetch("/flow-demo/runs");
  const runs = response.ok ? await response.json() : [];
  const select = $("#archive-select");
  select.replaceChildren();
  for (const run of runs) {
    const label = `${statusLabel(run.source)} · ${run.run_id} · ${run.status}`;
    select.add(new Option(label, run.run_id));
  }
  const runId = selectedId || state.selectedRunId || state.context?.default_run_id || runs[0]?.run_id;
  if (runId) {
    select.value = runId;
    await loadView(runId);
  }
}

async function initialize() {
  const response = await fetch("/flow-demo/context");
  state.context = response.ok ? await response.json() : { mode: "offline", live_enabled: false, benchmarks: {} };
  $("#mode").textContent = state.context.mode.toUpperCase();
  $("#mode").dataset.mode = state.context.mode;
  $("#new-run").disabled = !state.context.live_enabled;
  await loadRuns(state.context.default_run_id);
}

$("#archive-select").addEventListener("change", (event) => {
  if (event.target.value) loadView(event.target.value);
});

$("#new-run").addEventListener("click", async () => {
  const button = $("#new-run");
  button.disabled = true;
  try {
    const response = await fetch("/flow-demo/runs", { method: "POST" });
    if (!response.ok) return;
    const run = await response.json();
    await loadRuns(run.run_id);
    stream(run.run_id);
  } finally {
    button.disabled = !state.context?.live_enabled;
  }
});

$("#cancel-run").addEventListener("click", async () => {
  const runId = state.view?.summary?.run_id;
  if (!runId) return;
  await fetch(`/survey/${encodeURIComponent(runId)}`, { method: "DELETE" });
  await loadView(runId);
});

for (const tab of document.querySelectorAll('[role="tab"]')) {
  tab.addEventListener("click", () => {
    for (const candidate of document.querySelectorAll('[role="tab"]')) {
      const selected = candidate === tab;
      candidate.setAttribute("aria-selected", String(selected));
      $(`#tab-${candidate.dataset.tab}`).hidden = !selected;
    }
  });
}

initialize();
