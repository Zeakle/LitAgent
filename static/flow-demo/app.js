const $ = (selector) => document.querySelector(selector);
const state = { artifact: null, selected: null, stream: null };

function formatJson(value) {
  return JSON.stringify(value ?? null, null, 2);
}

function elapsed(artifact) {
  const start = Date.parse(artifact.started_at || "");
  const end = Date.parse(artifact.completed_at || "") || Date.now();
  return Number.isFinite(start) ? `${((end - start) / 1000).toFixed(1)}s` : "-";
}

function totalTokens(artifact) {
  return Object.values(artifact.nodes || {}).reduce((sum, node) => sum + (node.usage?.total_tokens || 0), 0);
}

function showInspector(label, value) {
  $("#node-kind").textContent = label;
  const box = $("#inspector");
  box.classList.remove("empty");
  box.textContent = formatJson(value);
}

function renderSummary() {
  const artifact = state.artifact;
  if (!artifact) return;
  const report = artifact.report || {};
  $("#run-id").textContent = artifact.run_id;
  $("#run-status").textContent = artifact.status;
  $("#total-tokens").textContent = totalTokens(artifact).toLocaleString();
  $("#total-elapsed").textContent = elapsed(artifact);
  $("#quality").textContent = report.quality?.status || "unverified";
  $("#delivery").textContent = report.delivery?.status || "pending";
  const download = $("#download");
  if (artifact.completed_at) {
    download.href = `/flow-demo/runs/${encodeURIComponent(artifact.run_id)}/download`;
    download.classList.remove("disabled");
  }
}

function renderDag() {
  const graph = state.artifact?.graph || { tasks: {}, dependencies: {} };
  const nodes = state.artifact?.nodes || {};
  const dag = $("#dag");
  dag.replaceChildren();
  const tasks = Object.entries(graph.tasks || {});
  $("#dag-count").textContent = `${tasks.length} tasks`;
  if (!tasks.length) {
    dag.innerHTML = '<p class="empty-state">Waiting for planner decomposition.</p>';
    return;
  }
  for (const [taskId, task] of tasks) {
    const node = document.importNode($("#node-template").content, true).querySelector("button");
    const execution = nodes[`worker:${taskId}`] || {};
    node.dataset.status = execution.status || task.status || "pending";
    node.querySelector(".node-name").textContent = `${task.agent_type || "worker"}  ${taskId}`;
    const deps = graph.dependencies?.[taskId] || [];
    node.querySelector("small").textContent = deps.length ? `after ${deps.join(", ")}` : "root";
    node.onclick = () => showInspector(`worker / ${taskId}`, { task, dependencies: deps, execution });
    dag.append(node);
  }
}

function renderTimeline() {
  const timeline = $("#timeline");
  const events = state.artifact?.events || [];
  timeline.replaceChildren();
  $("#event-count").textContent = `${events.length} events`;
  if (!events.length) {
    timeline.innerHTML = '<p class="empty-state">Waiting for runtime events.</p>';
    return;
  }
  for (const event of events) {
    const button = document.createElement("button");
    button.className = "event";
    const time = new Date(event.at).toLocaleTimeString();
    button.innerHTML = `<time>${time}</time><span class="event-name">${event.event}</span><span class="event-task">${event.data?.task_id || event.data?.operation_id || ""}</span>`;
    button.onclick = () => showInspector(`event / ${event.event}`, event);
    timeline.append(button);
  }
}

function render() {
  renderSummary();
  renderDag();
  renderTimeline();
}

async function loadRun(runId) {
  const response = await fetch(`/flow-demo/runs/${encodeURIComponent(runId)}`);
  if (!response.ok) return;
  state.artifact = await response.json();
  render();
}

function stream(runId) {
  state.stream?.close();
  const source = new EventSource(`/flow-demo/runs/${encodeURIComponent(runId)}/events`);
  state.stream = source;
  source.addEventListener("trace", () => loadRun(runId));
  source.addEventListener("done", () => { loadRun(runId); source.close(); loadArchives(); });
  source.onerror = () => source.close();
}

async function loadArchives(selectedId) {
  const response = await fetch("/flow-demo/runs");
  const runs = response.ok ? await response.json() : [];
  const select = $("#archive-select");
  select.replaceChildren();
  if (!runs.length) select.add(new Option("No archived runs", ""));
  for (const run of runs) select.add(new Option(`${run.run_id} (${run.status})`, run.run_id));
  const runId = selectedId || state.artifact?.run_id || runs[0]?.run_id;
  if (runId) {
    select.value = runId;
    await loadRun(runId);
  }
}

$("#archive-select").addEventListener("change", (event) => {
  if (event.target.value) loadRun(event.target.value);
});
$("#new-run").addEventListener("click", async () => {
  const button = $("#new-run");
  button.disabled = true;
  try {
    const response = await fetch("/flow-demo/runs", { method: "POST" });
    const run = await response.json();
    state.artifact = { run_id: run.run_id, query: run.query, status: "running", nodes: {}, graph: {}, events: [] };
    render();
    await loadArchives(run.run_id);
    stream(run.run_id);
  } finally {
    button.disabled = false;
  }
});

loadArchives();
