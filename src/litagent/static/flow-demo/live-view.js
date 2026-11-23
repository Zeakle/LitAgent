const liveStatuses = new Set(["queued", "running", "cancelling"]);
const sequence = event => event?.seq ?? event?.sequence ?? -1;
const transitions = new Map([
  ["worker.input", "pending"], ["worker.start", "running"], ["worker.complete", "completed"],
  ["worker.failed", "failed"], ["worker.cancelled", "cancelled"], ["subspan.start", "running"],
]);

function appendTrace(view, event) {
  const events = [...view.events, event];
  const data = event.data || {};
  const taskId = data.task_id;
  const state = event.event === "subspan.end"
    ? (data.error_type === "CancelledError" ? "cancelled" : data.error ? "failed" : "completed")
    : transitions.get(event.event);
  if (!state || !taskId) return { ...view, events };

  const kind = event.event.split(".")[0];
  const id = `${kind}:${taskId}`;
  const graph = view.graph;
  const previous = graph.nodes[id];
  const node = { ...previous, id, kind, task_id: taskId, status: state,
    name: data.name || data.agent_type || previous?.name || taskId };
  if (event.event === "subspan.start") node.input = data;
  if (event.event === "worker.input") node.input = data.input;
  if ("output" in data) node.output = data.output;
  if (data.error) node.error = data.error;
  const tasks = graph.tasks;
  const needsDefinition = kind === "worker" && !Object.hasOwn(tasks, taskId);
  const status = view.identity.status === "queued" && state === "running" ? "running" : view.identity.status;
  return { ...view, events, identity: { ...view.identity, status }, summary: { ...view.summary, status },
    graph: { ...graph, nodes: { ...graph.nodes, [id]: node },
      tasks: needsDefinition ? { ...tasks, [taskId]: { task_id: taskId, agent_type: node.name } } : tasks } };
}

// Recorder sequences start at zero; SQLite review streams use one-based seq.
// Both are monotonic for the lifetime of the selected run.
function reconcileSnapshot(snapshot, current) {
  if (!liveStatuses.has(snapshot.identity?.status)) return snapshot;
  const cutoff = sequence(snapshot.events.at(-1));
  const unseen = current.events.filter(event => sequence(event) > cutoff);
  return unseen.reduce(appendTrace, snapshot);
}

// Keep detailed payloads synchronized without waiting on a GET to mark a step.
export function createLiveViewRefresh({ current, load, apply, onError, delay = 1000 }) {
  let timer = null;
  let generation = 0;
  let busy = false;
  let dirty = false;
  function cancel() {
    generation += 1;
    clearTimeout(timer); timer = null; dirty = false; busy = false;
  }
  function schedule() {
    dirty = true;
    if (timer || busy) return;
    const state = current();
    if (!state.view?.identity) return;
    const { identity } = state.view;
    const epoch = state.requestEpoch;
    const captured = generation;
    timer = setTimeout(async () => {
      timer = null; busy = true; dirty = false;
      try {
        const view = await load(identity);
        const now = current();
        if (generation === captured && now.requestEpoch === epoch
          && now.view?.identity?.key === identity.key
          && !now.view?.displayRevision) apply(reconcileSnapshot(view, now.view));
      } catch (error) {
        if (generation === captured) onError?.(error);
      } finally {
        if (generation === captured) { busy = false; if (dirty) schedule(); }
      }
    }, delay);
  }
  function receive(event) {
    const view = current().view;
    if (!view || view.displayRevision || !liveStatuses.has(view.identity.status)) return;
    if (sequence(event) <= sequence(view.events.at(-1))) return;
    apply(appendTrace(view, event));
    schedule();
  }
  return { schedule, cancel, receive };
}
