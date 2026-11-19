import { buildExecutionGraph } from "./execution-view.js?v=15.10.6.431";

const STAGE_ORDER = { planning: 0, discovery: 10, evidence: 30, other: 55, writing: 60, evaluation: 80, human: 90, delivery: 100 };
const TASK_ORDER = { planner: 0, search: 10, recall: 10, dedup: 20, relevance: 30, relevance_gate: 30, extractor: 40, graph: 40, synthesis: 60, reviewer: 70, adversarial: 60, adversarial_review: 60 };
const roundNumber = record => Number(record.raw?.input?.round ?? record.raw?.metadata?.round ?? /\.r(\d+)$/.exec(record.name || "")?.[1]) || 0;

export function phaseRecords(presentation, parentTaskId, role) {
  const names = role === "synthesis" ? /^(synthesis|revision)(\.|$)/ : /^(reviewer|review)(\.|$)/;
  return (presentation.observations || []).filter(record => record.kind === "subspan"
    && record.parentTaskId === parentTaskId && names.test(record.name || ""))
    .sort((a, b) => roundNumber(a) - roundNumber(b)
      || String(a.startedAt || "").localeCompare(String(b.startedAt || "")) || a.id.localeCompare(b.id));
}

// Business order is a display constraint, never a mutation of the scheduler DAG.
export function buildBusinessFlow(presentation = {}, executionNodes = {}) {
  const tasks = presentation.tasks || [];
  const definitions = Object.fromEntries(tasks.map(task => [task.id, {
    ...task.task, labelKey: task.labelKey, summaryFacts: task.summaryFacts,
    durationMs: task.durationMs, error: task.error,
  }]));
  const observations = (presentation.observations || []).filter(item => item.stageId
    && ["subspan", "delivery", "human", "verification", "planning"].includes(item.kind)
    && !item.parentTaskId && !tasks.some(task => task.id === item.taskId))
    .sort((a, b) => String(a.startedAt || "").localeCompare(String(b.startedAt || "")) || a.id.localeCompare(b.id));
  for (const item of observations) definitions[item.id] = {
    task_id: item.taskId || item.id, agent_type: item.name, labelKey: item.labelKey, observation: item,
  };
  const dependencies = Object.fromEntries(tasks.map(task => [task.id, []]));
  for (const edge of presentation.dependencies || []) dependencies[edge.to]?.push(edge.from);
  const graph = buildExecutionGraph(definitions, dependencies, executionNodes);
  const byTask = new Map(tasks.map(task => [task.id, task]));
  const replacements = new Map();
  const loopEdges = [];
  let nodes = graph.nodes.flatMap(node => {
    const task = byTask.get(node.id);
    const observation = node.task.observation;
    const stage = task?.stageId || observation?.stageId || "other";
    const phase = observation?.kind === "verification" ? 50
      : observation?.name?.includes("paper_selection") ? 35
      : TASK_ORDER[task?.type] ?? STAGE_ORDER[stage] ?? 55;
    const rank = phase * 1000 + node.rank * 10 + (observation ? observations.indexOf(observation) : 0);
    if (!["adversarial", "adversarial_review"].includes(task?.type)) {
      return [{ ...node, rank, presentationOnly: Boolean(observation) }];
    }
    const recordsByRole = Object.fromEntries(["synthesis", "reviewer"].map(role => [
      role, phaseRecords(presentation, task.id, role),
    ]));
    const reported = node.execution?.output?.total_rounds;
    const rounds = new Set(Object.values(recordsByRole).flat().map(roundNumber).filter(round => round > 0));
    // A summary may establish the number of rounds, but never their missing I/O.
    if (Number.isInteger(reported) && reported > 0 && reported <= 100) {
      for (let round = 1; round <= reported; round += 1) rounds.add(round);
    }
    if (!rounds.size) rounds.add(1);
    const roles = [];
    for (const round of [...rounds].sort((a, b) => a - b)) {
      for (const role of ["synthesis", "reviewer"]) {
        const records = recordsByRole[role].filter(record => roundNumber(record) === round);
        for (const [attempt, record] of (records.length ? records : [null]).entries()) {
          roles.push({
            ...node, id: `${task.id}::${role}:r${round}${attempt ? `:${attempt + 1}` : ""}`,
            rank: rank + roles.length,
            name: role === "synthesis" ? "Synthesis" : "Review",
            labelKey: role === "synthesis" ? "flowSynthesis" : "flowReview", round,
            task: { task_id: record?.taskId || `${task.id}::${role}:r${round}`, agent_type: role },
            execution: record || null, status: record?.status || "unknown",
            durationMs: record?.durationMs ?? null, error: record?.error || null,
            summaryFacts: [], role, parentTaskId: task.id, roundRecords: recordsByRole[role],
            parentExecution: node.execution,
          });
        }
      }
    }
    replacements.set(task.id, [roles[0].id, roles.at(-1).id]);
    for (let index = 1; index < roles.length; index += 1) {
      const previous = roles[index - 1], current = roles[index];
      const feedback = previous.role === "reviewer" && current.role === "synthesis";
      loopEdges.push({
        id: `${previous.id}>${current.id}`, from: previous.id, to: current.id,
        kind: feedback ? "feedback" : "sequence",
        labelKey: feedback ? "flowFeedbackObserved" : null,
      });
    }
    if (rounds.size === 1) loopEdges.push({
      id: `${task.id}:feedback`, from: roles.at(-1).id, to: roles[0].id, kind: "feedback",
      labelKey: "flowFeedbackConditional",
    });
    return roles;
  });
  const edges = graph.edges.map(edge => ({
    ...edge, kind: "dependency", sourceEdge: { from: edge.from, to: edge.to },
    from: replacements.get(edge.from)?.[1] || edge.from,
    to: replacements.get(edge.to)?.[0] || edge.to,
  })).concat(loopEdges);

  // Keep recorded dependency precedence even for an unusual/custom task order.
  if (!graph.cyclic) for (let pass = 0; pass < nodes.length; pass += 1) {
    let changed = false;
    const byId = new Map(nodes.map(node => [node.id, node]));
    for (const edge of edges.filter(edge => edge.kind !== "feedback")) {
      const from = byId.get(edge.from), to = byId.get(edge.to);
      if (from && to && from.rank >= to.rank) { to.rank = from.rank + 1; changed = true; }
    }
    if (!changed) break;
  }
  const ranks = [...new Set(nodes.map(node => node.rank))].sort((a, b) => a - b);
  nodes = nodes.map(node => ({ ...node, rank: ranks.indexOf(node.rank) }));
  const connect = (from, to) => {
    if (!edges.some(edge => edge.from === from.id && edge.to === to.id)) {
      edges.push({ id: `sequence:${from.id}>${to.id}`, from: from.id, to: to.id, kind: "sequence" });
    }
  };
  // Independent observed phases join the displayed flow with distinct sequence edges.
  for (const node of nodes.filter(item => item.presentationOnly)) {
    for (const previous of nodes.filter(item => item.rank === node.rank - 1)) connect(previous, node);
    for (const next of nodes.filter(item => item.rank === node.rank + 1)) connect(node, next);
  }
  return { ...graph, nodes, edges, signature: JSON.stringify({
    nodes: nodes.map(node => [node.id, node.rank]), edges: edges.map(edge => [edge.from, edge.to, edge.kind]),
  }) };
}
