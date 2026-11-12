const TERMINAL = new Set(["completed", "failed", "cancelled", "skipped"]);

const TASK_TYPES = {
  planner: { stageId: "planning", labelKey: "taskPlanner" },
  search: { stageId: "discovery", labelKey: "taskSearch" },
  recall: { stageId: "discovery", labelKey: "taskRecall" },
  dedup: { stageId: "evidence", labelKey: "taskDedup" },
  relevance_gate: { stageId: "evidence", labelKey: "taskRelevance" },
  extractor: { stageId: "evidence", labelKey: "taskExtraction" },
  graph: { stageId: "evidence", labelKey: "taskGraphAnalysis" },
  synthesis: { stageId: "writing", labelKey: "taskSynthesis" },
  reviewer: { stageId: "writing", labelKey: "taskReviewer" },
  adversarial_review: { stageId: "writing", labelKey: "taskAdversarial" },
};

const STAGES = [
  { id: "planning", labelKey: "stagePlanning" },
  { id: "discovery", labelKey: "stageDiscovery" },
  { id: "evidence", labelKey: "stageEvidence" },
  { id: "writing", labelKey: "stageWriting" },
  { id: "evaluation", labelKey: "stageEvaluation" },
  { id: "human", labelKey: "stageHumanReview" },
  { id: "delivery", labelKey: "stageDelivery" },
  { id: "other", labelKey: "stageOther" },
];

function object(value) {
  return value && typeof value === "object" && !Array.isArray(value) ? value : {};
}

function taskEntries(tasks) {
  if (Array.isArray(tasks)) {
    return tasks
      .filter((task) => task && typeof task === "object" && task.task_id != null)
      .map((task) => [String(task.task_id), task]);
  }
  return Object.entries(object(tasks));
}

function normalizeStatus(value) {
  if (value === "done") return "completed";
  if (value === "error") return "failed";
  return typeof value === "string" && value ? value : "unknown";
}

function matchingExecution(nodes, taskId) {
  const records = object(nodes);
  return records[`worker:${taskId}`]
    || Object.values(records).find(
      (node) => node?.task_id === taskId && node?.kind === "worker",
    )
    || null;
}

function finiteDuration(execution) {
  const direct = Number(execution?.duration_ms ?? execution?.elapsed_ms);
  if (Number.isFinite(direct) && direct >= 0) return direct;
  const started = Date.parse(execution?.started_at || "");
  const completed = Date.parse(execution?.completed_at || "");
  return Number.isFinite(started) && Number.isFinite(completed) && completed >= started
    ? completed - started
    : null;
}

function uniqueCount(items, field) {
  if (!Array.isArray(items)) return null;
  const values = items
    .map((item) => item?.[field])
    .filter((value) => typeof value === "string" && value);
  return values.length ? new Set(values).size : null;
}

function evidenceCount(items) {
  if (!Array.isArray(items)) return null;
  const ids = new Set();
  for (const item of items) {
    for (const evidence of Array.isArray(item?.evidence_items) ? item.evidence_items : []) {
      if (typeof evidence?.evidence_id === "string" && evidence.evidence_id) {
        ids.add(evidence.evidence_id);
      }
    }
  }
  return ids.size || null;
}

export function summarizeExecutionTask(task, execution) {
  const output = execution?.output;
  const type = task?.agent_type || execution?.name || "worker";
  const sourcePath = `graph.nodes.${execution?.id || `worker:${task?.task_id || "unknown"}`}.output`;
  const facts = [];
  const papers = uniqueCount(output, "paper_id");
  if (["search", "recall"].includes(type) && papers != null) {
    facts.push({ kind: "paper_count", value: papers, labelKey: "papersFound", sourcePath });
  } else if (["dedup", "relevance_gate"].includes(type) && papers != null) {
    facts.push({ kind: "paper_count", value: papers, labelKey: "papersRetained", sourcePath });
  } else if (type === "extractor" && papers != null) {
    facts.push({ kind: "paper_count", value: papers, labelKey: "papersExtracted", sourcePath });
    const evidence = evidenceCount(output);
    if (evidence != null) {
      facts.push({ kind: "evidence_count", value: evidence, labelKey: "evidenceGenerated", sourcePath });
    }
  } else if (type === "graph" && Array.isArray(output?.papers)) {
    facts.push({ kind: "paper_count", value: output.papers.length, labelKey: "papersAnalyzed", sourcePath });
  } else if (type === "adversarial_review" && output && typeof output === "object") {
    if (Number.isFinite(Number(output.total_rounds))) {
      facts.push({ kind: "round_count", value: Number(output.total_rounds), labelKey: "reviewRounds", sourcePath });
    }
    if (Number.isFinite(Number(output.evidence_selection?.selected_count))) {
      facts.push({
        kind: "evidence_count",
        value: Number(output.evidence_selection.selected_count),
        labelKey: "evidenceSelected",
        sourcePath,
      });
    }
  }
  return facts;
}

export function aggregateStageState(members = []) {
  const counts = {};
  for (const member of members) {
    const status = normalizeStatus(member?.status);
    counts[status] = (counts[status] || 0) + 1;
  }
  const unknownCount = counts.unknown || 0;
  return {
    counts,
    active: Boolean((counts.running || 0) + (counts.queued || 0) + (counts.cancelling || 0)),
    waitingHuman: Boolean(counts.awaiting_human),
    hasFailures: Boolean(counts.failed),
    terminal: members.length > 0 && unknownCount === 0
      && members.every((member) => TERMINAL.has(normalizeStatus(member?.status))),
    unknownCount,
  };
}

function observationStage(node) {
  const name = `${node?.task_id || ""} ${node?.name || ""}`;
  if (/query_decomposition|planner/.test(name)) return "planning";
  if (/evaluation|evidence_rewrite|human_revision|citation_normalization/.test(name)) return "evaluation";
  if (/human\.|paper_selection|final_approval/.test(name)) return "human";
  return null;
}

function normalizeObservation(id, node) {
  const separator = id.indexOf(":");
  const kind = node?.kind || "system";
  const inferredOperationId = separator >= 0 && !["worker", "subspan"].includes(kind)
    ? id.slice(separator + 1)
    : null;
  return {
    id,
    kind,
    taskId: node?.task_id || null,
    parentTaskId: node?.input?.parent_task_id || node?.metadata?.parent_task_id || null,
    operationId: node?.operation_id || node?.metadata?.operation_id || inferredOperationId,
    name: node?.name || id,
    status: normalizeStatus(node?.status),
    startedAt: node?.started_at || null,
    completedAt: node?.completed_at || null,
    durationMs: finiteDuration(node),
    attempt: node?.metadata?.attempt ?? null,
    input: node?.input ?? null,
    output: node?.output ?? null,
    usage: node?.usage ?? null,
    error: node?.error ?? null,
    raw: node,
    stageId: observationStage(node),
  };
}

function hasCycle(taskIds, dependencies) {
  const children = new Map(taskIds.map((id) => [id, []]));
  const indegree = new Map(taskIds.map((id) => [id, 0]));
  for (const edge of dependencies) {
    children.get(edge.from)?.push(edge.to);
    indegree.set(edge.to, (indegree.get(edge.to) || 0) + 1);
  }
  const ready = [...indegree].filter(([, count]) => count === 0).map(([id]) => id);
  let visited = 0;
  while (ready.length) {
    const id = ready.shift();
    visited += 1;
    for (const child of children.get(id) || []) {
      indegree.set(child, indegree.get(child) - 1);
      if (indegree.get(child) === 0) ready.push(child);
    }
  }
  return visited !== taskIds.length;
}

export function buildExecutionPresentation(view = {}, { locale = "zh-CN" } = {}) {
  const graph = object(view.graph);
  const nodes = object(graph.nodes);
  const tasks = taskEntries(graph.tasks).map(([id, task]) => {
    const execution = matchingExecution(nodes, id);
    const type = task.agent_type || execution?.name || "worker";
    const definition = TASK_TYPES[type] || { stageId: "other", labelKey: "taskOther" };
    return {
      id,
      type,
      stageId: definition.stageId,
      labelKey: definition.labelKey,
      status: normalizeStatus(execution?.status || task.status),
      durationMs: finiteDuration(execution),
      summaryFacts: summarizeExecutionTask(task, execution),
      error: execution?.error || task.error || null,
      task,
      execution,
    };
  });
  const taskIds = new Set(tasks.map((task) => task.id));
  const dependencies = [];
  const diagnostics = [];
  for (const task of tasks) {
    for (const dependencyId of graph.dependencies?.[task.id] || []) {
      if (!taskIds.has(dependencyId)) {
        diagnostics.push({ code: "missing_dependency", taskId: task.id, dependencyId });
        continue;
      }
      dependencies.push({ from: dependencyId, to: task.id });
    }
  }
  if (hasCycle([...taskIds], dependencies)) diagnostics.push({ code: "cycle_detected" });

  const observations = Object.entries(nodes).map(([id, node]) => normalizeObservation(id, node));
  if (view.delivery) {
    observations.push({
      id: "system:delivery",
      kind: "delivery",
      taskId: null,
      parentTaskId: null,
      operationId: null,
      name: "delivery",
      // The delivery policy has finished even when its business conclusion is blocked.
      status: "completed",
      startedAt: null,
      completedAt: null,
      durationMs: null,
      attempt: null,
      input: { quality: view.quality || null },
      output: view.delivery,
      usage: null,
      error: null,
      raw: view.delivery,
      stageId: "delivery",
    });
  }
  if (view.review && !observations.some((item) => item.stageId === "human")) {
    observations.push({
      id: "system:human-review",
      kind: "human",
      taskId: null,
      parentTaskId: null,
      operationId: null,
      name: "human review",
      status: "awaiting_human",
      startedAt: null,
      completedAt: null,
      durationMs: null,
      attempt: null,
      input: view.review,
      output: null,
      usage: null,
      error: null,
      raw: view.review,
      stageId: "human",
    });
  }
  const tasksById = new Map(tasks.map((task) => [task.id, task]));
  const observationsById = new Map(observations.map((item) => [item.id, item]));
  const stages = STAGES.map((stage) => {
    const memberTaskIds = tasks.filter((task) => task.stageId === stage.id).map((task) => task.id);
    const memberObservationIds = observations
      .filter((item) => item.stageId === stage.id && !memberTaskIds.includes(item.taskId))
      .map((item) => item.id);
    const members = [
      ...memberTaskIds.map((id) => tasksById.get(id)),
      ...memberObservationIds.map((id) => observationsById.get(id)),
    ];
    return { ...stage, memberTaskIds, memberObservationIds, state: aggregateStageState(members) };
  }).filter((stage) => stage.memberTaskIds.length || stage.memberObservationIds.length
    || (stage.id === "delivery" && (view.quality || view.delivery))
    || (stage.id === "human" && view.review));

  return {
    locale,
    tasks,
    stages,
    dependencies,
    observations,
    events: Array.isArray(view.events) ? view.events : [],
    diagnostics,
    quality: view.quality || null,
    delivery: view.delivery || null,
    review: view.review || null,
  };
}

export function buildStageOverview(presentation, expandedStageIds = []) {
  const expanded = new Set(expandedStageIds || []);
  const taskById = new Map((presentation.tasks || []).map((task) => [task.id, task]));
  const observationById = new Map(
    (presentation.observations || []).map((observation) => [observation.id, observation]),
  );
  const nodes = [];
  const displayId = new Map();
  for (const stage of presentation.stages || []) {
    const members = [
      ...(stage.memberTaskIds || []).map((id) => taskById.get(id)).filter(Boolean),
      ...(stage.memberObservationIds || []).map((id) => observationById.get(id)).filter(Boolean),
    ];
    if (expanded.has(stage.id) && stage.memberTaskIds?.length) {
      for (const taskId of stage.memberTaskIds) {
        const task = taskById.get(taskId);
        if (task) nodes.push({ ...task, nodeKind: "task" });
        displayId.set(taskId, taskId);
      }
    } else {
      const id = `stage:${stage.id}`;
      nodes.push({ ...stage, id, nodeKind: "stage", state: aggregateStageState(members) });
      for (const taskId of stage.memberTaskIds || []) displayId.set(taskId, id);
    }
  }
  const grouped = new Map();
  for (const edge of presentation.dependencies || []) {
    const from = displayId.get(edge.from);
    const to = displayId.get(edge.to);
    if (!from || !to || from === to) continue;
    const key = `${from}>${to}`;
    if (!grouped.has(key)) grouped.set(key, { from, to, sourceEdges: [] });
    grouped.get(key).sourceEdges.push(edge);
  }
  return { nodes, edges: [...grouped.values()] };
}
