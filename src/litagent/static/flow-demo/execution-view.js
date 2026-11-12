function taskEntries(tasks) {
  if (Array.isArray(tasks)) return tasks.map((task) => [String(task.task_id), task]);
  return Object.entries(tasks || {});
}

function matchingExecution(nodes, taskId) {
  return nodes?.[`worker:${taskId}`]
    || Object.values(nodes || {}).find(
      (node) => node?.task_id === taskId && node?.kind === "worker",
    )
    || null;
}

export function buildExecutionGraph(tasks = {}, dependencies = {}, executionNodes = {}) {
  const definitions = new Map(taskEntries(tasks));
  const missingDependencies = [];
  const adjacency = new Map([...definitions.keys()].map((id) => [id, []]));
  const indegree = new Map([...definitions.keys()].map((id) => [id, 0]));
  const edges = [];

  for (const taskId of definitions.keys()) {
    for (const dependencyId of dependencies?.[taskId] || []) {
      if (!definitions.has(dependencyId)) {
        missingDependencies.push({ taskId, dependencyId });
        continue;
      }
      adjacency.get(dependencyId).push(taskId);
      indegree.set(taskId, indegree.get(taskId) + 1);
      edges.push({ id: `${dependencyId}>${taskId}`, from: dependencyId, to: taskId });
    }
  }

  const ready = [...indegree.entries()]
    .filter(([, count]) => count === 0)
    .map(([id]) => id)
    .sort();
  const order = [];
  while (ready.length) {
    const id = ready.shift();
    order.push(id);
    for (const child of adjacency.get(id) || []) {
      indegree.set(child, indegree.get(child) - 1);
      if (indegree.get(child) === 0) ready.push(child);
    }
    ready.sort();
  }
  const cyclic = order.length !== definitions.size;
  const stableOrder = cyclic ? [...definitions.keys()].sort() : order;
  const rank = new Map(stableOrder.map((id) => [id, 0]));
  if (!cyclic) {
    for (const id of stableOrder) {
      for (const child of adjacency.get(id) || []) {
        rank.set(child, Math.max(rank.get(child), rank.get(id) + 1));
      }
    }
  }

  const nodes = stableOrder.map((id) => {
    const task = definitions.get(id) || {};
    const execution = matchingExecution(executionNodes, id);
    return {
      id,
      rank: rank.get(id) || 0,
      task,
      execution,
      name: task.agent_type || execution?.name || "worker",
      labelKey: task.labelKey || null,
      summaryFacts: task.summaryFacts || [],
      status: execution?.status || task.status || "pending",
      durationMs: execution?.duration_ms ?? execution?.elapsed_ms ?? task.durationMs ?? null,
      usage: execution?.usage || null,
      error: execution?.error || task.error || null,
    };
  });
  return {
    nodes,
    edges,
    missingDependencies,
    cyclic,
    signature: `${nodes.map((node) => node.id).join("|")}::${edges
      .map((edge) => `${edge.from}>${edge.to}`)
      .join("|")}`,
  };
}

function layoutOptions(options) {
  if (typeof options === "string") return { direction: options };
  return options || {};
}

export function layoutExecutionGraph(graph, options = {}) {
  const config = layoutOptions(options);
  const direction = config.direction || "LR";
  const nodeWidth = Number(config.nodeSizes?.width) || 248;
  const nodeHeight = Number(config.nodeSizes?.height) || 108;
  const primaryGap = 104;
  const crossGap = 28;
  const padding = 32;
  const groups = new Map();
  for (const node of graph.nodes || []) {
    const rank = node.rank || 0;
    if (!groups.has(rank)) groups.set(rank, []);
    groups.get(rank).push(node);
  }
  const incomingOrder = new Map();
  for (const edge of graph.edges || []) {
    if (!incomingOrder.has(edge.to)) incomingOrder.set(edge.to, []);
    incomingOrder.get(edge.to).push(edge.from);
  }
  const previousOrder = new Map();
  const positioned = [];
  for (const [rank, nodes] of [...groups.entries()].sort((a, b) => a[0] - b[0])) {
    nodes.sort((left, right) => {
      const score = (node) => {
        const parents = incomingOrder.get(node.id) || [];
        if (!parents.length) return Number.POSITIVE_INFINITY;
        const positions = parents.map((id) => previousOrder.get(id)).filter(Number.isFinite);
        return positions.length
          ? positions.reduce((sum, value) => sum + value, 0) / positions.length
          : Number.POSITIVE_INFINITY;
      };
      return score(left) - score(right) || left.id.localeCompare(right.id);
    });
    nodes.forEach((node, index) => {
      previousOrder.set(node.id, index);
      const primary = padding + rank * (nodeWidth + primaryGap);
      const cross = padding + index * (nodeHeight + crossGap);
      positioned.push({
        ...node,
        x: direction === "LR" ? primary : cross,
        y: direction === "LR" ? cross : primary,
        width: nodeWidth,
        height: nodeHeight,
      });
    });
  }
  const byId = new Map(positioned.map((node) => [node.id, node]));
  const contentWidth = positioned.length
    ? Math.max(...positioned.map((node) => node.x + node.width)) + padding
    : 0;
  const contentHeight = positioned.length
    ? Math.max(...positioned.map((node) => node.y + node.height)) + padding
    : 0;
  const bounds = { x: 0, y: 0, width: contentWidth, height: contentHeight };
  return {
    ...graph,
    nodes: positioned,
    byId,
    width: Math.max(520, contentWidth),
    height: Math.max(300, contentHeight),
    bounds,
    direction,
  };
}

export function fitGraphToViewport(
  bounds,
  viewport,
  { padding = 24, minScale = 0.25, maxScale = 1 } = {},
) {
  const contentWidth = Number(bounds?.width) || 0;
  const contentHeight = Number(bounds?.height) || 0;
  const viewportWidth = Number(viewport?.width) || 0;
  const viewportHeight = Number(viewport?.height) || 0;
  if (!contentWidth || !contentHeight || !viewportWidth || !viewportHeight) {
    return { scale: 1, translateX: 0, translateY: 0, ready: false };
  }
  const availableWidth = Math.max(1, viewportWidth - padding * 2);
  const availableHeight = Math.max(1, viewportHeight - padding * 2);
  const scale = Math.min(
    maxScale,
    Math.max(minScale, Math.min(availableWidth / contentWidth, availableHeight / contentHeight)),
  );
  return {
    scale,
    translateX: (viewportWidth - contentWidth * scale) / 2 - (Number(bounds.x) || 0) * scale,
    translateY: (viewportHeight - contentHeight * scale) / 2 - (Number(bounds.y) || 0) * scale,
    ready: true,
  };
}

export function getNodeNeighborhood(graph, nodeId) {
  if (!(graph.nodes || []).some((node) => node.id === nodeId)) {
    return { upstreamIds: [], downstreamIds: [], edgeIds: [] };
  }
  const upstreamIds = [];
  const downstreamIds = [];
  const edgeIds = [];
  for (const edge of graph.edges || []) {
    if (edge.to === nodeId) {
      upstreamIds.push(edge.from);
      edgeIds.push(edge.id || `${edge.from}>${edge.to}`);
    }
    if (edge.from === nodeId) {
      downstreamIds.push(edge.to);
      edgeIds.push(edge.id || `${edge.from}>${edge.to}`);
    }
  }
  return { upstreamIds, downstreamIds, edgeIds: [...new Set(edgeIds)] };
}

export function renderExecutionGraph(container, graph, onSelect, options = {}) {
  container.replaceChildren();
  if (!graph.nodes.length) return null;
  const stage = document.createElement("div");
  stage.className = "graph-stage";
  stage.style.width = `${graph.width}px`;
  stage.style.height = `${graph.height}px`;
  stage.style.transform = `translate(${options.translateX || 0}px, ${options.translateY || 0}px) scale(${options.scale || 1})`;

  const svg = document.createElementNS("http://www.w3.org/2000/svg", "svg");
  svg.classList.add("graph-edges");
  svg.setAttribute("viewBox", `0 0 ${graph.width} ${graph.height}`);
  const defs = document.createElementNS("http://www.w3.org/2000/svg", "defs");
  const marker = document.createElementNS("http://www.w3.org/2000/svg", "marker");
  marker.setAttribute("id", "graph-arrow");
  marker.setAttribute("viewBox", "0 0 10 10");
  marker.setAttribute("refX", "9");
  marker.setAttribute("refY", "5");
  marker.setAttribute("markerWidth", "6");
  marker.setAttribute("markerHeight", "6");
  marker.setAttribute("orient", "auto-start-reverse");
  const arrow = document.createElementNS("http://www.w3.org/2000/svg", "path");
  arrow.setAttribute("d", "M 0 0 L 10 5 L 0 10 z");
  marker.append(arrow);
  defs.append(marker);
  svg.append(defs);

  const neighborhood = options.selectedId
    ? getNodeNeighborhood(graph, options.selectedId)
    : { upstreamIds: [], downstreamIds: [], edgeIds: [] };
  const related = new Set([
    options.selectedId,
    ...neighborhood.upstreamIds,
    ...neighborhood.downstreamIds,
  ].filter(Boolean));
  for (const edge of graph.edges) {
    const from = graph.byId.get(edge.from);
    const to = graph.byId.get(edge.to);
    if (!from || !to) continue;
    const path = document.createElementNS("http://www.w3.org/2000/svg", "path");
    path.classList.add("graph-edge");
    const x1 = from.x + from.width;
    const y1 = from.y + from.height / 2;
    const x2 = to.x;
    const y2 = to.y + to.height / 2;
    const mid = (x1 + x2) / 2;
    path.setAttribute("d", `M ${x1} ${y1} C ${mid} ${y1}, ${mid} ${y2}, ${x2} ${y2}`);
    path.setAttribute("marker-end", "url(#graph-arrow)");
    path.dataset.edgeId = edge.id || `${edge.from}>${edge.to}`;
    if (options.selectedId && !neighborhood.edgeIds.includes(path.dataset.edgeId)) {
      path.dataset.dimmed = "true";
    }
    svg.append(path);
  }
  stage.append(svg);
  for (const node of graph.nodes) {
    const button = document.createElement("button");
    button.type = "button";
    button.className = "graph-node";
    button.dataset.status = node.status;
    button.dataset.selected = String(node.id === options.selectedId);
    button.dataset.dimmed = String(Boolean(options.selectedId && !related.has(node.id)));
    button.style.left = `${node.x}px`;
    button.style.top = `${node.y}px`;
    button.style.width = `${node.width}px`;
    button.style.height = `${node.height}px`;
    const name = document.createElement("strong");
    name.textContent = options.label?.(node.labelKey, node.name) || node.name;
    const summary = document.createElement("span");
    summary.className = "graph-node-summary";
    summary.textContent = options.summary?.(node) || node.id;
    const status = document.createElement("small");
    status.textContent = options.status?.(node)
      || `${node.status}${node.durationMs == null ? "" : ` · ${node.durationMs} ms`}`;
    button.append(name, summary, status);
    button.addEventListener("click", () => onSelect?.({ kind: "node", id: node.id, value: node }, button));
    stage.append(button);
  }
  container.append(stage);
  return stage;
}

export function classifyEvent(name = "") {
  if (/failed|error|cancelled/.test(name)) return "error";
  if (name.startsWith("worker.")) return "worker";
  if (name.startsWith("llm.")) return "llm";
  if (name.startsWith("tool.")) return "tool";
  if (/rag|retriev|evidence/.test(name)) return "rag";
  if (name.startsWith("memory.")) return "memory";
  if (/evaluation|quality/.test(name)) return "evaluation";
  if (/rewrite|revision/.test(name)) return "rewrite";
  return "system";
}

export function filterTimeline(events = [], category = "all") {
  return category === "all"
    ? events
    : events.filter((event) => classifyEvent(event.event) === category);
}
