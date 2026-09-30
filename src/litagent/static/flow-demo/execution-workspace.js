import { buildExecutionPresentation } from "./execution-model.js";
import { renderStageOverview } from "./execution-overview.js";
import { buildCallTimeline, renderCallTimeline } from "./execution-timeline.js";
import {
  buildExecutionGraph,
  fitGraphToViewport,
  layoutExecutionGraph,
  renderExecutionGraph,
} from "./execution-view.js";

const $ = (selector) => document.querySelector(selector);

export function createExecutionWorkspace({ store, t, onInspect }) {
  let currentLayout = null;
  let pendingFitSignature = null;
  let graphPan = null;

  function update(patch) {
    const runId = store.get().view?.identity?.runId;
    if (runId) store.setExecutionUi(runId, patch);
  }

  function status(node) {
    const key = `status${String(node.status || "unknown").replace(
      /(^|_)([a-z])/g,
      (_match, _prefix, letter) => letter.toUpperCase(),
    )}`;
    const elapsed = node.durationMs == null
      ? ""
      : ` · ${(node.durationMs / 1000).toFixed(1)}${t("secondsShort")}`;
    return `${t(key)}${elapsed}`;
  }

  function select(selection, trigger) {
    onInspect(selection, trigger);
    const runId = store.get().view?.identity?.runId;
    if (runId && selection.kind === "node") {
      store.setExecutionUi(runId, { selectedNodeId: selection.id });
    }
  }

  function fit(signature = currentLayout?.signature) {
    const container = $("#execution-graph");
    if (!currentLayout || container.hidden) {
      pendingFitSignature = null;
      return;
    }
    const transform = fitGraphToViewport(currentLayout.bounds, {
      width: container.clientWidth,
      height: container.clientHeight,
    });
    pendingFitSignature = null;
    if (transform.ready) update({ ...transform, fittedSignature: signature });
  }

  function zoom(step) {
    const container = $("#execution-graph");
    const ui = store.getExecutionUi(store.get().view?.identity?.runId);
    const nextScale = Math.min(2, Math.max(0.25, ui.scale + step));
    const centerX = container.clientWidth / 2;
    const centerY = container.clientHeight / 2;
    const ratio = nextScale / ui.scale;
    update({
      scale: nextScale,
      translateX: centerX - (centerX - ui.translateX) * ratio,
      translateY: centerY - (centerY - ui.translateY) * ratio,
      fittedSignature: currentLayout?.signature || null,
    });
  }

  function focus() {
    const container = $("#execution-graph");
    const ui = store.getExecutionUi(store.get().view?.identity?.runId);
    const node = currentLayout?.byId?.get(ui.selectedNodeId);
    if (!node) return;
    update({
      translateX: container.clientWidth / 2 - (node.x + node.width / 2) * ui.scale,
      translateY: container.clientHeight / 2 - (node.y + node.height / 2) * ui.scale,
    });
  }

  function render(view) {
    const runId = view?.identity?.runId;
    const ui = store.getExecutionUi(runId);
    const presentation = buildExecutionPresentation(view || {}, { locale: store.get().locale });
    const source = view?.graph || { tasks: {}, dependencies: {}, nodes: {} };
    const definitions = Object.fromEntries(presentation.tasks.map((task) => [task.id, {
      ...task.task,
      labelKey: task.labelKey,
      summaryFacts: task.summaryFacts,
      durationMs: task.durationMs,
      error: task.error,
    }]));
    const dependencies = Object.fromEntries(presentation.tasks.map((task) => [task.id, []]));
    for (const edge of presentation.dependencies) dependencies[edge.to].push(edge.from);
    const graph = buildExecutionGraph(definitions, dependencies, source.nodes);
    currentLayout = layoutExecutionGraph(graph, { direction: "LR" });

    const diagnostic = $("#graph-diagnostic");
    const messages = presentation.diagnostics.map((item) => item.code === "missing_dependency"
      ? `Missing dependency: ${item.taskId} ← ${item.dependencyId}`
      : item.code);
    diagnostic.hidden = !messages.length;
    diagnostic.textContent = messages.join(" ");
    for (const button of document.querySelectorAll("[data-execution-mode]")) {
      button.setAttribute("aria-pressed", String(button.dataset.executionMode === ui.mode));
    }
    $("#stage-overview").hidden = ui.mode !== "overview";
    $("#execution-graph").hidden = ui.mode !== "dag";
    $("#execution-timeline").hidden = ui.mode !== "timeline";
    $("#graph-tools").hidden = ui.mode !== "dag";
    $("#timeline-filter-control").hidden = ui.mode !== "timeline";
    $("#timeline-filter").value = ui.timelineFilter;
    $("#graph-scale").textContent = `${Math.round(ui.scale * 100)}%`;
    $("#graph-focus").disabled = !ui.selectedNodeId;
    document.querySelector(".app-shell").dataset.historyCollapsed = String(ui.historyCollapsed);

    renderStageOverview($("#stage-overview"), presentation, {
      expandedStageIds: ui.expandedStageIds,
      t,
      onToggle(stageId) {
        const expanded = new Set(store.getExecutionUi(runId).expandedStageIds);
        if (expanded.has(stageId)) expanded.delete(stageId); else expanded.add(stageId);
        store.setExecutionUi(runId, { expandedStageIds: [...expanded] });
      },
      onSelect: select,
    });
    renderExecutionGraph($("#execution-graph"), currentLayout, select, {
      scale: ui.scale,
      translateX: ui.translateX,
      translateY: ui.translateY,
      selectedId: ui.selectedNodeId,
      label: (key, fallback) => key ? t(key) : fallback,
      summary: (node) => {
        const fact = node.summaryFacts?.[0];
        return fact
          ? `${t(fact.labelKey)} ${Number(fact.value).toLocaleString()}`
          : t("outputNotRecorded");
      },
      status,
    });
    renderCallTimeline(
      $("#timeline"),
      buildCallTimeline(presentation, { category: ui.timelineFilter }),
      {
        locale: store.get().locale,
        onSelect: select,
        emptyText: t("emptyEvents"),
        showAllText: t("showAll"),
        durationUnit: t("secondsShort"),
      },
    );
    if (ui.mode === "dag"
      && ui.fittedSignature !== graph.signature
      && pendingFitSignature !== graph.signature) {
      pendingFitSignature = graph.signature;
      requestAnimationFrame(() => fit(graph.signature));
    }
  }

  function bind() {
    document.querySelectorAll("[data-execution-mode]").forEach((button) => {
      button.addEventListener("click", () => update({ mode: button.dataset.executionMode }));
    });
    $("#timeline-filter").addEventListener("change", (event) => {
      update({ timelineFilter: event.target.value });
    });
    $("#history-collapse").addEventListener("click", () => {
      const ui = store.getExecutionUi(store.get().view?.identity?.runId);
      update({ historyCollapsed: !ui.historyCollapsed });
    });
    $("#graph-zoom-in").addEventListener("click", () => zoom(0.1));
    $("#graph-zoom-out").addEventListener("click", () => zoom(-0.1));
    $("#graph-fit").addEventListener("click", () => fit());
    $("#graph-focus").addEventListener("click", focus);
    $("#graph-fullscreen").addEventListener("click", () => {
      $("#section-execution").classList.toggle("execution-fullscreen");
      requestAnimationFrame(() => fit());
    });
    const graph = $("#execution-graph");
    graph.addEventListener("pointerdown", (event) => {
      if (event.button !== 0 || event.target.closest(".graph-node")) return;
      const ui = store.getExecutionUi(store.get().view?.identity?.runId);
      graphPan = {
        pointerId: event.pointerId,
        x: event.clientX,
        y: event.clientY,
        startX: ui.translateX,
        startY: ui.translateY,
      };
      event.currentTarget.setPointerCapture(event.pointerId);
      event.currentTarget.dataset.panning = "true";
    });
    graph.addEventListener("pointermove", (event) => {
      if (!graphPan || graphPan.pointerId !== event.pointerId) return;
      const stage = graph.querySelector(".graph-stage");
      if (!stage) return;
      const ui = store.getExecutionUi(store.get().view?.identity?.runId);
      stage.style.transform = `translate(${graphPan.startX + event.clientX - graphPan.x}px, ${graphPan.startY + event.clientY - graphPan.y}px) scale(${ui.scale})`;
    });
    graph.addEventListener("pointerup", (event) => {
      if (!graphPan || graphPan.pointerId !== event.pointerId) return;
      const translateX = graphPan.startX + event.clientX - graphPan.x;
      const translateY = graphPan.startY + event.clientY - graphPan.y;
      graphPan = null;
      event.currentTarget.dataset.panning = "false";
      update({ translateX, translateY });
    });
    graph.addEventListener("wheel", (event) => {
      if (!event.ctrlKey && !event.metaKey) return;
      event.preventDefault();
      zoom(event.deltaY < 0 ? 0.1 : -0.1);
    }, { passive: false });
  }

  return { bind, render };
}
