import { createApiClient, openEventStream } from "./api-client.js";
import { renderBenchmarkDetail, renderBenchmarkList, renderBenchmarkObservation } from "./benchmark-view.js?v=15.10.3";
import { adaptArchiveView, adaptInteractiveView, chooseDefaultSection, normalizeRunList } from "./adapters.js";
import { evidenceMap, renderEvidenceList } from "./evidence-view.js";
import { createExecutionWorkspace } from "./execution-workspace.js";
import { renderExecutionInspector } from "./execution-inspector.js";
import { renderHistory } from "./history-view.js";
import { applyTranslations, translator } from "./i18n.js?v=15.10.3";
import { renderQuality } from "./quality-view.js";
import { renderOutline, renderReportDocument } from "./report-view.js";
import { renderReview } from "./review-view.js";
import { createWorkbenchStore } from "./state.js";

const $ = (selector) => document.querySelector(selector);
const locale = localStorage.getItem("litagent-locale") || "zh-CN";
const store = createWorkbenchStore({ locale });
let context = null;
let eventSource = null;
let inspectorTrigger = null;
const benchmarkState = { results: [], selectedId: null, detail: null, observation: null, loaded: false };
const api = createApiClient(() => context?.operation_token);
const executionWorkspace = createExecutionWorkspace({ store, t, onInspect: setInspector });

function t(key) { return translator(store.get().locale)(key); }
function setNotice(message = "", kind = "info") {
  const element = $("#notice");
  element.hidden = !message;
  element.dataset.kind = kind;
  element.textContent = message;
}

function setInspector(selection, trigger = null) {
  const drawer = $("#inspector-drawer");
  const content = $("#inspector-content");
  if (!selection) {
    drawer.dataset.open = "false";
    drawer.setAttribute("aria-hidden", "true");
    if (inspectorTrigger?.isConnected) inspectorTrigger.focus();
    inspectorTrigger = null;
    return;
  }
  inspectorTrigger = trigger;
  $("#inspector-title").textContent = selection.id || selection.kind;
  content.replaceChildren();
  if (renderExecutionInspector(content, selection, t)) {
    drawer.dataset.open = "true";
    drawer.setAttribute("aria-hidden", "false");
    $("#close-inspector").focus();
    return;
  }
  const description = document.createElement("dl");
  const fields = selection.kind === "evidence"
    ? ["evidence_id", "paper_id", "paper_title", "content_scope", "locator", "confidence", "text"]
    : [];
  if (fields.length) {
    for (const field of fields) {
      if (selection.value?.[field] == null || selection.value[field] === "") continue;
      const term = document.createElement("dt"); term.textContent = field;
      const value = document.createElement("dd"); value.textContent = String(selection.value[field]);
      description.append(term, value);
    }
  } else {
    const raw = document.createElement("pre");
    raw.textContent = JSON.stringify(selection.value ?? null, null, 2);
    description.append(raw);
  }
  content.append(description);
  drawer.dataset.open = "true";
  drawer.setAttribute("aria-hidden", "false");
  $("#close-inspector").focus();
}

function summaryValue(value, suffix = "") { return value == null || value === "" ? "—" : `${value}${suffix}`; }
function renderSummary(view) {
  const summary = view?.summary || {};
  $("#run-query").textContent = view?.identity?.query || t("noRun");
  $("#run-source-label").textContent = view ? `${view.identity.source} / ${summary.report_language || view.identity.reportLanguage || "—"}` : "LITAGENT";
  $("#summary-status").textContent = summaryValue(view?.identity?.status);
  $("#summary-delivery").textContent = summaryValue(view?.delivery?.status || summary.delivery_status);
  $("#summary-quality").textContent = summaryValue(view?.quality?.status || summary.quality_status);
  $("#summary-tokens").textContent = summary.total_tokens == null ? "—" : Number(summary.total_tokens).toLocaleString();
  $("#summary-elapsed").textContent = summary.elapsed_ms == null ? "—" : `${(Number(summary.elapsed_ms) / 1000).toFixed(1)}s`;
  $("#summary-run-id").textContent = summaryValue(view?.identity?.runId);
}

function renderReport(view) {
  const evidence = evidenceMap(view?.evidence || []);
  const projection = view?.renderedReport?.html
    ? view.renderedReport
    : { html: "", headings: [], plainText: view?.report?.survey || "" };
  if (projection.plainText && !projection.html) {
    $("#report-document").textContent = projection.plainText;
    $("#report-document").classList.add("render-fallback");
    $("#report-outline").replaceChildren();
    return;
  }
  $("#report-document").classList.remove("render-fallback");
  const result = renderReportDocument($("#report-document"), projection, evidence, t, (id, trigger) => setInspector({ kind: "evidence", id, value: evidence.get(id) }, trigger));
  renderOutline($("#report-outline"), result.headings, (heading) => {
    const target = document.getElementById(heading.anchor); if (!target) return; target.tabIndex = -1; target.scrollIntoView({ behavior: "smooth", block: "start" }); target.focus({ preventScroll: true });
  });
}

function renderControls(view) {
  const capabilities = view?.capabilities || {};
  $("#cancel-run").hidden = !capabilities.canCancel;
  $("#download-artifact").hidden = !capabilities.canDownloadArtifact;
  $("#trace-link").hidden = !capabilities.hasTrace;
  if (capabilities.hasTrace) $("#trace-link").href = view.trace.trace_url;
  const versions = view?.versions || [];
  const versionControl = $(".version-control"); versionControl.hidden = !versions.length;
  const select = $("#version-select"); const selected = select.value; select.replaceChildren();
  for (const version of versions) select.add(new Option(`v${version.revision} · ${version.delivery?.status || "—"}`, version.revision));
  if ([...select.options].some((option) => option.value === selected)) select.value = selected;
  $("#export-menu").hidden = !capabilities.canExport;
  $("#review-tab").dataset.pending = String(Boolean(view?.review));
}

function render() {
  const state = store.get(); const view = state.view;
  applyTranslations(document, t);
  $("#locale-toggle").textContent = state.locale === "zh-CN" ? "EN" : "中文";
  $("#run-count").textContent = state.runs.length;
  renderHistory($("#run-list"), state.runs, state.selectedRunKey, $("#run-search").value, state.locale, t, selectRun);
  renderSummary(view); renderControls(view);
  const benchmarkActive = state.activeSection === "benchmark";
  $("#run-summary").hidden = benchmarkActive;
  if (benchmarkActive) {
    $("#run-query").textContent = t("benchmarkResults");
    $("#run-source-label").textContent = "LITAGENT / BENCHMARK";
    for (const selector of ["#trace-link", "#download-artifact", "#cancel-run", ".version-control", "#export-menu"]) $(selector).hidden = true;
  }
  for (const tab of document.querySelectorAll("[data-section]")) tab.setAttribute("aria-current", tab.dataset.section === state.activeSection ? "page" : "false");
  for (const section of document.querySelectorAll(".workbench-section")) section.hidden = section.id !== `section-${state.activeSection}`;
  renderReport(view); executionWorkspace.render(view);
  $("#review-stage").textContent = view?.review?.stage || "";
  const requestId = view?.review?.request_id; const draft = state.draftReviewState[requestId] || {};
  renderReview($("#review-content"), view?.review, draft, t, submitReview, (nextDraft) => {
    if (requestId) state.draftReviewState[requestId] = nextDraft;
  });
  $("#evidence-count").textContent = view?.evidence?.length || 0;
  renderEvidenceList($("#evidence-list"), view?.evidence || [], t, setInspector);
  renderQuality($("#quality-content"), view?.runtimeEvaluation || {}, view?.quality, view?.delivery, setInspector);
  renderBenchmarkState();
}

function renderBenchmarkState() {
  renderBenchmarkList($("#benchmark-list"), benchmarkState.results, benchmarkState.selectedId, selectBenchmark, t);
  renderBenchmarkDetail($("#benchmark-detail"), benchmarkState.detail, selectBenchmarkObservation, t);
  if (benchmarkState.observation) renderBenchmarkObservation($("#benchmark-observation-detail"), benchmarkState.observation, t);
}

async function loadBenchmarks() {
  try {
    benchmarkState.results = await api.json("/workbench/benchmarks");
    benchmarkState.loaded = true;
    const preferred = benchmarkState.selectedId || benchmarkState.results[0]?.benchmark_id;
    if (preferred) await selectBenchmark(preferred);
    else renderBenchmarkState();
  } catch (error) { setNotice(`${t("operationFailed")}: ${error.message}`, "error"); }
}

async function selectBenchmark(id) {
  try {
    const detail = await api.json(`/workbench/benchmarks/${encodeURIComponent(id)}`);
    benchmarkState.selectedId = id;
    benchmarkState.detail = detail;
    benchmarkState.observation = null;
    renderBenchmarkState();
  } catch (error) { setNotice(`${t("operationFailed")}: ${error.message}`, "error"); }
}

async function selectBenchmarkObservation(id) {
  try {
    const payload = await api.json(`/workbench/benchmarks/${encodeURIComponent(benchmarkState.selectedId)}/observations/${encodeURIComponent(id)}`);
    benchmarkState.observation = payload;
    renderBenchmarkObservation($("#benchmark-observation-detail"), payload, t);
  } catch (error) { setNotice(`${t("operationFailed")}: ${error.message}`, "error"); }
}

function setSection(section, user = true) {
  store.update({ activeSection: section, sectionWasUserSelected: user || store.get().sectionWasUserSelected });
  const params = new URLSearchParams(location.search); params.set("view", section); history.replaceState(null, "", `?${params}`);
  if (section === "benchmark" && !benchmarkState.loaded) loadBenchmarks();
}

function connectEvents(runId, kind = "interactive") {
  eventSource?.close();
  eventSource = openEventStream(runId, {
    onTrace(event) {
      const view = store.get().view; if (!view || view.identity.runId !== runId) return;
      const sequence = event.seq ?? event.sequence;
      const source = event.source || "sse";
      const known = view.events.some((item) => (
        sequence != null
        && (item.seq ?? item.sequence) === sequence
        && (item.source || "sse") === source
      ));
      if (!known) store.update({ view: { ...view, events: [...view.events, { ...event, source }] } });
    },
    async onSettle() { if (kind === "interactive") await refreshInteractive(runId); else await refreshLegacy(runId, store.get().view?.identity?.query || ""); await loadRunList(); },
  });
}

async function refreshInteractive(runId, preferredSection = null) {
  const epoch = store.beginRequest();
  try {
    const raw = await api.json(`/survey/${encodeURIComponent(runId)}/view`); if (!store.isCurrent(epoch)) return;
    const identity = store.get().runs.find((item) => item.key === `interactive:${runId}`) || { key: `interactive:${runId}`, runId, source: "interactive", query: raw.summary?.query || "", status: raw.summary?.status || "unknown" };
    const view = adaptInteractiveView(raw, identity);
    const activeSection = preferredSection || (store.get().sectionWasUserSelected ? store.get().activeSection : chooseDefaultSection(view));
    store.update({ selectedRunKey: identity.key, view, activeSection, loading: false, error: null });
    if (["queued", "running", "cancelling"].includes(view.identity.status)) connectEvents(runId);
    return true;
  } catch (error) { if (store.isCurrent(epoch)) { store.update({ loading: false, error }); setNotice(`${t("operationFailed")}: ${error.message}`, "error"); } return false; }
}

async function refreshLegacy(runId, query) {
  const status = await api.json(`/survey/${encodeURIComponent(runId)}`); let report = null;
  if (["completed", "cancelled"].includes(status.status)) report = await api.json(`/survey/${encodeURIComponent(runId)}/report`);
  const identity = { key: `legacy:${runId}`, runId, source: "legacy", query, status: status.status };
  const view = { identity, summary: { run_id: runId, query, status: status.status }, report, renderedReport: { html: "", headings: [] }, graph: { tasks: {}, dependencies: {}, nodes: {} }, events: [], evidence: [], runtimeEvaluation: report?.evaluation || {}, quality: report?.quality, delivery: report?.delivery, review: null, versions: [], trace: {}, capabilities: { canCancel: ["queued", "running", "cancelling"].includes(status.status) } };
  store.update({ selectedRunKey: identity.key, view, activeSection: chooseDefaultSection(view), loading: false });
  if (view.capabilities.canCancel) connectEvents(runId, "legacy");
}

async function selectRun(key, preferredSection = null) {
  eventSource?.close(); api.abort(); store.update({ sectionWasUserSelected: Boolean(preferredSection) });
  const identity = store.get().runs.find((item) => item.key === key); if (!identity) return;
  if (identity.source === "interactive") await refreshInteractive(identity.runId, preferredSection);
  else {
    const epoch = store.beginRequest();
    try { const raw = await api.json(`/flow-demo/runs/${encodeURIComponent(identity.runId)}/view`); if (!store.isCurrent(epoch)) return; const view = adaptArchiveView(raw, identity); store.update({ selectedRunKey: key, view, activeSection: preferredSection || chooseDefaultSection(view), loading: false }); }
    catch (error) { if (store.isCurrent(epoch)) { store.update({ loading: false, error }); setNotice(`${t("operationFailed")}: ${error.message}`, "error"); } }
  }
  history.replaceState(null, "", `?run=${encodeURIComponent(key)}&view=${encodeURIComponent(store.get().activeSection)}`);
  $("#run-sidebar").dataset.open = "false"; $("#sidebar-scrim").hidden = true;
}

async function loadRunList(preferred = null) {
  const [interactive, archives] = await Promise.all([api.json("/workbench/runs"), api.json("/flow-demo/runs")]);
  const runs = normalizeRunList(interactive, archives); store.update({ runs });
  const key = preferred || store.get().selectedRunKey || runs[0]?.key; if (key && key !== store.get().selectedRunKey) await selectRun(key, new URLSearchParams(location.search).get("view"));
}

async function submitReview(action, values, form) {
  const view = store.get().view; const request = view?.review; if (!request) return;
  for (const button of form.querySelectorAll("button")) button.disabled = true;
  const payload = { request_id: request.request_id, revision: request.revision, action };
  if (action === "confirm") payload.selected_paper_ids = values.selectedPaperIds;
  if (action === "revise") payload.feedback = values.feedback;
  try { await api.json(`/survey/${encodeURIComponent(view.identity.runId)}/review/decision`, { method: "POST", mutate: true, body: payload }); if (!await refreshInteractive(view.identity.runId)) setNotice(t("submittedSyncFailed"), "error"); }
  catch (error) { setNotice(`${t("operationFailed")}: ${error.message}`, "error"); for (const button of form.querySelectorAll("button")) button.disabled = false; }
}

async function startRun(event) {
  event.preventDefault(); if (!context?.live_enabled) return setNotice(t("offline"), "error");
  const query = $("#query-input").value.trim(); if (!query) return;
  const humanReview = $("#human-review").checked;
  try {
    const created = await api.json("/survey", { method: "POST", mutate: true, body: { query, report_language: $("#report-language").value, human_review: humanReview } });
    $("#new-run-dialog").close();
    if (humanReview) { await loadRunList(`interactive:${created.task_id}`); } else { await refreshLegacy(created.task_id, query); }
  } catch (error) { setNotice(`${t("operationFailed")}: ${error.message}`, "error"); }
}

async function initialize() {
  context = await api.json("/workbench/context"); $("#runtime-mode").textContent = context.mode;
  $("#open-new-run").disabled = !context.live_enabled; $("#live-warning").textContent = context.live_enabled ? "" : t("offline");
  const params = new URLSearchParams(location.search); await loadRunList(params.get("run"));
  if (params.get("view") === "benchmark") setSection("benchmark", false);
}

store.subscribe(render);
document.querySelectorAll("[data-section]").forEach((tab) => tab.addEventListener("click", () => setSection(tab.dataset.section)));
$("#run-search").addEventListener("input", render);
$("#benchmark-refresh").addEventListener("click", loadBenchmarks);
$("#close-inspector").addEventListener("click", () => setInspector(null));
$("#open-new-run").addEventListener("click", () => $("#new-run-dialog").showModal());
$("#close-new-run").addEventListener("click", () => $("#new-run-dialog").close());
$("#new-run-form").addEventListener("submit", startRun);
$("#locale-toggle").addEventListener("click", () => { const next = store.get().locale === "zh-CN" ? "en" : "zh-CN"; localStorage.setItem("litagent-locale", next); store.update({ locale: next }); });
$("#cancel-run").addEventListener("click", async () => { const view = store.get().view; if (!view) return; await api.request(`/survey/${encodeURIComponent(view.identity.runId)}`, { method: "DELETE", mutate: true }); await refreshInteractive(view.identity.runId); });
$("#download-artifact").addEventListener("click", () => { const view = store.get().view; if (view) location.href = `/flow-demo/runs/${encodeURIComponent(view.identity.runId)}/download`; });
$("#version-select").addEventListener("change", async (event) => { const view = store.get().view; const previous = view.report; try { const version = await api.json(`/survey/${encodeURIComponent(view.identity.runId)}/versions/${event.target.value}`); store.update({ view: { ...view, report: version.report, renderedReport: version.rendered_report, runtimeEvaluation: version.report.evaluation || {}, quality: version.report.quality, delivery: version.report.delivery } }); } catch (error) { view.report = previous; setNotice(`${t("operationFailed")}: ${error.message}`, "error"); } });
document.querySelectorAll("[data-format]").forEach((button) => button.addEventListener("click", () => { const view = store.get().view; const revision = $("#version-select").value; if (view && revision) { const style = $("#export-style").value; location.href = `/survey/${encodeURIComponent(view.identity.runId)}/export?revision=${revision}&format=${button.dataset.format}&citation_style=${style}`; } }));
$("#sidebar-toggle").addEventListener("click", () => { $("#run-sidebar").dataset.open = "true"; $("#sidebar-scrim").hidden = false; });
$("#sidebar-scrim").addEventListener("click", () => { $("#run-sidebar").dataset.open = "false"; $("#sidebar-scrim").hidden = true; });
executionWorkspace.bind();

render();
initialize().catch((error) => setNotice(`${t("operationFailed")}: ${error.message}`, "error"));
