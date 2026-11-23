import { bindAuditControls, bindSurveyVerification } from "./audit-controls.js?v=15.10.6.434";
import { bindFormatControls } from "./format-controls.js?v=15.10.11";
import { createApiClient, openEventStream } from "./api-client.js?v=15.10.6.434";
import { createBenchmarkController } from "./benchmark-controller.js?v=15.10.6.163-summary-contract";
import { renderBenchmarkDetail, renderBenchmarkList, renderBenchmarkObservation } from "./benchmark-view.js?v=15.10.6.163-summary-contract";
import { adaptArchiveView, adaptInteractiveView, chooseDefaultSection, normalizeRunList } from "./adapters.js?v=15.10.6.434";
import { evidenceMap, renderEvidenceList, renderEvidenceInspector } from "./evidence-view.js?v=15.10.6.434";
import { buildEvidenceLedger, filterEvidence } from "./evidence-model.js?v=15.10.6.434";
import { createLiveViewRefresh } from "./live-view.js?v=15.10.6.434";
import { createExecutionWorkspace } from "./execution-workspace.js?v=15.10.6.434";
import { renderExecutionInspector } from "./execution-inspector.js?v=15.10.6.434";
import { renderHistory } from "./history-view.js?v=15.10.6.434";
import { applyTranslations, translator } from "./i18n.js?v=15.10.6.434";
import { renderQuality, locateReportDiagnostic } from "./quality-view.js?v=15.10.6.434";
import { navigateReportHeading, renderOutline, renderReportDocument, renderSearchRecords } from "./report-view.js?v=15.10.6.434";
import { renderReview } from "./review-view.js?v=15.10.5-strict";
import { createWorkbenchStore } from "./state.js?v=15.10.6.434";
const $ = (selector) => document.querySelector(selector);
const locale = localStorage.getItem("litagent-locale") || "zh-CN";
const store = createWorkbenchStore({ locale });
let context = null;
let independentAuditState = null;
const citationPreferences = new Map();
let eventSource = null;
let inspectorTrigger = null;
let activeInspection = null;
let inspectedView = null;
let inspectedLocale = null;
let historyLoading = false;
let submittingSurvey = false;
const mobileSidebar = globalThis.matchMedia?.("(max-width: 760px)");
const benchmarkState = { results: [], selectedId: null, detail: null, observation: null, loaded: false };
const api = createApiClient(() => context?.operation_token);
const executionWorkspace = createExecutionWorkspace({ store, t, onInspect: setInspector });
const surveyVerification = bindSurveyVerification({ capabilities: () => ({...context?.auto_audit_capabilities, default_generation_strategy:context?.automatic_acceptance?.default_generation_strategy, default_budget_policy:context?.automatic_acceptance?.default_budget_policy, maximum_survey_tokens:context?.automatic_acceptance?.maximum_survey_tokens, default_verification:context?.automatic_acceptance?.default_verification, maximum_survey_requests:context?.automatic_acceptance?.maximum_survey_requests}), t });
function t(key) { return translator(store.get().locale)(key); }
function setNotice(message = "", kind = "info") {
  const element = $("#notice");
  element.hidden = !message;
  element.dataset.kind = kind;
  element.textContent = message;
}
function closeRunSidebar(restoreFocus = false) {
  $("#run-sidebar").dataset.open = "false";
  $("#sidebar-scrim").hidden = true;
  $("#sidebar-toggle").setAttribute("aria-expanded", "false");
  if (restoreFocus && (!mobileSidebar || mobileSidebar.matches)) $("#sidebar-toggle").focus();
}
function setInspector(selection, trigger = null, refresh = false) {
  const drawer = $("#inspector-drawer");
  const content = $("#inspector-content");
  if (!selection) {
    const previous = activeInspection;
    activeInspection = null;
    drawer.dataset.open = "false";
    drawer.inert = true;
    drawer.setAttribute("aria-hidden", "true");
    if (inspectorTrigger?.isConnected) inspectorTrigger.focus();
    else if (previous?.kind === "node") [...document.querySelectorAll(".graph-node")]
      .find(node => node.dataset.nodeId === previous.id)?.focus();
    inspectorTrigger = null;
    return;
  }
  activeInspection = selection;
  drawer.inert = false;
  inspectedView = store.get().view;
  inspectedLocale = store.get().locale;
  inspectorTrigger = trigger;
  $("#inspector-title").textContent = selection.id || selection.kind;
  content.replaceChildren();
  if (renderExecutionInspector(content, selection, t, { view: store.get().view, onSelect: setInspector, onOpenReport: () => { setInspector(null); store.update({ activeSection: "report" }); } })
    || renderEvidenceInspector(content, selection, t, buildEvidenceLedger(store.get().view || {}))) {
    drawer.dataset.open = "true";
    drawer.setAttribute("aria-hidden", "false");
    if (!refresh) $("#close-inspector").focus();
    return;
  }
  if (selection.kind === "quality-diagnostic") {
    const quote = document.createElement("blockquote"); quote.textContent = selection.value?.source_quote || "—"; content.append(quote);
    const explanation = document.createElement("p"); explanation.textContent = selection.value?.explanation || ""; content.append(explanation);
    const locate = document.createElement("button"); locate.type = "button"; locate.textContent = store.get().locale === "zh-CN" ? "跳转原文" : "Open report source";
    locate.addEventListener("click", () => {
      store.update({activeSection:"report"});
      if (!locateReportDiagnostic($("#report-document"), store.get().view?.report?.survey || "", selection.value)) setNotice(store.get().locale === "zh-CN" ? "无法唯一定位；请核对所示原文与引用。" : "Source location is not unique; inspect the exact quote and citations.", "error");
    }); content.append(locate);
    for (const id of selection.value?.evidence_ids || []) { const button=document.createElement("button");button.type="button";button.textContent=`[E:${id}]`;button.addEventListener("click",()=>{const item=evidenceMap(store.get().view?.evidence || []).get(`E:${id}`);if(item)setInspector({kind:"evidence",id,value:item},button);else setNotice(store.get().locale === "zh-CN" ? `证据不可用：${id}` : `Evidence unavailable: ${id}`,"error");});content.append(button); }
    drawer.dataset.open="true";drawer.setAttribute("aria-hidden","false");if (!refresh) $("#close-inspector").focus();return;
  }
  if (selection.citationScope?.source_quote) {
    const quote = document.createElement("blockquote");
    quote.textContent = selection.citationScope.source_quote.replace(/\[E:[^\]]+\]/g, "");
    content.append(quote);
  }
  const description = document.createElement("dl");
  const fields = selection.kind === "evidence"
    ? ["evidence_id", "paper_id", "paper_title", "content_scope", "locator", "confidence", "text", "supporting_text"]
    : [];
  if (fields.length) {
    for (const item of selection.references || [selection.value]) {
      for (const field of fields) {
        if (item?.[field] == null || item[field] === "") continue;
        const term = document.createElement("dt"); term.textContent = field;
        const value = document.createElement("dd"); value.textContent = String(item[field]);
        description.append(term, value);
      }
    }
  } else {
    const raw = document.createElement("pre");
    raw.textContent = JSON.stringify(selection.value ?? null, null, 2);
    description.append(raw);
  }
  content.append(description);
  drawer.dataset.open = "true";
  drawer.setAttribute("aria-hidden", "false");
  if (!refresh) $("#close-inspector").focus();
}

function summaryValue(value, suffix = "") { return value == null || value === "" ? "—" : `${value}${suffix}`; }
function renderSummary(view) {
  const summary = view?.summary || {};
  $("#run-query").textContent = view?.identity?.query || t("noRun");
  const source = view?.identity?.source === "legacy" ? t("flowLiveRun") : view?.identity?.source;
  $("#run-source-label").textContent = view ? `${source} / ${summary.report_language || view.identity.reportLanguage || "—"}` : "LITAGENT";
  $("#summary-status").textContent = view?.identity?.status === "completed" ? t("executionCompleted") : summaryValue(view?.identity?.status);
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
  const result = renderReportDocument($("#report-document"), projection, evidence, t, (ids, trigger, citationScope) => {
    const references = Array.isArray(ids) ? ids : [ids];
    setInspector({ kind: "evidence", citationScope, id: references.join(", "), value: evidence.get(references[0]), references: references.map(id => evidence.get(id)).filter(Boolean) }, trigger);
  });
  renderOutline($("#report-outline"), result.headings, (heading) => {
    navigateReportHeading($("#report-document"), heading, $("#section-report"));
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
  const select = $("#version-select"); const selected = String(view?.displayRevision || versions[0]?.revision || 1); select.replaceChildren();
  for (const version of versions) select.add(new Option(`v${version.revision} · ${version.delivery?.status || "—"}`, version.revision));
  if ([...select.options].some((option) => option.value === selected)) select.value = selected;
  $("#export-menu").hidden = !capabilities.canExport;
  $("#citation-control").hidden = !view?.report?.survey;
  const style = citationPreferences.get(view?.identity.runId) || view?.renderedReport?.citation_style || "numbered";
  $("#export-style").value = style;
  const warnings = view?.renderedReport?.warnings || [];
  $("#format-warning").hidden = !warnings.length;
  const warningTypes = [...new Set(warnings.map(item => item.split(":")[0]))].map(code => t(`warning_${code}`));
  $("#format-warning").textContent = warnings.length ? t("metadataWarning") + warnings.length + t("metadataWarningEnd") + " " + warningTypes.join(" · ") : "";
  $("#review-tab").dataset.pending = String(Boolean(view?.review));
}

function render() {
  const state = store.get(); const view = state.view;
  applyTranslations(document, t);
  surveyVerification.render();
  document.documentElement.lang = state.locale;
  $("#locale-toggle").textContent = state.locale === "zh-CN" ? "EN" : "中文";
  $("#run-count").textContent = state.runs.length;
  renderHistory($("#run-list"), state.runs, state.selectedRunKey, $("#run-search").value, state.locale, t, selectRun, view?.identity?.status);
  renderSummary(view); renderControls(view);
  if (view?.displayWarnings?.some((item) => item.code === "nonfinite_numeric_values")) {
    setNotice(store.get().locale === "zh-CN"
      ? "本次运行的重排分数存在无效值，页面显示为缺失。原始记录已保留；自动评估通过不代表排序质量已验证。"
      : "This run contains invalid ranking scores, shown as missing. The original record is preserved; passing evaluation does not validate ranking quality.", "error");
  }
  const benchmarkActive = state.activeSection === "benchmark";
  $("#run-summary").hidden = benchmarkActive;
  if (benchmarkActive) {
    $("#run-query").textContent = t("benchmarkResults");
    $("#run-source-label").textContent = "LITAGENT / BENCHMARK";
    for (const selector of ["#trace-link", "#download-artifact", "#cancel-run", ".version-control", "#export-menu", "#citation-control", "#format-warning"]) $(selector).hidden = true;
  }
  for (const tab of document.querySelectorAll("[data-section]")) tab.setAttribute("aria-current", tab.dataset.section === state.activeSection ? "page" : "false");
  for (const section of document.querySelectorAll(".workbench-section")) section.hidden = section.id !== `section-${state.activeSection}`;
  renderReport(view); executionWorkspace.render(view);
  $("#review-stage").textContent = view?.review?.stage || "";
  const requestId = view?.review?.request_id; const draft = state.draftReviewState[requestId] || {};
  renderReview($("#review-content"), view?.review, draft, t, submitReview, (nextDraft) => {
    if (requestId) state.draftReviewState[requestId] = nextDraft;
  });
  renderEvidenceContent();
  if (activeInspection && (inspectedView !== view || inspectedLocale !== state.locale)) {
    if (!view || inspectedView?.identity?.key !== view.identity?.key
      || inspectedView?.displayRevision !== view.displayRevision
      || (["evidence", "quality-diagnostic"].includes(activeInspection.kind)
        && inspectedView?.reportSha !== view.reportSha)) setInspector(null);
    else {
      const content = $("#inspector-content"); const scroll = content.scrollTop;
      const open = [...content.querySelectorAll("details")].map(item => item.open);
      setInspector(activeInspection, inspectorTrigger, true);
      content.querySelectorAll("details").forEach((item, index) => { if (open[index] != null) item.open = open[index]; });
      content.scrollTop = scroll;
    }
  }
  renderSearchRecords($("#search-records"), view?.renderedReport, t);
  renderQualityContent();
  renderBenchmarkState();
}

function renderEvidenceContent() {
  const items = buildEvidenceLedger(store.get().view || {});
  const filtered = filterEvidence(items, $("#evidence-search").value || "", $("#evidence-filter").value || "all");
  $("#evidence-count").textContent = `${filtered.length} / ${items.length}`;
  renderEvidenceList($("#evidence-list"), filtered, t, setInspector);
}

function renderQualityContent() {
  const state = store.get(); const view = state.view;
  const revision = Number(view?.displayRevision || view?.versions?.[0]?.revision || 1);
  const snapshot = independentAuditState?.target && view?.identity && independentAuditState.target.runId === view.identity.runId && independentAuditState.target.revision === revision ? independentAuditState : null;
  renderQuality($("#quality-content"), view?.runtimeEvaluation || {}, view?.quality, view?.delivery, setInspector, t, { budgetPlan: view?.budgetPlan, budgetState: view?.budgetState, runId: view?.identity?.runId, locale: state.locale, revision: view?.report?.survey ? revision : null, sha: snapshot?.data?.source_report_sha256 || view?.reportSha, audits: snapshot, reportJudgeRetired: true, qualityPolicy: view?.report?.metadata?.quality_policy, verification: view?.report?.metadata?.verification, bodyDiagnostics: view?.report?.metadata?.body_diagnostics, analysisPlan: view?.report?.metadata?.analysis_plan, pipelineVerification: view?.pipelineVerification, extractionVerification: view?.report?.metadata?.extraction_verification, automaticAcceptance: view?.report?.metadata?.automatic_acceptance, evaluationHistory: view?.report?.metadata?.evaluation_history, reviewWorkflow: view?.report?.metadata?.review_workflow, providerFailures: view?.report?.metadata?.provider_failures, execution: view?.report?.metadata?.execution, writerFailure: view?.report?.metadata?.writer_failure, writerFormat: view?.report?.metadata?.writer_format, writerContract: view?.report?.metadata?.writing_bindings?.writer_contract, targetedRepairState: view?.report?.metadata?.targeted_repair_state || (view?.report?.survey ? null : view?.targetedRepairState) });
}

function renderBenchmarkState() {
  renderBenchmarkList($("#benchmark-list"), benchmarkState.results, benchmarkState.selectedId, selectBenchmark, t);
  renderBenchmarkDetail($("#benchmark-detail"), benchmarkState.detail, selectBenchmarkObservation, t);
  if (benchmarkState.observation) renderBenchmarkObservation($("#benchmark-observation-detail"), benchmarkState.observation, t);
}

const { load: loadBenchmarks, select: selectBenchmark, selectObservation: selectBenchmarkObservation } = createBenchmarkController({
  api, state: benchmarkState, render: renderBenchmarkState,
  renderObservation: payload => renderBenchmarkObservation($("#benchmark-observation-detail"), payload, t),
  clearObservation: () => $("#benchmark-observation-detail")?.replaceChildren(),
  onError: error => setNotice(`${t("operationFailed")}: ${error.message}`, "error"),
});

function setSection(section, user = true) {
  store.update({ activeSection: section, sectionWasUserSelected: user || store.get().sectionWasUserSelected });
  const params = new URLSearchParams(location.search); params.set("view", section); history.replaceState(null, "", `?${params}`);
  if (section === "benchmark" && !benchmarkState.loaded) loadBenchmarks();
}

const liveViewRefresh = createLiveViewRefresh({
  current: () => store.get(),
  async load(identity) {
    const prefix = identity.source === "interactive" ? "/survey" : "/flow-demo/runs";
    const style = citationPreferences.get(identity.runId);
    const raw = await api.json(`${prefix}/${encodeURIComponent(identity.runId)}/view${style ? `?citation_style=${encodeURIComponent(style)}` : ""}`, { navigation: true });
    if (raw.summary?.run_id !== identity.runId) throw new Error("live_run_identity_mismatch");
    const view = identity.source === "interactive" ? adaptInteractiveView(raw, identity) : adaptArchiveView(raw, identity);
    if (identity.source === "legacy") view.capabilities.canCancel = ["queued", "running", "cancelling"].includes(view.identity.status);
    return view;
  },
  apply: view => store.update({ view }),
  onError: () => setNotice(t("liveRefreshFailed"), "error"),
});

function connectEvents(runId, kind = "interactive") {
  liveViewRefresh.cancel();
  eventSource?.close();
  const key = store.get().selectedRunKey;
  eventSource = openEventStream(runId, {
    onTrace(event) {
      const view = store.get().view; if (!view || view.identity.runId !== runId || store.get().selectedRunKey !== key) return;
      liveViewRefresh.receive(event);
    },
    onError() { if (store.get().selectedRunKey === key) liveViewRefresh.schedule(); },
    async onSettle() {
      if (store.get().selectedRunKey !== key) return;
      liveViewRefresh.cancel();
      if (kind === "interactive") await refreshInteractive(runId);
      else await refreshLegacy(runId, store.get().view?.identity?.query || "");
      await loadRunList();
    },
  });
}

async function refreshInteractive(runId, preferredSection = null) {
  const epoch = store.beginRequest();
  try {
    const raw = await api.json(`/survey/${encodeURIComponent(runId)}/view${citationPreferences.has(runId) ? "?citation_style=" + citationPreferences.get(runId) : ""}`, { navigation: true }); if (!store.isCurrent(epoch)) return false;
    const identity = store.get().runs.find((item) => item.key === `interactive:${runId}`) || { key: `interactive:${runId}`, runId, source: "interactive", query: raw.summary?.query || "", status: raw.summary?.status || "unknown" };
    const view = adaptInteractiveView(raw, identity);
    const activeSection = preferredSection || (store.get().sectionWasUserSelected ? store.get().activeSection : chooseDefaultSection(view));
    store.update({ selectedRunKey: identity.key, view, activeSection, loading: false, error: null });
    if (["queued", "running", "cancelling"].includes(view.identity.status)) connectEvents(runId);
    return true;
  } catch (error) { if (store.isCurrent(epoch)) { store.update({ loading: false, error }); setNotice(`${t("operationFailed")}: ${error.message}`, "error"); } return false; }
}

async function refreshLegacy(runId, query, preferredSection = null) {
  liveViewRefresh.cancel();
  eventSource?.close();
  const epoch = store.beginRequest();
  try {
    const raw = await api.json(`/flow-demo/runs/${encodeURIComponent(runId)}/view${citationPreferences.has(runId) ? "?citation_style=" + citationPreferences.get(runId) : ""}`, { navigation: true });
    if (!store.isCurrent(epoch)) return false;
    if (raw.summary?.run_id !== runId) throw new Error("live_run_identity_mismatch");
    const status = raw.summary?.status || "unknown";
    const active = ["queued", "running", "cancelling"].includes(status);
    const source = active ? "legacy" : "archive";
    const identity = { key: `${source}:${runId}`, runId, source, query: raw.summary?.query || query, status };
    const view = adaptArchiveView(raw, identity);
    view.capabilities.canCancel = active;
    const activeSection = preferredSection || (store.get().sectionWasUserSelected ? store.get().activeSection : chooseDefaultSection(view));
    const runs = [identity, ...store.get().runs.filter(item => !(item.runId === runId && ["legacy", "archive"].includes(item.source)))];
    store.update({ selectedRunKey: identity.key, runs, view, activeSection, loading: false, error: null });
    history.replaceState(null, "", `?run=${encodeURIComponent(identity.key)}&view=${encodeURIComponent(activeSection)}`);
    if (active) connectEvents(runId, "legacy");
    return true;
  } catch (error) {
    if (store.isCurrent(epoch)) {
      store.update({ loading: false, error });
      if (error.name !== "AbortError") setNotice(`${t("operationFailed")}: ${error.message}`, "error");
    }
    return false;
  }
}

async function selectRun(key, preferredSection = null) {
  liveViewRefresh.cancel();
  setInspector(null);
  eventSource?.close(); api.abort(); store.update({ sectionWasUserSelected: Boolean(preferredSection) });
  const identity = store.get().runs.find((item) => item.key === key); if (!identity) return false;
  $("#evidence-search").value = "";
  $("#evidence-filter").value = "all";
  store.update({ selectedRunKey: key, view: null, inspector: null, loading: true });
  if (identity.source === "interactive") {
    if (!await refreshInteractive(identity.runId, preferredSection)) return false;
  }
  else if (identity.source === "legacy") {
    if (!await refreshLegacy(identity.runId, identity.query, preferredSection)) return false;
    key = store.get().selectedRunKey;
  }
  else {
    const epoch = store.beginRequest();
    try {
      const raw = await api.json(`/flow-demo/runs/${encodeURIComponent(identity.runId)}/view${citationPreferences.has(identity.runId) ? "?citation_style=" + citationPreferences.get(identity.runId) : ""}`, { navigation: true });
      if (!store.isCurrent(epoch)) return false;
      if (raw.summary?.run_id && raw.summary.run_id !== identity.runId) throw new Error("archive_run_identity_mismatch");
      const resolved = { ...identity, query: raw.summary?.query || identity.query, status: raw.summary?.status || identity.status };
      if (raw.source === "live" && ["queued", "running", "cancelling"].includes(resolved.status)) {
        resolved.source = "legacy"; resolved.key = `legacy:${identity.runId}`;
      }
      const view = adaptArchiveView(raw, resolved);
      view.capabilities.canCancel = resolved.source === "legacy";
      store.update({ selectedRunKey: resolved.key, runs: store.get().runs.map(item => item.key === key ? resolved : item), view, activeSection: preferredSection || chooseDefaultSection(view), loading: false });
      key = resolved.key;
      if (view.capabilities.canCancel) connectEvents(identity.runId, "legacy");
    } catch (error) {
      if (!store.isCurrent(epoch)) return false;
      store.update({ loading: false, error });
      if (error.name !== "AbortError") setNotice(`${t("operationFailed")}: ${error.message}`, "error");
    }
  }
  if (store.get().selectedRunKey !== key) return false;
  history.replaceState(null, "", `?run=${encodeURIComponent(key)}&view=${encodeURIComponent(store.get().activeSection)}`);
  closeRunSidebar();
  return Boolean(store.get().view);
}

async function loadRunList(preferred = null) {
  const selection = store.get().selectedRunKey, epoch = store.get().requestEpoch;
  const [interactive, archives] = await Promise.all([api.json("/workbench/runs"), api.json("/flow-demo/runs")]);
  const runs = normalizeRunList(interactive, archives);
  const selected = store.get().runs.find(item => item.key === store.get().selectedRunKey);
  if (selected && !runs.some(item => item.key === selected.key)) runs.unshift(selected);
  store.update({ runs });
  if (submittingSurvey || !store.isCurrent(epoch) || selection !== store.get().selectedRunKey) return;
  const key = preferred || store.get().selectedRunKey || runs[0]?.key;
  if (key && key !== store.get().selectedRunKey) await selectRun(key, new URLSearchParams(location.search).get("view"));
}

async function reloadHistory() {
  if (historyLoading) return;
  historyLoading = true; $("#load-run-history").disabled = true;
  try { await loadRunList(); }
  catch (error) { if (error.name !== "AbortError") setNotice(`${t("operationFailed")}: ${error.message}`, "error"); }
  finally { historyLoading = false; $("#load-run-history").disabled = false; }
}

async function submitReview(action, values, form) {
  const view = store.get().view; const request = view?.review; if (!request) return;
  for (const button of form.querySelectorAll("button")) button.disabled = true;
  const payload = { request_id: request.request_id, revision: request.revision, action };
  if (action === "confirm") payload.selected_paper_ids = values.selectedPaperIds;
  if (action === "revise") payload.feedback = values.feedback;
  const stillSelected = () => store.get().selectedRunKey === view.identity.key;
  try {
    await api.json(`/survey/${encodeURIComponent(view.identity.runId)}/review/decision`, { method: "POST", mutate: true, body: payload });
    if (stillSelected() && !await refreshInteractive(view.identity.runId) && stillSelected()) {
      setNotice(t("submittedSyncFailed"), "error");
    }
  } catch (error) {
    if (stillSelected()) setNotice(`${t("operationFailed")}: ${error.message}`, "error");
    for (const button of form.querySelectorAll("button")) button.disabled = false;
  }
}

async function startRun(event) {
  event.preventDefault();
  if (submittingSurvey) return;
  if (!context?.live_enabled) return setNotice(t("offline"), "error");
  const query = $("#query-input").value.trim(); if (!query) return;
  const qualityPolicy = $("#quality-policy").value;
  const humanReview = qualityPolicy === "strict_v1" || (!["strict_auto_v1", "strict_auto_v2"].includes(qualityPolicy) && $("#human-review").checked);
  const citationStyle = $("#new-citation-style").value;
  const submit = $('#new-run-form button[type="submit"]');
  const notice = $("#survey-submit-status");
  submittingSurvey = true;
  submit.disabled = true;
  notice.hidden = false;
  notice.textContent = t("surveySubmitting");
  let created;
  try {
    const response = await api.request("/survey", { method: "POST", mutate: true, body: { query, quality_policy: qualityPolicy, report_language: $("#report-language").value, human_review: humanReview, review_structure: $("#review-structure").value, citation_style: citationStyle, ...surveyVerification.options() } });
    created = await response.json();
    if (response.status !== 202 || typeof created?.task_id !== "string" || !created.task_id.trim()) {
      throw new Error("survey_receipt_invalid");
    }
  } catch (error) {
    const rejected = error.status >= 400 && error.status < 500 && error.status !== 408;
    notice.textContent = rejected ? `${t("operationFailed")}: ${error.message}` : t("surveyReceiptUnknown");
    if (!$("#new-run-dialog").open) setNotice(notice.textContent, "error");
    return;
  } finally {
    submittingSurvey = false;
    submit.disabled = false;
  }
  notice.hidden = true;
  const runId = created.task_id;
  const source = humanReview ? "interactive" : "legacy";
  const key = `${source}:${runId}`;
  const receipt = { key, runId, source, query, status: created.status || "queued" };
  citationPreferences.set(runId, citationStyle);
  store.update({ runs: [receipt, ...store.get().runs.filter(run => run.runId !== runId)] });
  history.replaceState(null, "", `?run=${encodeURIComponent(key)}&view=execution`);
  $("#new-run-dialog").close();
  if (!await selectRun(key, "execution") && store.get().selectedRunKey === key) {
    setNotice(`${t("surveyViewUnavailable")} (${runId})`, "error");
  }
}
async function initialize() {
  context = await api.json("/workbench/context"); $("#runtime-mode").textContent = context.mode;
  surveyVerification.render();
  $("#open-new-run").disabled = !context.live_enabled; $("#live-warning").textContent = context.live_enabled ? "" : t("offline");
  if (context.format_capabilities?.academic_engine_available === false) {
    for (const select of [$("#new-citation-style"), $("#export-style")]) for (const option of select.options) if (!["numbered", "evidence"].includes(option.value)) option.disabled = true;
    $("#new-citation-style").value = "numbered"; setNotice(t("engineUnavailable"), "error");
  }
  const params = new URLSearchParams(location.search), requested = params.get("run");
  const direct = /^(archive|sample|legacy|interactive):(.+)$/.exec(requested || "");
  if (direct) {
    store.update({ runs: [{ key: requested, runId: direct[2], source: direct[1], query: "", status: "unknown" }] });
    await selectRun(requested, params.get("view"));
  } else await loadRunList(requested);
  if (params.get("view") === "benchmark") setSection("benchmark", false);
}
store.subscribe(render);
document.querySelectorAll("[data-section]").forEach((tab) => tab.addEventListener("click", () => setSection(tab.dataset.section)));
$("#run-search").addEventListener("input", render);
$("#load-run-history").addEventListener("click", reloadHistory);
$("#benchmark-refresh").addEventListener("click", loadBenchmarks);
$("#close-inspector").addEventListener("click", () => setInspector(null));
$("#evidence-search").addEventListener("input", renderEvidenceContent);
$("#evidence-filter").addEventListener("change", renderEvidenceContent);
$("#open-new-run").addEventListener("click", () => $("#new-run-dialog").showModal());
$("#close-new-run").addEventListener("click", () => $("#new-run-dialog").close());
$("#dismiss-new-run").addEventListener("click", () => $("#new-run-dialog").close());
$("#new-run-form").addEventListener("submit", startRun);
$("#quality-policy").addEventListener("change", () => {
  const strict = $("#quality-policy").value === "strict_v1";
  $("#human-review").disabled = strict || ["strict_auto_v1", "strict_auto_v2"].includes($("#quality-policy").value);
  $("#human-review").checked = strict;
  surveyVerification.render();
});
$("#locale-toggle").addEventListener("click", () => { const next = store.get().locale === "zh-CN" ? "en" : "zh-CN"; localStorage.setItem("litagent-locale", next); store.update({ locale: next }); });
$("#cancel-run").addEventListener("click", async () => { const view = store.get().view; if (!view) return; await api.request(`/survey/${encodeURIComponent(view.identity.runId)}`, { method: "DELETE", mutate: true }); if (store.get().view?.identity.key !== view.identity.key) return; if (view.identity.source === "interactive") await refreshInteractive(view.identity.runId); else await refreshLegacy(view.identity.runId, view.identity.query); });
$("#download-artifact").addEventListener("click", () => { const view = store.get().view; if (view) location.href = `/flow-demo/runs/${encodeURIComponent(view.identity.runId)}/download`; });
bindFormatControls({ store, api, t, setNotice, citationPreferences });
bindAuditControls({ store, api, onSource: claim => {if (!locateReportDiagnostic($("#report-document"), store.get().view?.report?.survey || "", claim)) setNotice(store.get().locale === "zh-CN" ? "无法唯一定位所示原文。" : "Cannot uniquely locate this source quote.", "error");}, onSnapshot: snapshot => { independentAuditState = snapshot; renderQualityContent(); }, onEvidence: (id, button) => {
  const view = store.get().view; const normalized = id.startsWith("E:") ? id : `E:${id}`;
  const item = evidenceMap(view?.evidence || []).get(normalized);
  if (item) setInspector({kind: "evidence", id: normalized, value: item}, button);
  else setNotice(store.get().locale === "zh-CN" ? `证据不可用：${id}` : `Evidence unavailable: ${id}`, "error");
} });
$("#sidebar-toggle").addEventListener("click", () => {
  if (mobileSidebar && !mobileSidebar.matches) return;
  $("#run-sidebar").dataset.open = "true";
  $("#sidebar-scrim").hidden = false;
  $("#sidebar-toggle").setAttribute("aria-expanded", "true");
});
$("#sidebar-scrim").addEventListener("click", () => closeRunSidebar(true));
mobileSidebar?.addEventListener("change", event => { if (!event.matches) closeRunSidebar(); });
document.addEventListener("keydown", event => {
  if (event.key !== "Escape" || document.querySelector("dialog[open]")) return;
  if ($("#inspector-drawer").dataset.open === "true") { event.preventDefault(); setInspector(null); return; }
  if ($("#run-sidebar").dataset.open !== "true") return;
  event.preventDefault(); closeRunSidebar(true);
});
closeRunSidebar();
executionWorkspace.bind();
render();
initialize().catch((error) => setNotice(`${t("operationFailed")}: ${error.message}`, "error"));
