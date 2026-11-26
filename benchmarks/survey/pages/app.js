import { renderExecutionInspector } from "./workbench/execution-inspector.js";
import { createExecutionWorkspace } from "./workbench/execution-workspace.js";
import { applyTranslations, translator } from "./workbench/i18n.js";
import { renderReportDocument } from "./workbench/report-view.js";
import { createWorkbenchStore } from "./workbench/state.js";

const copy = {
  "zh-CN": {
    publishedExecution: "执行归档",
    archiveRun: "运行",
    archiveBenchmark: "基准总表",
    archiveMetrics: "运行指标",
    archiveViews: "运行视图",
    archiveExecutionTab: "执行过程",
    archiveReportTab: "报告正文",
    archiveReportSource: "原始 Markdown",
    archiveReportPartial: "归档中只有部分报告内容。",
    archiveReportMissing: "本次运行没有已归档的 Survey 报告。",
    archiveInputMissing: "此步骤没有已归档的输入。",
    archiveOutputMissing: "此步骤没有已归档的输出。",
    archiveIOReused: "以下记录来自复用的执行步骤。",
    archiveIOPlanned: "以下内容来自执行计划。",
    archiveIOOmitted: "公开归档省略了部分字段。",
    archiveIOWithheld: "此项内容未公开。",
    outputNotRecorded: "查看步骤详情",
    archiveGraph: "Agent 任务依赖图",
    recordedStages: "阶段与里程碑耗时",
    archiveLoading: "正在载入执行归档…",
    archiveSelect: "选择一条运行",
    archiveUnknown: "此运行不在公开归档中，请从列表选择。",
    archiveUnavailable: "归档暂时无法载入，请刷新页面重试。",
    strictDelivery: "严格交付",
    citationAccuracyMetric: "引用 ID 有效性",
    consistencyMetric: "内部一致性",
    recordingPartial: "部分",
    recordingReconciled: "已对账",
    workbench_final_report: "提交到终稿可读",
    workbench_terminal: "提交到终态可见",
    controlled_downstream: "受控回放耗时",
    summary_checkpoint: "摘要续跑耗时",
    archived_elapsed: "归档总耗时",
    first_progress: "提交到首个有效步骤",
    first_draft: "提交到首份可读草稿",
    terminal: "提交到终态可见",
    final_report: "提交到终稿可读",
    ready: "通过",
    blocked: "未通过",
    partial: "部分完成",
  },
  en: {
    publishedExecution: "Execution archive",
    archiveRun: "Run",
    archiveBenchmark: "Benchmark results",
    archiveMetrics: "Run metrics",
    archiveViews: "Run views",
    archiveExecutionTab: "Execution",
    archiveReportTab: "Survey report",
    archiveReportSource: "Original Markdown",
    archiveReportPartial: "The archive contains a partial report.",
    archiveReportMissing: "This run has no archived Survey report.",
    archiveInputMissing: "This step has no archived input.",
    archiveOutputMissing: "This step has no archived output.",
    archiveIOReused: "These records come from a reused step.",
    archiveIOPlanned: "These fields come from the execution plan.",
    archiveIOOmitted: "The public archive omits some fields.",
    archiveIOWithheld: "This content was withheld from the public archive.",
    outputNotRecorded: "Open step details",
    archiveGraph: "Agent task dependency graph",
    recordedStages: "Stage durations and milestones",
    archiveLoading: "Loading execution archive…",
    archiveSelect: "Select a run",
    archiveUnknown: "Choose a run from the public archive.",
    archiveUnavailable: "The archive could not load. Refresh to try again.",
    strictDelivery: "Strict delivery",
    citationAccuracyMetric: "Citation ID validity",
    consistencyMetric: "Internal consistency",
    recordingPartial: "partial",
    recordingReconciled: "reconciled",
    workbench_final_report: "Submit to readable report",
    workbench_terminal: "Submit to visible terminal state",
    controlled_downstream: "Controlled replay duration",
    summary_checkpoint: "Summary continuation duration",
    archived_elapsed: "Archived total duration",
    first_progress: "Submit to first active step",
    first_draft: "Submit to first readable draft",
    terminal: "Submit to visible terminal state",
    final_report: "Submit to readable report",
    ready: "Passed",
    blocked: "Blocked",
    partial: "Partial",
  },
};

const stageLabels = {
  "规划": "Planning",
  "搜索／召回": "Search and retrieval",
  "去重／筛选": "Deduplication and filtering",
  "全文获取": "Full-text acquisition",
  "抽取（含全文获取）": "Extraction (including full text)",
  "写作／审阅": "Writing and review",
  "adversarial_review:synthesis:r1": "Synthesis 1",
  "adversarial_review:reviewer:r1": "Review 1",
  "evaluation.initial": "First evaluation",
  "修订（含准备）": "Revisions (including preparation)",
  "evaluation.post_rewrite": "Re-evaluations",
};

const chineseStages = {
  "adversarial_review:synthesis:r1": "初稿写作 1",
  "adversarial_review:reviewer:r1": "审阅 1",
  "evaluation.initial": "首次 Evaluation",
  "evaluation.post_rewrite": "复评",
};

const byId = id => document.getElementById(id);
const select = byId("archive-run");
const title = byId("archive-title");
const message = byId("archive-message");
const content = byId("archive-content");
const executionSection = byId("section-execution");
const reportSection = byId("section-report");
const viewButtons = [...byId("archive-views").querySelectorAll("button")];
const drawer = byId("inspector-drawer");
const closeButton = byId("close-inspector");
const inspector = byId("inspector-content");
const localeButton = byId("archive-language");
const records = new Map();
const runIdPattern = /^[a-z0-9-]{1,120}$/;

function route() {
  return new URL(location.href);
}

const store = createWorkbenchStore({
  locale: route().searchParams.get("lang") === "en" ? "en" : "zh-CN",
});
let workbenchT = translator(store.get().locale);
const t = key => copy[store.get().locale][key] ?? workbenchT(key);
const workspace = createExecutionWorkspace({ store, t, onInspect: inspect });
let inspectorTrigger = null;
let currentSelection = null;
let activeMessage = "archiveLoading";
let displayedView = null;

function showMessage(key) {
  activeMessage = key;
  message.hidden = !key;
  if (!key) return;
  message.textContent = t(key);
  message.dataset.kind = key === "archiveLoading" ? "status" : "error";
}

function runName(run) {
  return [run.case_id, run.profile_id, `r${run.repetition}`].join(" · ");
}

function renderTitle(run = records.get(select.value)) {
  title.textContent = run ? runName(run) : t("publishedExecution");
  document.title = `${title.textContent} · LitAgent`;
}

function renderOptions() {
  const previous = select.value;
  const placeholder = new Option(t("archiveSelect"), "");
  placeholder.disabled = true;
  const options = [...records.values()].map(run =>
    new Option(`${runName(run)} · ${run.run_id.slice(-12)}`, run.run_id)
  );
  select.replaceChildren(placeholder, ...options);
  select.value = previous;
}

function makeMetric(target, label, value, status) {
  const item = document.createElement("div");
  const key = document.createElement("dt");
  const result = document.createElement("dd");
  key.textContent = label;
  result.textContent = value;
  if (status) item.dataset.status = status;
  item.append(key, result);
  target.append(item);
}

function seconds(ms) {
  if (!Number.isFinite(ms) || ms < 0) return "—";
  const formatted = (ms / 1000).toLocaleString(store.get().locale, {
    minimumFractionDigits: 1,
    maximumFractionDigits: 1,
  });
  return `${formatted} ${t("secondsShort")}`;
}

function score(metric, showClaims = false) {
  const value = metric?.score;
  if (!["passed", "failed"].includes(metric?.status)
      || !Number.isFinite(value) || value < 0 || value > 1) return "—";
  const percent = `${(100 * value).toFixed(2)}%`;
  if (!showClaims || !Number.isInteger(metric.supported_count)
      || !Number.isInteger(metric.factual_claim_count) || metric.factual_claim_count < 1) return percent;
  return `${percent} (${metric.supported_count}/${metric.factual_claim_count})`;
}

function tokens(usage = {}) {
  const count = usage.reconciled_billed_tokens
    ?? (usage.usage_uncertain
      ? usage.reported_tokens ?? usage.total_tokens
      : usage.billed_tokens ?? usage.total_tokens ?? usage.reported_tokens);
  if (!Number.isFinite(count) || count < 0) return "—";
  const number = count.toLocaleString(store.get().locale);
  if (usage.reconciled_billed_tokens != null) return `${number} · ${t("recordingReconciled")}`;
  return usage.usage_uncertain ? `${number} · ${t("recordingPartial")}` : number;
}

function renderMetrics(view) {
  const target = byId("archive-metrics");
  target.replaceChildren();
  const metrics = view.metrics || {};
  const delivery = view.delivery?.status;
  const deliveryName = ["ready", "blocked", "partial"].includes(delivery) ? t(delivery) : "—";
  makeMetric(target, t("strictDelivery"), deliveryName, delivery);
  makeMetric(target, "Faithfulness ≥ 0.95", score(metrics.faithfulness, true));
  makeMetric(target, t("citationAccuracyMetric"), score(metrics.citation_accuracy));
  makeMetric(target, t("consistencyMetric"), score(metrics.internal_consistency));
  makeMetric(target, t(metrics.duration_basis ?? "elapsed"), seconds(metrics.display_elapsed_ms));
  makeMetric(target, t("tokens"), tokens(metrics.usage));

  const timings = byId("archive-stage-times");
  timings.replaceChildren();
  if (Number.isFinite(metrics.elapsed_ms)) {
    makeMetric(timings, t("archived_elapsed"), seconds(metrics.elapsed_ms));
  }
  for (const name of ["first_progress", "first_draft", "terminal", "final_report"]) {
    const duration = metrics.browser_ms?.[name];
    if (Number.isFinite(duration)) makeMetric(timings, t(name), seconds(duration));
  }
  for (const name of Object.keys(stageLabels)) {
    const duration = metrics.stages_ms?.[name];
    if (duration == null) continue;
    const label = store.get().locale === "en" ? stageLabels[name] : chineseStages[name] ?? name;
    const values = Array.isArray(duration) ? duration : [duration];
    if (values.length) makeMetric(timings, label, values.map(seconds).join(" · "));
  }
  byId("archive-stages").hidden = timings.childElementCount === 0;
}

function linkReportReferences(documentBody) {
  const bibliography = [...documentBody.querySelectorAll(":scope > h2")].findLast(
    heading => /^(References|参考文献)$/i.test(heading.textContent.trim()),
  );
  const references = new Map();
  for (let entry = bibliography?.nextElementSibling; entry && !/^H[12]$/.test(entry.tagName); entry = entry.nextElementSibling) {
    const number = entry.tagName === "P" && /^\s*\[([1-9]\d*)\]/.exec(entry.textContent)?.[1];
    if (!number) continue;
    const leadingText = entry.firstChild;
    if (leadingText?.nodeType === Node.TEXT_NODE) {
      leadingText.data = leadingText.data.replace(/^(\s*\[[1-9]\d*\])(?!\s)/, "$1 ");
    }
    entry.id = `archive-ieee-reference-${number}`;
    entry.tabIndex = -1;
    references.set(number, entry.id);
  }
  for (const citation of documentBody.querySelectorAll('a[href^="#citation-"]')) {
    const parts = citation.textContent.split(/(\[[1-9]\d*\])/);
    citation.replaceWith(...parts.map((part, index) => {
      const destination = index % 2 ? references.get(part.slice(1, -1)) : null;
      if (!destination) return part;
      const link = document.createElement("a");
      link.href = `#${destination}`;
      link.textContent = part;
      return link;
    }));
  }
}

function renderReport(report) {
  const documentBody = byId("archive-report-document");
  const status = byId("archive-report-status");
  const markdown = byId("archive-report-markdown");
  const available = typeof report?.markdown === "string" && report.markdown.length > 0;
  const statusKey = !available ? "archiveReportMissing" : report.state === "partial" ? "archiveReportPartial" : null;
  status.hidden = !statusKey;
  status.textContent = statusKey ? t(statusKey) : "";
  documentBody.hidden = !available;
  documentBody.replaceChildren();
  documentBody.classList.remove("render-fallback");
  markdown.hidden = true;
  markdown.open = false;
  const original = markdown.querySelector("pre");
  original.textContent = "";
  if (!available) return;
  if (report.html) {
    renderReportDocument(documentBody, report, new Map(), t);
    if (documentBody.childElementCount && !documentBody.classList.contains("render-fallback")) {
      linkReportReferences(documentBody);
      // Citation IDs remain readable without implying they failed a ledger lookup.
      for (const citation of documentBody.querySelectorAll(".citation-unresolved")) {
        citation.replaceWith(citation.textContent);
      }
      original.textContent = report.markdown;
      markdown.hidden = false;
      return;
    }
  }
  documentBody.classList.remove("render-fallback");
  const source = document.createElement("pre");
  source.className = "archive-report-source";
  source.textContent = report.markdown;
  documentBody.replaceChildren(source);
}

function renderViewRoute() {
  const report = route().searchParams.get("view") === "report";
  executionSection.hidden = report;
  reportSection.hidden = !report;
  for (const button of viewButtons) {
    if (button.dataset.archiveView === (report ? "report" : "execution")) button.setAttribute("aria-current", "page");
    else button.removeAttribute("aria-current");
  }
  const view = store.get().view;
  if (view && !report) workspace.render(view);
}

function archivedStepIO(view, selection) {
  const entries = view.step_io;
  if (!entries) return null;
  const value = selection.value;
  // Role cards and presentation-only nodes can have IDs different from the recorded span.
  let keys = [selection.id];
  if (selection.kind === "node") {
    if (value?.role) keys = [value.execution?.id];
    else if (value?.task?.observation) keys = [value.task.observation.id];
    else keys = [value?.execution?.id, `worker:${selection.id}`, selection.id];
  }
  const key = keys.find(id => typeof id === "string" && Object.hasOwn(entries, id));
  return key ? entries[key] : null;
}

function appendIONote(target, key) {
  const note = document.createElement("p");
  note.className = "archive-io-note";
  note.textContent = t(key);
  target.append(note);
}

function renderArchivedStepIO(entry) {
  const sections = [...inspector.querySelectorAll(":scope > .execution-business")];
  for (const section of sections) {
    const direction = section.dataset.direction;
    const heading = section.querySelector("h3");
    const value = entry?.[direction];
    const omitted = entry?.omitted?.includes(direction);
    section.replaceChildren(heading);
    if (value == null) {
      const missing = direction === "input" ? "archiveInputMissing" : "archiveOutputMissing";
      appendIONote(section, omitted ? "archiveIOWithheld" : missing);
      continue;
    }
    if (entry.status === "reused") appendIONote(section, "archiveIOReused");
    if (entry.status === "planned") appendIONote(section, "archiveIOPlanned");
    if (omitted) appendIONote(section, "archiveIOOmitted");
    const source = document.createElement("pre");
    source.className = "archive-io-value";
    source.tabIndex = 0;
    source.textContent = typeof value === "string" && value.length ? value : JSON.stringify(value, null, 2);
    section.append(source);
  }
  const rawLabels = [t("ioRawInput"), t("ioRawOutput")];
  for (const summary of inspector.querySelectorAll(":scope > .inspector-json > summary")) {
    if (rawLabels.includes(summary.textContent)) summary.parentElement.remove();
  }
}

function inspect(selection, trigger, moveFocus = true) {
  const view = store.get().view;
  if (!view) return;
  inspector.replaceChildren();
  if (!renderExecutionInspector(inspector, selection, t, { view, onSelect: inspect })) return;
  renderArchivedStepIO(archivedStepIO(view, selection));
  currentSelection = selection;
  if (drawer.dataset.open !== "true") inspectorTrigger = trigger;
  const value = selection.value || {};
  byId("inspector-title").textContent = value.labelKey ? t(value.labelKey)
    : value.name || value.event || selection.id;
  drawer.inert = false;
  drawer.dataset.open = "true";
  drawer.setAttribute("aria-hidden", "false");
  if (moveFocus) closeButton.focus();
}

function closeInspector(restoreFocus = true) {
  if (!currentSelection) return;
  if (restoreFocus) {
    const fallback = byId("execution-mode").querySelector('[aria-pressed="true"]');
    (inspectorTrigger?.isConnected ? inspectorTrigger : fallback)?.focus();
  }
  currentSelection = null;
  inspectorTrigger = null;
  drawer.inert = true;
  drawer.dataset.open = "false";
  drawer.setAttribute("aria-hidden", "true");
}

async function json(path) {
  const response = await fetch(new URL(path, document.baseURI));
  if (!response.ok) throw new Error("Archive unavailable");
  return response.json();
}

async function loadRun(id) {
  closeInspector(false);
  executionSection.classList.remove("execution-fullscreen");
  store.update({ view: null });
  const epoch = store.beginRequest();
  const record = records.get(id);
  select.value = record ? id : "";
  renderTitle(record);
  content.hidden = true;
  if (!record) {
    store.update({ loading: false });
    showMessage("archiveUnknown");
    return;
  }
  showMessage("archiveLoading");
  try {
    const view = await json(`./runs/${id}.json`);
    if (!store.isCurrent(epoch)) return;
    if (view.identity?.runId !== id || !view.graph?.tasks || !Array.isArray(view.events)) {
      throw new Error("Archive identity mismatch");
    }
    content.hidden = false;
    showMessage(null);
    store.update({ view, loading: false });
  } catch {
    if (!store.isCurrent(epoch)) return;
    content.hidden = true;
    store.update({ view: null, loading: false });
    showMessage("archiveUnavailable");
  }
}

function changeLocale() {
  const locale = store.get().locale === "en" ? "zh-CN" : "en";
  const url = route();
  if (locale === "en") url.searchParams.set("lang", "en");
  else url.searchParams.delete("lang");
  history.replaceState(null, "", url);
  setLocale(locale);
  if (currentSelection) inspect(currentSelection, inspectorTrigger, false);
}

function setLocale(locale) {
  document.documentElement.lang = locale;
  workbenchT = translator(locale);
  displayedView = null;
  store.update({ locale });
  applyTranslations(document, t);
  localeButton.textContent = locale === "en" ? "中文" : "EN";
  localeButton.setAttribute("aria-label", locale === "en" ? "切换到中文" : "Switch to English");
  renderOptions();
  renderTitle();
  showMessage(activeMessage);
}

async function boot() {
  try {
    const index = await json("./runs/index.json");
    for (const item of index.runs || []) {
      if (typeof item.run_id !== "string" || !runIdPattern.test(item.run_id) || records.has(item.run_id)) {
        throw new Error("Invalid archive index");
      }
      records.set(item.run_id, item);
    }
    if (!records.size) throw new Error("Empty archive index");
    renderOptions();
    select.disabled = false;
    const url = route();
    if (!url.searchParams.has("run")) {
      url.searchParams.set("run", records.keys().next().value);
      history.replaceState(null, "", url);
    }
    await loadRun(url.searchParams.get("run"));
  } catch {
    showMessage("archiveUnavailable");
  }
}

workspace.bind();
store.subscribe(({ view }) => {
  if (!view) return;
  if (view !== displayedView) {
    renderMetrics(view);
    renderReport(view.report);
    displayedView = view;
  }
  if (!executionSection.hidden) workspace.render(view);
});
for (const button of viewButtons) {
  button.addEventListener("click", () => {
    if (button.hasAttribute("aria-current")) return;
    const url = route();
    if (button.dataset.archiveView === "report") url.searchParams.set("view", "report");
    else {
      url.searchParams.delete("view");
      url.hash = "";
    }
    history.pushState(null, "", url);
    closeInspector(false);
    renderViewRoute();
  });
}
select.addEventListener("change", () => {
  const url = route();
  url.searchParams.set("run", select.value);
  url.hash = "";
  history.pushState(null, "", url);
  void loadRun(select.value);
});
localeButton.addEventListener("click", changeLocale);
closeButton.addEventListener("click", () => closeInspector());
document.addEventListener("keydown", event => {
  if (event.key === "Escape") closeInspector();
});
addEventListener("popstate", () => {
  const url = route();
  const locale = url.searchParams.get("lang") === "en" ? "en" : "zh-CN";
  closeInspector(false);
  renderViewRoute();
  if (locale !== store.get().locale) setLocale(locale);
  if (url.searchParams.get("run") !== select.value) void loadRun(url.searchParams.get("run"));
});
renderViewRoute();
setLocale(store.get().locale);
void boot();
