import { renderExecutionInspector } from "./workbench/execution-inspector.js";
import { createExecutionWorkspace } from "./workbench/execution-workspace.js";
import { applyTranslations, translator } from "./workbench/i18n.js";
import { createWorkbenchStore } from "./workbench/state.js";

const copy = {
  "zh-CN": {
    publishedExecution: "执行归档",
    archiveRun: "运行",
    archiveBenchmark: "基准总表",
    archiveMetrics: "运行指标",
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

function inspect(selection, trigger, moveFocus = true) {
  const view = store.get().view;
  if (!view) return;
  inspector.replaceChildren();
  if (!renderExecutionInspector(inspector, selection, t, { view, onSelect: inspect })) return;
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
  byId("section-execution").classList.remove("execution-fullscreen");
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
    displayedView = view;
  }
  workspace.render(view);
});
select.addEventListener("change", () => {
  const url = route();
  url.searchParams.set("run", select.value);
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
  setLocale(url.searchParams.get("lang") === "en" ? "en" : "zh-CN");
  void loadRun(url.searchParams.get("run"));
});
setLocale(store.get().locale);
void boot();
