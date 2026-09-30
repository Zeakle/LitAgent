const TERMINAL = new Set(["completed", "cancelled", "rejected", "failed", "interrupted"]);

function text(value, fallback = "") {
  return typeof value === "string" ? value : fallback;
}

function timestamp(value) {
  const parsed = Date.parse(value || "");
  return Number.isFinite(parsed) ? parsed : 0;
}

function normalizeInteractive(item) {
  if (!item || typeof item !== "object" || !text(item.run_id)) return null;
  return {
    key: `interactive:${item.run_id}`,
    runId: item.run_id,
    source: "interactive",
    query: text(item.query),
    status: text(item.status, "unknown"),
    startedAt: item.created_at || null,
    completedAt: TERMINAL.has(item.status) ? item.updated_at || null : null,
    updatedAt: item.updated_at || null,
    reportLanguage: item.options?.report_language || null,
    qualityStatus: item.quality_status || null,
    deliveryStatus: item.delivery_status || null,
    hasPendingReview: item.status === "awaiting_human",
  };
}

function normalizeArchive(item) {
  if (!item || typeof item !== "object" || !text(item.run_id)) return null;
  const source = item.source === "sample" ? "sample" : "archive";
  return {
    key: `${source}:${item.run_id}`,
    runId: item.run_id,
    source,
    query: text(item.query),
    status: text(item.status, "unknown"),
    startedAt: item.started_at || null,
    completedAt: item.completed_at || null,
    updatedAt: item.completed_at || item.started_at || null,
    reportLanguage: item.report_language || null,
    qualityStatus: item.quality?.status || null,
    deliveryStatus: item.delivery?.status || null,
    hasPendingReview: false,
  };
}

export function normalizeRunList(interactiveRuns = [], archiveRuns = []) {
  const normalized = [
    ...interactiveRuns.map(normalizeInteractive),
    ...archiveRuns.map(normalizeArchive),
  ].filter(Boolean);
  return normalized.sort((left, right) => {
    const delta = timestamp(right.completedAt || right.updatedAt || right.startedAt)
      - timestamp(left.completedAt || left.updatedAt || left.startedAt);
    return delta || left.key.localeCompare(right.key);
  });
}

export function chooseDefaultSection(view) {
  const status = view?.identity?.status || view?.summary?.status;
  if (status === "awaiting_human" && view?.review) return "review";
  if (["queued", "running", "cancelling"].includes(status)) return "execution";
  if (view?.report?.survey) return "report";
  return "execution";
}

export function adaptArchiveView(raw, identity) {
  return {
    identity: { ...identity, status: raw?.summary?.status || identity.status },
    summary: raw?.summary || {},
    report: raw?.report || null,
    renderedReport: raw?.rendered_report || { html: "", headings: [] },
    graph: raw?.dag || { tasks: {}, dependencies: {}, nodes: {} },
    events: Array.isArray(raw?.timeline) ? raw.timeline : [],
    evidence: Array.isArray(raw?.evidence) ? raw.evidence : [],
    runtimeEvaluation: raw?.evaluation || {},
    benchmarkJudge: raw?.benchmark_judge || null,
    quality: raw?.quality || null,
    delivery: raw?.delivery || null,
    review: null,
    versions: [],
    trace: raw?.trace || {},
    capabilities: {
      canCancel: false,
      canReview: false,
      canExport: false,
      canDownloadArtifact: identity.source === "archive",
      hasTrace: Boolean(raw?.trace?.trace_url),
    },
  };
}

export function adaptInteractiveView(raw, identity) {
  const capabilities = raw?.capabilities || {};
  return {
    identity: { ...identity, status: raw?.summary?.status || identity.status },
    summary: raw?.summary || {},
    report: raw?.report || null,
    renderedReport: raw?.rendered_report || { html: "", headings: [] },
    graph: raw?.graph || { tasks: {}, dependencies: {}, nodes: {} },
    events: Array.isArray(raw?.timeline) ? raw.timeline : [],
    evidence: Array.isArray(raw?.evidence) ? raw.evidence : [],
    runtimeEvaluation: raw?.evaluation || {},
    benchmarkJudge: raw?.benchmark_judge || null,
    quality: raw?.quality || null,
    delivery: raw?.delivery || null,
    review: raw?.review || null,
    versions: Array.isArray(raw?.versions) ? raw.versions : [],
    trace: raw?.trace || {},
    capabilities: {
      canCancel: Boolean(capabilities.can_cancel),
      canReview: Boolean(capabilities.can_review),
      canExport: Boolean(capabilities.can_export),
      canDownloadArtifact: Boolean(capabilities.can_download_artifact),
      hasTrace: Boolean(capabilities.has_trace),
    },
  };
}
