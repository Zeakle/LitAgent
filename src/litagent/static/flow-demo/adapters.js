const TERMINAL = new Set(["completed", "cancelled", "rejected", "failed", "interrupted"]);
const BUDGET_STOP_REASONS = new Set(["strict_usage_unknown", "strict_worker_failed", "planned_usage_bound_violation", "planned_ledger_persistence_failed"]);
const BUDGET_FIELDS = ["plan_identity", "known_tokens", "requests", "usage_uncertain", "stop_reason", "unknown_hold"];
const BUDGET_KEYS = new Set(["contract", "scope", "run_id", "availability", "reason", ...BUDGET_FIELDS]);

function budgetState(raw, runId, plan) {
  const unavailable = availability => ({
    contract: "run-budget-state-v1", scope: "run",
    run_id: typeof runId === "string" ? runId : "",
    availability, reason: availability === "missing" ? "not_recorded" : "invalid_record",
    ...Object.fromEntries(BUDGET_FIELDS.map(key => [key, null])),
  });
  if (raw == null) return unavailable("missing");
  if (!raw || typeof raw !== "object" || Array.isArray(raw)
      || Object.keys(raw).some(key => !BUDGET_KEYS.has(key))
      || [...BUDGET_KEYS].some(key => !Object.hasOwn(raw, key))) return unavailable("invalid");
  if (raw.contract !== "run-budget-state-v1" || raw.scope !== "run"
      || typeof runId !== "string" || !runId || raw.run_id !== runId) return unavailable("invalid");
  if (["missing", "invalid"].includes(raw.availability)) {
    if (raw.reason !== (raw.availability === "missing" ? "not_recorded" : "invalid_record")
        || !BUDGET_FIELDS.every(key => raw[key] === null)) return unavailable("invalid");
    return unavailable(raw.availability);
  }
  const integer = value => Number.isSafeInteger(value) && value >= 0;
  if (raw.availability !== "available" || raw.reason !== null
      || typeof raw.plan_identity !== "string" || !/^[a-f0-9]{64}$/.test(raw.plan_identity)
      || ![raw.known_tokens, raw.requests, raw.unknown_hold].every(integer)
      || typeof raw.usage_uncertain !== "boolean"
      || !(raw.stop_reason === null || BUDGET_STOP_REASONS.has(raw.stop_reason))
      || (raw.unknown_hold > 0 && !raw.usage_uncertain)) return unavailable("invalid");
  if (["planned-budget-v2", "planned-budget-v3", "planned-budget-v4", "planned-budget-v5", "planned-budget-v6", "planned-budget-v7"].includes(plan?.contract)
      && Object.hasOwn(plan, "identity") && plan.identity !== raw.plan_identity) return unavailable("invalid");
  // Copy only the public display contract; never retain archive extras or ledger entries.
  return {
    contract: raw.contract, scope: raw.scope, run_id: runId,
    availability: "available", reason: null,
    ...Object.fromEntries(BUDGET_FIELDS.map(key => [key, raw[key]])),
  };
}

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
    displayWarnings: raw?.display_warnings || [],
    identity: { ...identity, status: raw?.summary?.status || identity.status },
    summary: raw?.summary || {},
    report: raw?.report || null,
    renderedReport: raw?.rendered_report || { html: "", headings: [] },
    reportSha: raw?.rendered_report?.source_sha256 || null,
    graph: raw?.dag || { tasks: {}, dependencies: {}, nodes: {} },
    events: Array.isArray(raw?.timeline) ? raw.timeline : [],
    evidence: Array.isArray(raw?.evidence) ? raw.evidence : [],
    runtimeEvaluation: raw?.evaluation || {},
    pipelineVerification: raw?.pipeline_verification || null,
    budgetPlan: raw?.budget_plan || null,
    budgetState: budgetState(raw?.budget_state, identity.runId, raw?.budget_plan),
    targetedRepairState: raw?.targeted_repair_state || null,
    benchmarkJudge: raw?.benchmark_judge || null,
    quality: raw?.quality || null,
    delivery: raw?.delivery || null,
    review: null,
    versions: [],
    trace: raw?.trace || {},
    capabilities: {
      canCancel: false,
      canReview: false,
      canExport: (identity.source === "archive" || raw?.presentation === true) && Boolean(raw?.report?.survey),
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
    reportSha: raw?.rendered_report?.source_sha256 || null,
    graph: raw?.graph || { tasks: {}, dependencies: {}, nodes: {} },
    events: Array.isArray(raw?.timeline) ? raw.timeline : [],
    evidence: Array.isArray(raw?.evidence) ? raw.evidence : [],
    runtimeEvaluation: raw?.evaluation || {},
    pipelineVerification: raw?.pipeline_verification || null,
    budgetPlan: raw?.budget_plan || null,
    budgetState: budgetState(raw?.budget_state, identity.runId, raw?.budget_plan),
    targetedRepairState: raw?.targeted_repair_state || null,
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
