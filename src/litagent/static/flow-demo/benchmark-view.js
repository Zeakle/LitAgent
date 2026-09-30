const el = (tag, text = "", className = "") => {
  const node = document.createElement(tag);
  node.textContent = text;
  if (className) node.className = className;
  return node;
};

const value = (item, digits = 2) => item == null ? "—" : typeof item === "number" ? Number(item).toFixed(digits) : String(item);
const percent = (item) => item == null ? "—" : `${(Number(item) * 100).toFixed(1)}%`;

function metricTable(headers, rows) {
  const scroll = el("div", "", "benchmark-table-wrap");
  const table = document.createElement("table");
  const head = document.createElement("thead");
  const headRow = document.createElement("tr");
  for (const header of headers) headRow.append(el("th", header));
  head.append(headRow);
  const body = document.createElement("tbody");
  for (const cells of rows) {
    const row = document.createElement("tr");
    for (const cell of cells) row.append(el("td", cell));
    body.append(row);
  }
  table.append(head, body);
  scroll.append(table);
  return scroll;
}

function safeExternalLink(label, href) {
  if (!href) return null;
  try {
    const url = new URL(href);
    if (!["http:", "https:"].includes(url.protocol)) return null;
    const link = el("a", label);
    link.href = url.href;
    link.target = "_blank";
    link.rel = "noopener noreferrer";
    return link;
  } catch {
    return null;
  }
}

function renderCitationAudits(target, audits, t) {
  target.append(el("h5", t("citationAuditExperiment")));
  if (!audits.length) {
    target.append(el("p", t("noCitationAudits"), "benchmark-empty"));
    return;
  }
  for (const audit of audits) {
    const block = el("section", "", "benchmark-citation-audit");
    const heading = el("div", "", "benchmark-citation-audit-heading");
    heading.append(
      el("strong", `${audit.actual_model || audit.requested_model || "Jev"} · ${audit.status}`),
      el(
        "span",
        audit.association_status === "current" ? t("currentArtifact") : t("staleDiagnostic"),
        `benchmark-association benchmark-association-${audit.association_status || "unknown"}`,
      ),
    );
    block.append(heading);
    const summary = audit.summary || {};
    const usage = audit.usage || {};
    block.append(metricTable(
      [t("auditClaims"), t("protocolCompletion"), t("uncertainRate"), t("inputTokens"), t("usageComplete")],
      [[
        value(summary.valid_questions, 0),
        percent(summary.protocol_completion_rate),
        percent(summary.uncertain_rate),
        value(usage.input_tokens, 0),
        value(usage.usage_complete),
      ]],
    ));
    const traceLink = safeExternalLink(t("auditTrace"), audit.trace_url);
    if (traceLink) block.append(traceLink);
    const decisions = audit.decisions || [];
    if (decisions.length) {
      const details = document.createElement("details");
      details.append(el("summary", `${t("auditDecisions")} (${decisions.length})`));
      details.append(metricTable(
        [t("claim"), t("auditTarget"), t("auditDecision"), t("confidence"), t("status")],
        decisions.map((decision) => [
          decision.claim_id || "—",
          decision.evidence_id || t("citationSet"),
          decision.choice || "—",
          value(decision.confidence),
          decision.uncertain ? t("uncertain") : decision.status || "—",
        ]),
      ));
      block.append(details);
    }
    for (const input of audit.inputs || []) {
      const details = document.createElement("details");
      details.className = "benchmark-audit-source";
      details.append(el("summary", `${input.claim_id} · ${input.status}`));
      details.append(
        el("p", input.claim_text || "—"),
        el("small", `${t("sourcePosition")}: ${input.source_unit_id} [${input.source_start}, ${input.source_end})`),
        el("pre", input.source_quote || "—", "benchmark-raw-data"),
      );
      const evidenceRows = (input.evidence || []).map((evidence) => [
        evidence.evidence_id || "—",
        evidence.paper_id || "—",
        evidence.input_scope || "—",
        evidence.text || "—",
      ]);
      if (evidenceRows.length) details.append(metricTable(["Evidence ID", "Paper ID", t("evidenceScope"), t("output")], evidenceRows));
      block.append(details);
    }
    block.append(el("small", `${t("reasonCodes")}: ${(audit.reason_codes || []).join(", ") || "—"}`));
    target.append(block);
  }
}

export function renderBenchmarkList(target, results, selectedId, onSelect, t) {
  target.replaceChildren();
  if (!results.length) return target.append(el("p", t("noBenchmark"), "benchmark-empty"));
  for (const result of results) {
    const button = el("button", "", "benchmark-result-button");
    button.type = "button";
    button.setAttribute("aria-current", result.benchmark_id === selectedId ? "true" : "false");
    button.append(
      el("strong", result.benchmark_id),
      el("small", `${result.stage} · ${result.observation_count} ${t("observations")} · ${result.status}`),
    );
    button.addEventListener("click", () => onSelect(result.benchmark_id));
    target.append(button);
  }
}

export function renderBenchmarkDetail(target, result, onObservation, t) {
  target.replaceChildren();
  if (!result) return target.append(el("p", t("selectBenchmark"), "benchmark-empty"));
  target.append(el("h3", result.run_id));
  const badges = el("div", "", "benchmark-flags");
  for (const label of [result.stage, result.status, result.judgment_status, result.self_judged ? "self_judged" : "independent_judge", result.formal_eligible ? "formal_eligible" : "pilot_or_ineligible"]) badges.append(el("span", label));
  target.append(badges);
  const corrected = result.corrected_diagnostic;
  if (corrected) target.append(el("p", `${t("correctedDiagnostic")}: ${corrected.metrics_version} · ${corrected.source_sha256}`, "benchmark-warning"));
  else if (!result.metrics_version) target.append(el("p", t("legacyMetricsWarning"), "benchmark-warning"));
  target.append(el("p", `${t("benchmarkSample")}: ${result.observations.length} ${t("observations")} · ${result.dataset_id} ${result.dataset_version}`));
  target.append(el("h4", t("profileSummary")));
  target.append(metricTable(
    ["Profile", "N", "Coverage", "Citation recall", "Citation precision", "Judge faithfulness", "Runtime delivery", "Product tokens", "Judge tokens", "p95 end-to-end ms", "p95 controlled ms"],
    (corrected?.profile_summaries || result.profile_summaries).map((item) => [
      item.profile_id, String(item.observation_count), percent(item.coverage_evidence), percent(item.citation_recall),
      percent(item.citation_precision), percent(item.faithfulness), percent(item.delivery_accuracy),
      value(item.product_total_tokens, 0), value(item.judge_total_tokens, 0), value(item.latency_p95_ms, 0),
      value(item.controlled_downstream_p95_ms, 0),
    ]),
  ));
  target.append(el("h4", t("pairedComparisons")));
  const comparisons = corrected?.profile_comparisons || (result.metrics_version ? result.profile_comparisons || [] : []);
  if (!comparisons.length) target.append(el("p", t("noComparisons"), "benchmark-empty"));
  for (const pair of comparisons) {
    const block = el("div", "", "benchmark-comparison");
    block.append(el("strong", `${pair.left_profile_id} → ${pair.right_profile_id}: ${pair.comparable ? t("comparable") : t("notComparable")}`));
    block.append(el("p", `${t("sharedCohorts")}: ${pair.shared_cohort_count} · ${(pair.reason_codes || []).join(", ") || "—"}`));
    const names = Object.keys(pair.metric_deltas || {});
    block.append(metricTable(["Metric", "Delta", "Paired N", "Exclusions"], names.map((name) => [name, value(pair.metric_deltas[name]), value(pair.metric_paired_counts?.[name], 0), (pair.metric_exclusion_reasons?.[name] || []).join("; ") || "—"])));
    target.append(block);
  }
  target.append(el("h4", t("benchmarkRuns")));
  const list = el("div", "", "benchmark-observations");
  for (const item of result.observations) {
    const button = el("button", `${item.case_id} · ${item.profile_id} · r${item.repetition} · ${item.status}`, "benchmark-observation-button");
    button.type = "button";
    button.addEventListener("click", () => onObservation(item.run_id));
    list.append(button);
  }
  target.append(list);
  const pane = el("div", "", "benchmark-observation-detail");
  pane.id = "benchmark-observation-detail";
  target.append(pane);
}

export function renderBenchmarkObservation(target, payload, t) {
  target.replaceChildren();
  const { observation, artifact } = payload;
  target.append(el("h4", `${observation.case_id} · ${observation.profile_id}`));
  const summary = el("dl", "", "benchmark-observation-metrics");
  const metrics = payload.corrected_observation?.metrics || observation.metrics || {};
  if (payload.corrected_observation) target.append(el("p", t("correctedDiagnostic"), "benchmark-warning"));
  const runtimeFaithfulness = metrics.runtime_evaluation?.faithfulness;
  const fields = [
    ["Runtime faithfulness", typeof runtimeFaithfulness === "object" ? runtimeFaithfulness?.score : runtimeFaithfulness],
    ["Judge faithfulness", observation.judge?.faithfulness],
    ["Delivery", metrics.delivery_status],
    ["Citation recall", metrics.citations?.recall],
    ["Product tokens", metrics.cost?.total_tokens],
    ["Product usage complete", metrics.cost?.usage_complete],
    ["Judge tokens", observation.judge_usage?.total_tokens],
    ["Controlled downstream ms", metrics.controlled_downstream_ms],
  ];
  for (const [label, datum] of fields) { summary.append(el("dt", label), el("dd", value(datum))); }
  target.append(summary);
  const links = el("div", "", "benchmark-trace-links");
  for (const [label, href] of [["Survey trace", observation.survey_trace_url], ["Judge trace", observation.judge_trace_url]]) {
    const link = safeExternalLink(label, href);
    if (link) links.append(link);
  }
  target.append(links);
  const reportLink = el("a", t("openReport"));
  reportLink.href = `/flow-demo?run=${encodeURIComponent(`archive:${observation.run_id}`)}&view=report`;
  target.append(reportLink);
  const report = artifact.report?.survey || artifact.report?.report?.survey || "";
  target.append(el("h5", t("report")), el("pre", report || t("noReport"), "benchmark-raw-report"));
  target.append(el("h5", t("evidenceFlow")));
  const flow = metrics.paper_flow || metrics.coverage || {};
  target.append(metricTable(["Stage", "Coverage", "Missing papers"],
    ["recall", "dedup", "relevance_gate", "evidence", "synthesis_context", "final_citation"].map((stage) => [stage, percent(flow[stage]), (flow.missing_by_stage?.[stage] || []).join(", ") || "—"])));
  target.append(el("h5", t("negativeVerdicts")));
  const negatives = observation.judge?.claim_assessments?.filter((claim) => !claim.supported) || [];
  if (!negatives.length) target.append(el("p", "—"));
  for (const claim of negatives) {
    const item = el("div", "", "benchmark-negative-claim");
    item.append(el("p", claim.claim_text), el("small", `${(claim.evidence_ids || []).join(", ") || "No cited evidence"} · ${claim.explanation || "—"}`));
    target.append(item);
  }
  renderCitationAudits(target, payload.citation_audits || [], t);
}
