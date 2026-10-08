/* Version-bound diagnostics never alter the original report or gate. */
function scoreText(result) {
  if (result?.skipped || result?.score == null) return "—";
  const numeric = Number(result.score);
  return Number.isFinite(numeric) ? numeric.toFixed(3) : "—";
}
const LABELS = {
  passed: ["通过", "Passed"], failed: ["未通过", "Failed"], unverified: ["未核验", "Unverified"],
  citation_accuracy: ["引用 ID 有效性", "Citation ID validity"], faithfulness: ["科学断言支持率", "Scientific assertion support"], internal_consistency: ["内部一致性", "Internal consistency"],
  idHelp: ["引用存在不代表论文支持该断言。", "A valid citation ID does not establish semantic support."],
  historical: ["历史评估口径；不能与新版直接比较。", "Historical evaluation scope; not directly comparable with v2."],
  previousSupport: ["前版支持原文口径：由模型抄写，不能与新片段选择口径直接比较。", "Previous quoted-support scope; not directly comparable with host-selected passages."],
  historicalV2: ["历史 v2：未启用支持原文契约，不能与新结果直接比较。", "Historical v2: support-span contract was not enabled; not directly comparable with new results."],
  current: ["新版评估 · 按断言实际引用核验", "Evaluation v2 · actual citations per assertion"],
  threshold: ["阈值", "Threshold"], unavailable: ["未记录／不可用", "Not recorded / unavailable"],
  supported: ["支持", "Supported"], unsupported: ["未获支持", "Unsupported"], pending: ["未核验", "Unverified"],
  counts: ["科学断言数量", "Scientific assertion counts"], coverage: ["提取覆盖（处理单元／全部单元）", "Extraction coverage (processed / total units)"],
  checks: ["结构、方法与语言检查", "Structure, methodology and language checks"], language_checks: ["正文语言", "Report prose language"], support: ["支持依据契约", "Support basis contract"], basis: ["实际引用中的支持原文", "Support text from actual citations"], structure_checks: ["模板结构", "Template structure"], method_checks: ["方法披露", "Method disclosures"],
  not_applicable: ["不适用", "Not applicable"], issues: ["问题定位", "Located diagnostics"], noIssues: ["没有记录可定位问题；不代表额外核验通过。", "No located issues recorded; this does not establish additional verification."],
  distribution: ["问题分布（章节／类型）", "Issue distribution (section / type)"], source: ["定位原文及证据", "Locate source and evidence"], unknownSection: ["位置未记录", "Location not recorded"],
  audits: ["独立复核 · 不参与交付门禁", "Independent review · outside the delivery gate"], auditNone: ["此版本暂无核验结果。", "No audits for this version."], attempts: ["核验尝试", "Audit attempt"],
  noTopics: ["未提供人工主题标准", "No human topic standard provided"], confidence: ["Jev 置信度不是正确率；一致率仅适用于相同且可比较的输入。", "Jev confidence is not accuracy; agreement applies only to comparable identical inputs."],
  completion: ["完成率", "Completion"], agreement: ["判断一致率", "Decision agreement"], localErrors: ["本地错误", "Local errors"], uncertain: ["低置信度", "Low confidence"],
  scope: ["核验范围（完成／计划）", "Audit scope (completed / planned)"], questions: ["问题", "Questions"], claims: ["断言", "Claims"], insufficient: ["证据不足", "Insufficient evidence"], contradicted: ["矛盾", "Contradicted"],
  queued: ["排队中", "Queued"], running: ["核验中", "Running"], completed: ["已完成", "Completed"], partial: ["部分完成", "Partial"], cancelled: ["已取消", "Cancelled"], interrupted: ["已中断", "Interrupted"], blocked: ["阻塞", "Blocked"], not_requested: ["未请求", "Not requested"], skipped: ["跳过", "Skipped"], unverifiable: ["无法核验", "Unverifiable"], verified: ["记录支持", "Verified against records"],
  report_language_mismatch: ["正文语言不符", "Report language mismatch"],
  support_passage_invalid: ["支持片段不存在或重复", "Support passage is missing or duplicated"],
  support_basis_ownership_invalid: ["支持片段超出实际引用范围", "Support passage is outside actual citations"],
  support_aspect_coverage_invalid: ["断言的支持检查不完整", "Assertion support checks are incomplete"],
  support_aspect_unverified: ["断言的部分内容未获核验", "An assertion aspect remains unverified"],
  support_relation_unverified: ["证据未确认该科学关系或条件", "Evidence does not verify this relation or condition"],
  support_entity_unverified: ["证据中的实体对应尚未确认", "Evidence entity correspondence is unverified"],
  support_basis_quote_invalid: ["支持原文与所选来源字段不匹配", "Support text does not match the selected source field"],
  insufficient_support: ["证据不足", "Insufficient support"], attribution_mismatch: ["引用归属错误", "Attribution mismatch"], uncited_claim: ["缺少引用", "Uncited claim"], unknown_evidence_ref: ["证据缺失", "Unknown evidence"], record_mismatch: ["运行记录不一致", "Record mismatch"],
  gate: ["内容质量／交付状态", "Content quality / delivery"], reasons: ["阻断或降级原因", "Blocking or degradation reasons"], noReasons: ["未记录降级原因。", "No degradation reason recorded."], diagnosticOnly: ["结果供复核；不修改评分或人工批准。", "Review evidence only; preserves scores and approval."],
};
function numeric(value) { return typeof value === "number" && Number.isFinite(value) ? value : null; }
function display(value) { return value == null ? "—" : String(value); }
function rate(value) { return numeric(value) == null ? "—" : `${(value * 100).toFixed(1)}%`; }
function node(parent, tag, content, className) {
  const element = document.createElement(tag); if (content != null) element.textContent = content;
  if (className) element.className = className; parent.append(element); return element;
}
function meter(parent, value, label, threshold = null) {
  if (numeric(value) == null) return;
  const wrapper = node(parent, "span", null, "quality-meter");
  const bar = node(wrapper, "progress"); bar.max = 1; bar.value = Math.max(0, Math.min(1, value));
  bar.setAttribute?.("aria-label", label); bar.setAttribute?.("aria-valuetext", String(value));
  if (numeric(threshold) != null) {
    const marker = node(wrapper, "span", null, "quality-threshold");
    marker.setAttribute?.("style", `left:${Math.max(0, Math.min(1, threshold)) * 100}%`);
    marker.setAttribute?.("aria-hidden", "true");
  }
}
export function renderQuality(container, evaluation, quality, delivery, onSelect, t = key => key, options = {}) {
  const L = key => LABELS[key]?.[options.locale === "en" ? 1 : 0] || key;
  const details = evaluation?.faithfulness?.details || {};
  const v2 = details.evaluation_version === "survey-evaluation-v2";
  container.replaceChildren();
  const metrics = node(container, "div", null, "metric-grid");
  for (const [name, result] of Object.entries(evaluation || {})) {
    const button = node(metrics, "button", null, "metric-row"); button.type = "button";
    button.dataset.status = result?.skipped ? "unverified" : result?.passed === true ? "passed" : result?.passed === false ? "failed" : "unverified";
    node(button, "span", L(name)); node(button, "strong", scoreText(result));
    node(button, "small", result?.skipped ? t("metricSkipped") : L(button.dataset.status));
    const threshold = numeric(result?.details?.threshold);
    node(button, "small", `${L("threshold")}: ${threshold == null ? L("unavailable") : threshold}`);
    if (!result?.skipped) meter(button, numeric(result?.score), L(name), threshold);
    if (name === "citation_accuracy") node(button, "small", L("idHelp"), "metric-explanation");
    button.addEventListener("click", () => onSelect({ kind: "metric", id: name, value: result }, button));
  }
  const gate = node(container, "section", null, "quality-gate");
  node(gate, "h3", `${quality?.status || "unverified"} / ${delivery?.status || "pending"}`);
  node(gate, "p", `${L("gate")} · ${L("reasons")}: ${(delivery?.reason_codes || []).join(", ") || L("noReasons")}`);
  node(gate, "p", v2 ? `${L(details.support_contract === "host-support-passages-v2" ? "current" : details.support_contract === "cited-support-spans-v1" ? "previousSupport" : "historicalV2")} · ${details.evaluation_version}` : L("historical"));
  if (options.revision) node(gate, "p", `r${options.revision}${options.sha ? ` · SHA ${options.sha}` : ""}`, "audit-binding");
  if (v2) node(gate,"p",`${L("support")}: ${details.support_contract || L("unavailable")}`);
  const counts = node(container, "section", null, "quality-section"); node(counts, "h3", L("counts"));
  const grid = node(counts, "div", null, "quality-count-grid");
  for (const [key,field] of [["supported","supported_count"],["unsupported","unsupported_count"],["pending","unverified_count"]]) {
    const box = node(grid, "div", null, `quality-count ${key}`); node(box, "strong", display(v2 ? numeric(details[field]) : null)); node(box, "span", L(key));
    if (v2 && numeric(details.factual_claim_count) > 0 && numeric(details[field]) != null) meter(box, details[field]/details.factual_claim_count, L(key));
  }
  node(counts, "p", `${L("coverage")}: ${v2 ? `${display(details.coverage?.processed_units)} / ${display(details.coverage?.source_units)}` : L("unavailable")}`);
  const checks = node(container, "section", null, "quality-section"); node(checks, "h3", L("checks"));
  for (const name of ["structure_checks","method_checks","language_checks"]) {
    const check = v2 ? details[name] : null;
    const row = node(checks, "div", null, "quality-check-row"); node(row, "strong", L(name)); node(row, "span", check ? L(check.status) : L("unavailable"));
    if (check?.issues?.length) node(row, "p", check.issues.join("; "));
  }
  const diagnostics = node(container, "section", null, "quality-section"); node(diagnostics, "h3", L("issues"));
  const problems = [...(Array.isArray(details.unsupported_claims) ? details.unsupported_claims : []), ...(details.language_checks?.diagnostics || [])];
  const groups = new Map();
  for (const issue of problems) {
    const label = `${issue.section || L("unknownSection")} / ${L(issue.reason_code || "insufficient_support")}`;
    groups.set(label, (groups.get(label) || 0) + 1);
  }
  if (groups.size) {
    node(diagnostics, "h4", L("distribution"));
    const table = node(diagnostics, "table", null, "quality-distribution"); const body = node(table, "tbody");
    for (const [label,count] of groups) { const row = node(body, "tr"); node(row, "th", label); node(row, "td", count); const cell=node(row,"td"); meter(cell, count/problems.length, label); }
  } else node(diagnostics, "p", L("noIssues"));
  for (const issue of problems) {
    const item = node(diagnostics, "details", null, "quality-diagnostic"); node(item, "summary", `${L(issue.reason_code || "insufficient_support")} · ${issue.claim_text || issue.statement_id || "—"}`);
    node(item, "p", issue.explanation || "—");
    if (issue.local_error) node(item, "p", L(issue.local_error) === issue.local_error ? issue.local_error : `${L(issue.local_error)} [${issue.local_error}]`);
    if (issue.source_quote) node(item, "blockquote", issue.source_quote);
    for (const span of issue.support_validation?.spans || []) {
      node(item,"small",`${L("basis")} · ${span.evidence_id} · ${span.source}`);
      node(item,"blockquote",span.quote);
    }
    const button=node(item,"button",L("source"));button.type="button";button.disabled=!issue.source_quote;
    button.addEventListener("click",()=>onSelect({kind:"quality-diagnostic",id:issue.statement_id || issue.claim_id || "diagnostic",value:issue},button));
  }
  const independent = node(container,"section",null,"quality-section");node(independent,"h3",L("audits"));node(independent,"p",L("diagnosticOnly"));
  const snapshot=options.audits;
  if (snapshot?.error) node(independent,"p",snapshot.error,"notice-error");
  const attempts=snapshot?.data?.audits || [];
  if (!attempts.length) {node(independent,"p",L("auditNone"));return;}
  const select=node(independent,"select");select.setAttribute?.("aria-label",L("attempts"));
  for (const audit of attempts) {const option=node(select,"option",`${L(audit.status)} · ${audit.audit_id.slice(-8)}`);option.value=audit.audit_id;}
  select.value=snapshot.selected;select.addEventListener("change",()=>snapshot.select(select.value));
  const audit=attempts.find(a=>a.audit_id===snapshot.selected);if(!audit)return;
  node(independent,"p",`r${audit.source_report_revision} · SHA ${audit.source_report_sha256}`,"audit-binding");
  node(independent,"h4",`Judge · ${L(audit.judge?.status || audit.status)} · ${audit.judge_model || "—"}${audit.self_judged ? " · self_judged" : ""}`);
  node(independent,"p",`${L("faithfulness")}: ${display(audit.judge?.faithfulness)}`);
  meter(independent,numeric(audit.judge?.faithfulness),"Judge");
  const frozenClaims = audit.judge?.inventory_snapshot?.claims;
  const judgedClaims = audit.judge?.claim_assessments;
  node(independent,"p",`${L("scope")}: ${L("claims")} ${display(Array.isArray(judgedClaims) ? judgedClaims.length : null)} / ${display(Array.isArray(frozenClaims) ? frozenClaims.length : null)}`);
  node(independent,"p",audit.judge?.topic_coverage == null ? L("noTopics") : `Topic coverage: ${audit.judge.topic_coverage}`);
  node(independent,"p",`${L("structure_checks")}: ${L(audit.structure?.status || "unverified")} · ${L("method_checks")}: ${L(audit.method_status || "unverified")}`);
  for (const assessment of [...(audit.judge?.claim_assessments || []),...(audit.method_assessments || [])]) {
    const claim=[...(audit.judge?.inventory_snapshot?.claims || []),...(audit.method_inventory || [])].find(c=>c.claim_id===assessment.claim_id);
    const button=node(independent,"button",`${L("source")}: ${assessment.claim_text || assessment.claim_id}`);button.type="button";button.disabled=!claim;
    button.addEventListener("click",()=>onSelect({kind:"quality-diagnostic",id:assessment.claim_id,value:{...claim,explanation:assessment.explanation}},button));
  }
  node(independent,"h4",`Jev · ${L(audit.jev?.status || "not_requested")}`);node(independent,"p",L("confidence"));
  const summary=audit.jev?.summary;
  if (summary) {
    node(independent,"p",`${L("scope")}: ${L("claims")} ${display(summary.audited_claims)} / ${display(summary.planned_claims)} · ${L("questions")} ${display(summary.valid_questions)} / ${display(summary.planned_questions)}`);
    node(independent,"p",`${L("completion")}: ${rate(summary.protocol_completion_rate)} · ${L("agreement")}: ${rate(summary.agreement)} · ${L("localErrors")}: ${display(summary.local_error_questions)} · ${L("uncertain")}: ${display(summary.uncertain_questions)}`);
    meter(independent,numeric(summary.protocol_completion_rate),L("completion"));
  }
  for (const decision of audit.jev?.decisions || []) {
    const claim=(audit.judge?.inventory_snapshot?.claims || []).find(c=>c.claim_id===decision.claim_id);
    const button=node(independent,"button",`${decision.claim_id} · ${L(decision.choice)} · ${display(decision.confidence)}${decision.uncertain ? ` · ${L("uncertain")}` : ""}`);button.type="button";button.disabled=!claim;
    button.addEventListener("click",()=>onSelect({kind:"quality-diagnostic",id:decision.claim_id,value:claim},button));
  }
  if (audit.reason_codes?.length) node(independent,"p",audit.reason_codes.join("; "));
}


export function locateReportDiagnostic(report, survey, issue) {
  const normalize = value => String(value || "").replace(/\[E:[^\]]+\]/g, "").replace(/\[\d+(?:[,–-]\d+)*\]/g, "").replace(/[*_`]/g, "").replace(/\s+/g, " ").trim();
  report.querySelectorAll(".audit-source-highlight").forEach(item => item.classList.remove("audit-source-highlight"));
  const ordinal = /^unit:(\d+):\d+$/.exec(issue?.source_unit_id || "")?.[1];
  const bound = ordinal != null && survey.slice(issue.source_start, issue.source_end) === issue.source_quote ? report.querySelector(`[data-source-unit-ordinal="${ordinal}"]`) : null;
  const needle = normalize(issue?.source_quote);
  const matches = bound ? [bound] : [...report.querySelectorAll("p,li,td,blockquote")].filter(item => {const clean=item.cloneNode(true);clean.querySelectorAll(".citation-button").forEach(c=>c.remove());return needle && normalize(clean.textContent).includes(needle);});
  if (matches.length !== 1) return false;
  const details = matches[0].closest("details"); if (details) details.open = true;
  matches[0].classList.add("audit-source-highlight");matches[0].scrollIntoView({block:"center"});return true;
}
