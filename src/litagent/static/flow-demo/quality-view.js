/* Version-bound diagnostics never alter the original report or gate. */
function scoreText(result) {
  if (result?.skipped || result?.score == null) return "—";
  const numeric = Number(result.score);
  return Number.isFinite(numeric) ? numeric.toFixed(3) : "—";
}
function renderAssessmentEffect(container, claim, L, currentReport) {
  const effect = claim?.assessment_conflict_effect;
  const codes = new Set(["positive_status_with_negative_required_aspect", "negative_status_with_complete_positive_diagnostics"]);
  if (!currentReport || effect?.contract !== "host-assessment-conflict-effect-v1" || effect.policy !== "host-unverified-continue-v1" || !codes.has(effect.code)) return;
  const grid = node(container, "div", null, "quality-conflict-grid");
  const original = node(grid, "section");
  node(original, "strong", L("originalDecision"));
  node(original, "p", `${L(effect.original_status)} · ${effect.original_reason_code}`);
  const host = node(grid, "section");
  node(host, "strong", L("hostEffect"));
  node(host, "p", L("pending"));
  node(host, "small", effect.code);
  node(container, "p", L("conflictHelp"));
}
const LABELS = {
  passed: ["通过", "Passed"], failed: ["未通过", "Failed"], unverified: ["未核验", "Unverified"],
  disabled: ["未启用", "Disabled"],
  not_run: ["未运行", "Not run"],
  citation_accuracy: ["引用 ID 有效性", "Citation ID validity"], faithfulness: ["科学断言支持率", "Scientific assertion support"], internal_consistency: ["内部一致性", "Internal consistency"],
  idHelp: ["引用存在不代表论文支持该断言。", "A valid citation ID does not establish semantic support."],
  historical: ["历史评估口径；不能与新版直接比较。", "Historical evaluation scope; not directly comparable with v2."],
  previousSupport: ["前版支持原文口径：由模型抄写，不能与新片段选择口径直接比较。", "Previous quoted-support scope; not directly comparable with host-selected passages."],
  historicalV2: ["历史 v2：未启用支持原文契约，不能与新结果直接比较。", "Historical v2: support-span contract was not enabled; not directly comparable with new results."],
  current: ["新版评估 · 按断言实际引用核验", "Evaluation v2 · actual citations per assertion"],
  threshold: ["阈值", "Threshold"], unavailable: ["未记录／不可用", "Not recorded / unavailable"],
  supported: ["支持", "Supported"], unsupported: ["未获支持", "Unsupported"], pending: ["未核验", "Unverified"],
  originalDecision: ["模型原判", "Original model decision"], hostEffect: ["宿主处理结果", "Host effect"],
  conflictHelp: ["模型原判保持原样。语义冲突仅记为未核验，不能贡献支持数量；完成诊断不代表科学通过。", "The original decision is preserved. A semantic conflict remains unverified and adds no supported assertion; completed diagnostics do not establish scientific acceptance."],
  counts: ["科学断言数量", "Scientific assertion counts"], coverage: ["提取覆盖（处理单元／全部单元）", "Extraction coverage (processed / total units)"],
  checks: ["结构、方法与语言检查", "Structure, methodology and language checks"], language_checks: ["正文语言", "Report prose language"], support: ["支持依据契约", "Support basis contract"], basis: ["实际引用中的支持原文", "Support text from actual citations"], structure_checks: ["模板结构", "Template structure"], method_checks: ["方法披露", "Method disclosures"],
  not_applicable: ["不适用", "Not applicable"], issues: ["问题定位", "Located diagnostics"], noIssues: ["没有记录可定位问题；不代表额外核验通过。", "No located issues recorded; this does not establish additional verification."],
  started_no_replay: ["已发出，禁止自动重发", "Sent; automatic replay forbidden"], interrupted_no_replay: ["中断，禁止自动重发", "Interrupted; automatic replay forbidden"],
  distribution: ["问题分布（章节／类型）", "Issue distribution (section / type)"], source: ["定位原文及证据", "Locate source and evidence"], unknownSection: ["位置未记录", "Location not recorded"],
  autoAudits: ["Judge/Jev 复核 · 自动交付门禁", "Judge/Jev review · automatic delivery gate"],
  autoAuditHelp: ["自动策略要求本版本的 Evaluation、Judge 和 Jev 全部通过。交付状态以绑定版本与 SHA 的系统验收记录为准。", "The automatic policy requires Evaluation, Judge and Jev to pass for this version. Delivery follows the system acceptance record bound to its version and SHA."],
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
  analysis_semantics_inadequate: ["模型判定分析内容不足", "Model found inadequate analysis"],
  analysis_claims_missing: ["缺少本章科学断言", "No scientific claims for this chapter"],
  analysis_duplicate_claim: ["断言 ID 重复", "Duplicated claim IDs"],
  analysis_unknown_claim: ["未知断言 ID", "Unknown claim IDs"],
  analysis_wrong_section_claim: ["断言不属于本章", "Claims do not belong to this chapter"],
  analysis_claim_not_supported: ["所用科学断言尚未获支持", "Selected scientific claims are not supported"],
  invalid_response: ["响应契约无效", "Invalid response contract"],
  evaluation_analysis_claim_ownership_invalid: ["章节断言归属无效，整体未核验", "Invalid chapter claim ownership; overall evaluation unverified"],
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
function renderRunBudget(container, value, runId, en, plan) {
  if (value == null && !runId) return;
  const panel = node(container, "section", null, "quality-section");
  node(panel, "h3", en ? "Run budget usage (last recorded)" : "运行预算用量（最后记录）");
  const fields = ["plan_identity", "known_tokens", "requests", "usage_uncertain", "stop_reason", "unknown_hold"];
  const keys = new Set(["contract", "scope", "run_id", "availability", "reason", ...fields]);
  const reasons = new Set(["strict_usage_unknown", "strict_worker_failed", "planned_usage_bound_violation", "planned_ledger_persistence_failed"]);
  const validBase = value && typeof value === "object" && !Array.isArray(value)
    && Object.keys(value).every(key => keys.has(key)) && [...keys].every(key => Object.hasOwn(value, key))
    && value.contract === "run-budget-state-v1" && value.scope === "run"
    && typeof runId === "string" && runId.length > 0 && value.run_id === runId;
  const integer = number => Number.isSafeInteger(number) && number >= 0;
  const planMatches = !["planned-budget-v2", "planned-budget-v3", "planned-budget-v4", "planned-budget-v5", "planned-budget-v6", "planned-budget-v7"].includes(plan?.contract) || !Object.hasOwn(plan, "identity") || plan.identity === value?.plan_identity;
  const valid = validBase && planMatches && value.availability === "available" && value.reason === null
    && typeof value.plan_identity === "string" && /^[a-f0-9]{64}$/.test(value.plan_identity)
    && [value.known_tokens, value.unknown_hold, value.requests].every(integer)
    && typeof value.usage_uncertain === "boolean" && (value.stop_reason === null || reasons.has(value.stop_reason))
    && !(value.unknown_hold > 0 && !value.usage_uncertain);
  if (!valid) {
    const missing = value == null || (validBase && value.availability === "missing" && value.reason === "not_recorded" && fields.every(key => value[key] === null));
    node(panel, "p", missing ? (en ? "Not recorded" : "未记录") : (en ? "Budget record unavailable" : "预算记录不可用"));
  } else {
    node(panel, "p", `${en ? "Known usage" : "已知用量"}: ${value.known_tokens} tokens`);
    node(panel, "p", `${en ? "Unknown reservation" : "未知占用"}: ${value.unknown_hold} tokens`);
    node(panel, "p", `${en ? "Requests sent" : "已发请求"}: ${value.requests}`);
    node(panel, "p", value.usage_uncertain ? (en ? "Usage incomplete" : "用量不完整") : (en ? "No unknown reservation recorded in this snapshot" : "本次快照未记录未知占用"));
    if (value.stop_reason) node(panel, "p", `${en ? "Recorded stop reason" : "已记录停止原因"}: ${value.stop_reason}`);
  }
  node(panel, "p", en ? "Run-scoped last recorded usage, not a score or cost for this report version. Unknown reservations are conservative holds, not known billed usage, refunds or authorization to replay. Reservations and allocation-time savings are shown separately." : "这是整个运行的最后记录，不是当前报告版本的评分或成本。未知占用是保守保留额，不是已知实际用量、退款或重发授权；规划预留及分配时差额另行显示。");
}
export function renderQuality(container, evaluation, quality, delivery, onSelect, t = key => key, options = {}) {
  const L = key => LABELS[key]?.[options.locale === "en" ? 1 : 0] || key;
  const details = evaluation?.faithfulness?.details || {};
  const v2 = ["survey-evaluation-v2", "survey-evaluation-v3"].includes(details.evaluation_version);
  const noFinalReport = details.skipped_reason === "final_report_unavailable";
  const hasPipeline = options.pipelineVerification && Object.keys(options.pipelineVerification).length > 0;
  container.replaceChildren();
  renderRunBudget(container, options.budgetState, options.runId, options.locale === "en", options.budgetPlan);
  const budgetPlan = options.budgetPlan;
  if (["planned-budget-v2", "planned-budget-v3", "planned-budget-v4", "planned-budget-v5", "planned-budget-v6", "planned-budget-v7"].includes(budgetPlan?.contract)) {
    const en = options.locale === "en", panel = node(container, "section", null, "quality-section");
    node(panel,"h3",en ? "Workflow budget admission" : "完整流程预算准入");
    node(panel,"p",budgetPlan.status === "preflight_rejected" ? (en ? "Preflight rejected · writing was not started" : "预算检查未通过 · 未启动正文生成") : (en ? "Budget admitted · scientific verification still required" : "预算准入通过 · 仍须科学核验"));
    node(panel,"p",`${en ? "Conservative reservation / allowance / ceiling" : "保守预留／计划额度／硬上限"}: ${budgetPlan.required_tokens} / ${budgetPlan.planned_tokens} / ${budgetPlan.maximum_tokens} tokens`);
    node(panel,"p",`${en ? "Requests / maximum" : "请求数／上限"}: ${budgetPlan.planned_requests} / ${budgetPlan.maximum_requests} · ${budgetPlan.reason || "—"}`);
    const adaptive = ["planned-budget-v6", "planned-budget-v7"].includes(budgetPlan.contract);
    const joint = budgetPlan.joint_round_escrow?.contract === "joint-round-escrow-v1" ? budgetPlan.joint_round_escrow : null;
    node(panel,"p",en ? "Run-scoped planning, not a score for this report version. Reservations are not actual charges." : "这是运行级规划，不是该报告版本的评分。预留不等于实际费用。");
    node(panel,"p",adaptive
      ? (en ? "Initial batch reservations are a planning baseline. Each evaluation round checks its complete requests; future candidates require another resource check. The initial batch count is not a guaranteed upper bound." : "初始批次预留是规划基线。每轮核验会检查完整请求，未来候选仍须重新检查资源；初始批数不是保证上界。")
      : (en ? "Unknown future input uses an enforced upper bound." : "未知后续输入使用可执行的上界。"));
    if (joint) node(panel,"p",`${en ? "Locked assessment domain per round: maximum requests / per-request cap / round total" : "每轮锁定核验资源域：最多请求数／单请求上限／整轮总预留"}: ${joint.assessment.max_requests} / ${joint.assessment.request_cap} / ${joint.assessment.reserved_tokens} · ${en ? "Future repair rounds retain the same domain; every complete candidate is checked before sending." : "后续修复轮保留同一资源域；每份完整候选仍在发送前检查。"}`);
    if (["planned-budget-v3", "planned-budget-v4", "planned-budget-v5", "planned-budget-v6", "planned-budget-v7"].includes(budgetPlan.contract)) node(panel,"p",`${budgetPlan.status === "admitted" ? (en ? "Locked shared repair attempts / maximum" : "写作前锁定的共享修复次数／最高") : (en ? "Rejected repair path / no allocation" : "未准入修复路径／尚未分配")}: ${budgetPlan.repair_limit} / 2 · ${en ? "Complete initial Evaluation included" : "已计入首次完整 Evaluation"}`);
    if (["planned-budget-v4", "planned-budget-v5", "planned-budget-v6", "planned-budget-v7"].includes(budgetPlan.contract) && budgetPlan.envelope) {
      const scope=budgetPlan.envelope;
      node(panel,"p",`${budgetPlan.status === "admitted" ? (en ? "Locked content capacity" : "写作前锁定的整体容量") : (en ? "Candidate capacity · unallocated" : "候选容量 · 尚未分配")}: ${en ? "body units / summary units per section / model assertions / assessment batches" : "正文单元／每节摘要结论单元／模型断言／支持批次"} ≤ ${scope.body_max_units} / ${scope.summary_max_units_per_section} / ${scope.model_assertions} / ${scope.assessment_batches}`);
      if (["planned-budget-v6", "planned-budget-v7"].includes(budgetPlan.contract)) node(panel,"p",en ? "Initial verification capacity; settled savings may fund complete additional batches. Capacity does not establish scientific support." : "这是初始核验容量；已结算释放的额度可用于完整扩批，容量准入不代表科学支持。");
      else node(panel,"p",en ? "Declared output and complete request size limits are enforced. Exceeding them stops the run without truncation; capacity admission does not establish scientific support." : "声明的输出及完整请求规模边界会被强制检查；越界保留原文并停止，不截断，不代表科学核验通过。");
    }
    const disclosure=node(panel,"details");node(disclosure,"summary",en ? "Stage reservations" : "各阶段预留");
    const table=node(disclosure,"table",null,"quality-distribution"),body=node(table,"tbody");
    for(const [name,stage] of Object.entries(budgetPlan.stages || {})){const row=node(body,"tr");node(row,"th",name);node(row,"td",joint ? `${en ? "Maximum requests" : "最多请求数"}: ${stage.count} · ${en ? "Per-request cap" : "单请求上限"}: ${stage.request_cap} · ${en ? "Total reservation" : "总预留"}: ${stage.reserved_tokens}` : `${stage.count} × ${stage.request_cap} = ${stage.reserved_tokens}`);node(row,"td",stage.bound);}
  }
  const capacity = details.stages?.capacity;
  if (capacity) {
    const panel=node(container,"section",null,"quality-section"),en=options.locale === "en";
    node(panel,"h3",en ? "Capacity and complete assessment" : "容量与完整核验");
    node(panel,"p",`${capacity.status} · ${capacity.reason || "—"}`);
    if (capacity.round) node(panel,"p",`${en ? "Evaluation round / branch / original assessment state" : "核验轮次／执行分支／原稿核验状态"}: ${capacity.round.number} / ${capacity.round.mode} / ${capacity.round.status}`);
    if (capacity.target_assertions != null) node(panel,"p",`${en ? "Refinement target, pending candidate resource checks: assertions / batches" : "精简目标，候选仍须资源检查：断言／批次"}: ${capacity.target_assertions} / ${capacity.target_batches}`);
    if (capacity.assessment_reserved_tokens != null) node(panel,"p",`${en ? "Current complete assessment reservation (not billed usage)" : "当前完整核验请求预留（非实际费用）"}: ${capacity.assessment_reserved_tokens} tokens`);
    node(panel,"p",`${en ? "Body / abstract and conclusion units" : "正文／摘要结论单元"}: ${capacity.authored_unit_counts?.body ?? "—"} / ${capacity.authored_unit_counts?.summary ?? "—"}`);
    const frozenEnvelope = details.stages?.writer_envelope;
    node(panel,"p",`${en ? "Initial / effective / actual assertions / batches" : "初始／有效／实际断言／批次"}: ${capacity.initial_assertions ?? frozenEnvelope?.capacity?.initial_assertions ?? "—"} / ${capacity.effective_assertions ?? frozenEnvelope?.model_assertions ?? "—"} / ${capacity.actual_assertions ?? "—"} / ${capacity.actual_batches ?? "—"}`);
    if (capacity.reserved_with_margin != null) node(panel,"p",`${en ? "Remaining workflow plan including spent / fixed allowance" : "含已知消耗的完整计划／固定额度"}: ${capacity.reserved_with_margin} / ${capacity.fixed_allowance} tokens`);
    if (capacity.known_tokens_at_allocation != null) node(panel,"p",`${en ? "Known usage / future locks at allocation / settled reservation savings" : "分配时已知用量／后续锁定／已结算预留差额"}: ${capacity.known_tokens_at_allocation} / ${capacity.future_locked_tokens} / ${capacity.settled_reservation_savings} tokens`);
    node(panel,"p",en ? "Every retained scientific assertion requires assessment. Refinement does not itself establish factual support." : "最终保留的每项科学断言均须核验；精简正文并不代表事实已获支持。");
  }
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
  const currentFailure = options.writerFailure;
  if (noFinalReport && currentFailure?.code === "writer_flat_schema_invalid") {
    const en=options.locale === "en";
    for (const issue of currentFailure.diagnostics || []) {
      if (issue.summary_constraints?.contract !== "effective-summary-limits-v1" || !["abstract","conclusion"].includes(issue.section) || !Number.isInteger(issue.actual) || !Number.isInteger(issue.maximum)) continue;
      node(gate,"p",`${en ? "Summary structure validation failed" : "摘要结构校验失败"} · ${issue.section} · ${en ? "Actual / maximum units" : "实际／最多单元"}: ${issue.actual} / ${issue.maximum}`,"notice-error");
    }
  }
  if (options.sha && currentFailure?.source_report_sha256 === options.sha) {
    const en=options.locale === "en";
    node(gate,"p",`${en ? "Terminal failure" : "最终失败原因"}: ${currentFailure.family} · ${currentFailure.code}`,"notice-error");
    if(details.source_report_sha256 === options.sha) {
      const upstream=details.skipped_reason || details.error || details.reason;
      if(upstream)node(gate,"p",`${en ? "Upstream Evaluation" : "上游 Evaluation 原因"}: ${upstream}`);
    }
    node(gate,"p",en ? "A rejected repair is not a scientific verdict; incomplete assessment remains unverified." : "修复候选被拒不等于科学事实不支持；未完成核验仍显示未核验。");
  }
  node(gate, "p", noFinalReport ? (options.locale === "en" ? "Writing validation failed; Evaluation was not run." : "写作校验失败；Evaluation 未运行。") : details.evaluation_version === "survey-evaluation-v3" ? `${options.locale === "en" ? "Evaluation v3 · complete batched assessment" : "评估 v3 · 完整分批核验"}` : v2 ? `${L(details.support_contract === "host-support-passages-v2" ? "current" : details.support_contract === "cited-support-spans-v1" ? "previousSupport" : "historicalV2")} · ${details.evaluation_version}` : hasPipeline && !Object.keys(evaluation || {}).length ? (options.locale === "en" ? "Evaluation was not run; no report scores were produced." : "Evaluation 未运行；尚未产生正文评分。") : L("historical"));
  if (options.revision) node(gate, "p", `r${options.revision}${options.sha ? ` · SHA ${options.sha}` : ""}`, "audit-binding");
  if (v2) node(gate,"p",`${L("support")}: ${details.support_contract || L("unavailable")}`);
  if (!noFinalReport && (options.bodyDiagnostics || details.body_analysis_contract)) {
    const section=node(container,"section",null,"quality-section"),en=options.locale === "en";
    node(section,"h3",en ? "Academic body · analytical depth" : "学术正文 · 分析质量");
    const d=options.bodyDiagnostics;
    if(d)node(section,"p",`${en ? "Academic characters / appendix and bibliography" : "学术正文字符／附录与书目"}: ${d.academic_characters} / ${d.non_body_characters} · ${en ? "Repeated units" : "重复单元"}: ${d.repeated_units}`);
    node(section,"p",en ? "Length and counts are diagnostics, not a quality score. Preparation candidates are not verified findings." : "长度与数量仅用于诊断；准备表中的候选材料不等于已核验科学发现。");
    for(const row of details.analysis_check?.sections || []) {
      node(section,"p",`${row.section_key}: ${L(row.adequate ? "passed" : "failed")} · ${row.explanation}`);
      if(row.host_validation) {
        const model=row.model_adequate == null ? "unverified" : row.model_adequate ? "passed" : "failed";
        node(section,"p",`${en ? "Model analysis decision" : "模型分析结论"}: ${L(model)} · ${en ? "Host gate checks" : "宿主门禁检查"}: ${L(row.host_validation.status)} · ${(row.host_validation.failure_codes || []).map(L).join("; ")}`);
      } else {
        node(section,"p",`${en ? "Host ownership check" : "宿主归属核验"}: ${L("unavailable")}`);
      }
      for(const id of row.claim_ids || []) {
        const claim=(details.assertions || []).find(c=>c.claim_id===id);if(!claim)continue;
        const button=node(section,"button",`${L("source")} · ${id}`);button.type="button";button.addEventListener("click",()=>onSelect({kind:"quality-diagnostic",id,value:claim},button));
      }
    }
    if(options.analysisPlan) {
      const plan=options.analysisPlan;
      const detail=node(section,"details",null,"quality-diagnostic");
      node(detail,"summary",en ? "Evidence preparation · candidates" : "写作前证据准备 · 候选材料");
      node(detail,"p",`${en ? "Candidate cards / core questions" : "候选事实卡／核心问题"}: ${plan.sources?.length || 0} / ${plan.requirements?.length || 0}`);
      const table=node(detail,"table",null,"quality-distribution");const body=node(table,"tbody");
      for(const req of plan.requirements || []) {const row=node(body,"tr");node(row,"th",req.text);node(row,"td",String(req.candidate_card_ids?.length || 0));node(row,"td",en ? (req.match_rule === "literal_source_match" ? "Literal matches only; a miss does not prove missing evidence" : "Selected source pool; answer verified separately") : (req.match_rule === "literal_source_match" ? "仅字面匹配；零命中不等于缺少证据" : "所选来源池；回答覆盖另行核验"));}
    }
  }
  const acceptance = options.automaticAcceptance;
  const verification = acceptance?.verification || options.verification;
  const optionalHelp = options.locale === "en" ? "Complete Evaluation is required. Only enabled Judge/Jev reviews affect delivery; disabled reviews make no calls." : "完整 Evaluation 为必需门禁；仅启用的 Judge/Jev 参与交付判定，关闭时不调用。";
  if (acceptance) {
    const panel = node(container, "section", null, "quality-section");
    node(panel, "h3", options.locale === "en" ? "Automatic acceptance · system" : "严格自动验收 · 系统判定");
    node(panel, "p", `${L(acceptance.status)} · SHA ${acceptance.report_sha256}`);
    for (const [name, status] of Object.entries(acceptance.gates || {})) node(panel, "p", `${name}: ${L(status)}`);
    if (acceptance.version === "automatic-acceptance-v1") node(panel, "p", options.locale === "en" ? "Historical v1 record: unrun audits may have been recorded as failures. Consult the request records; this view does not re-evaluate the report." : "历史 v1 判定记录：未启动核验可能被记录为失败。请以调用记录为准；本页不重新评估报告。", "metric-unavailable");
    if (acceptance.policy === "strict_auto_v2" && acceptance.version === "automatic-acceptance-v3" && !acceptance.extraction_ledger_contract) node(panel, "p", options.locale === "en" ? "Historical extraction gate: accepted-source freezing was not recorded. Preflight may have recorded extraction as failed; consult the original extraction receipt. This view does not re-evaluate the report." : "历史提取门禁：未记录已接受证据集合冻结契约；前置检查可能将提取记录为失败。请核对原始提取收据，本页不重新评估报告。", "metric-unavailable");
    node(panel, "p", (acceptance.failure_codes || []).join(", "));
    node(panel, "p", options.locale === "en" ? "No human calibration. self_judged is not an independent gold standard." : "未经真人校准；self_judged 不代表独立金标准。");
  }
  if (details.evaluation_version === "survey-evaluation-v3") {
    const en = options.locale === "en";
    const strict = node(container, "section", null, "quality-section");
    node(strict, "h3", en ? "Strict acceptance · scientific support required: 100%" : "严格交付门禁 · 科学支持率要求 100%");
    node(strict, "p", `${L(details.strict_acceptance?.status || "unverified")} · ${(details.strict_acceptance?.failure_codes || []).join(", ")}`);
    node(strict, "p", verification ? optionalHelp : acceptance?.policy === "strict_auto_v2" ? (en ? "Jev extraction verification, complete Evaluation and Judge must pass; no human approval." : "Jev提取核验、完整Evaluation与Judge通过才可自动交付；无人工审批。") : acceptance ? (en ? "Evaluation, Judge and Jev must all pass; no human approval stage." : "Evaluation、Judge、Jev 全部通过才可自动交付；无人工审批阶段。") : en ? "Execution complete ≠ verified ≠ human approved. Judge/Jev do not override this gate." : "执行完成、自动核验通过和人工批准是三个状态。Judge/Jev 不覆盖此门禁。");
    const stages = details.stages || {};
    node(strict, "p", `${en ? "Assessed assertions" : "已处理断言"}: ${display(stages.processed_assertions)} / ${display(stages.expected_assertions)}`);
    const table = node(strict, "table", null, "quality-distribution quality-batches"); const body = node(table, "tbody");
    for (const batch of stages.batches || []) { const row = node(body, "tr"); node(row, "th", `${en ? "Batch" : "批次"} ${batch.index + 1}`); node(row, "td", L(batch.status)); node(row, "td", `${batch.claim_ids?.length || 0} ${L("claims")}`); }
    node(strict, "p", `${en ? "Global consistency" : "全局一致性"}: ${L(stages.global_check || "unverified")}`);
    if (details.global_findings_contract) {
      node(strict,"p",`${en ? "Inventory validity" : "抽取清单有效性"}: ${L(details.inventory_check?.status || "unverified")}`);
      node(strict,"p",en ? "Completed requests and partial scientific decisions do not establish a complete valid evaluation. Inventory defects cannot authorize rewriting scientific prose." : "请求完成和部分科学判断不代表整体验证有效。抽取清单缺陷不能自动授权改写正文。");
      for(const issue of details.global_issues || stages.global_issues || []) {
        const owner = issue.owner === "inventory" ? (en ? "Inventory defect" : "抽取清单问题") : issue.owner === "report" ? (en ? "Report issue" : "正文问题") : (en ? "Unclassified global finding" : "未分类全局问题");
        node(strict,"p",`${owner} · ${issue.code || "—"} · ${issue.explanation}`);
      }
    }
    if(stages.global_reason_code)node(strict,"p",L(stages.global_reason_code));
    if(details.scope_notice_contract)node(strict,"p",`${en ? "Substantive limitations" : "具体科学局限"}: ${L(details.limitations_check?.adequate === true ? "passed" : "unverified")} · ${details.limitations_check?.explanation || L("unavailable")}`);
    for (const requirement of details.coverage_requirements || []) {
      const verdict = (details.core_coverage || []).find(c => c.requirement_id === requirement.requirement_id);
      const supportedIds = new Set((details.assertions || []).filter(c => c.kind === "scientific" && c.status === "supported").map(c => c.claim_id));
      const verified = details.coverage?.complete === true && verdict?.answered === true && Array.isArray(verdict.claim_ids) && verdict.claim_ids.length > 0 && verdict.claim_ids.every(id => supportedIds.has(id));
      node(strict, "p", `${requirement.text}: ${L(verified ? "passed" : "unverified")}`);
    }
    const cards = node(strict, "details", null, "quality-diagnostic");
    node(cards, "summary", `${en ? "Fact card bindings" : "事实卡绑定"} · ${Object.keys(details.fact_card_map || {}).length}`);
    node(cards, "p", en ? "A valid source location does not establish scientific support. Missing metadata remains unknown." : "来源定位有效不代表科学支持成立；缺失元数据保持未知。");
    for (const [id, card] of Object.entries(details.fact_card_map || {})) {
      node(cards, "p", `${id} · ${card.paper_id || "—"} · E:${card.evidence_id || "—"}`);
      node(cards, "p", `${card.content_scope || "—"} · ${card.source_locator || card.chunk_key || "—"} · ${en ? "page / block" : "页码／block"}: ${card.page ?? "—"} / ${card.block_index ?? "—"}`);
      node(cards, "p", `SHA ${card.content_sha256 || "—"}`, "audit-binding");
    }
  }
  const counts = node(container, "section", null, "quality-section"); node(counts, "h3", L("counts"));
  const grid = node(counts, "div", null, "quality-count-grid");
  for (const [key,field] of [["supported","supported_count"],["unsupported","unsupported_count"],["pending","unverified_count"]]) {
    const box = node(grid, "div", null, `quality-count ${key}`); node(box, "strong", display(v2 ? numeric(details[field]) : null)); node(box, "span", L(key));
    if (v2 && numeric(details.factual_claim_count) > 0 && numeric(details[field]) != null) meter(box, details[field]/details.factual_claim_count, L(key));
  }
  node(counts, "p", `${L("coverage")}: ${v2 ? `${display(details.coverage?.processed_units)} / ${display(details.coverage?.source_units)}` : L("unavailable")}`);
  if(details.assessment_stage === "host_validated" && details.coverage?.complete !== true)node(counts,"p",options.locale === "en" ? "Host-checked decisions; the overall evaluation remains invalid and cannot authorize delivery." : "以上为宿主校验过的局部判断；整体验证仍无效，不能交付。");
  if(details.partial_assessment?.stage === "host_validated") {
    const p=details.partial_assessment,box=node(counts,"section",null,"quality-section");
    node(box,"h4",options.locale === "en" ? "Partial host-checked diagnostics · not a final score" : "部分宿主校验诊断 · 非最终分数");
    node(box,"p",`${L("supported")}: ${display(p.supported_count)} · ${L("unsupported")}: ${display(p.unsupported_count)} · ${L("pending")}: ${display(p.unverified_count)}`);
    node(box,"p",`${options.locale === "en" ? "Processed assertions" : "已处理断言"}: ${display(p.processed_assertions)} / ${display(p.expected_assertions)}`);
    const list=node(box,"details",null,"quality-diagnostic");
    node(list,"summary",options.locale === "en" ? "Processed partial assertions · locate source" : "已处理的局部断言 · 定位原文");
    for(const claim of p.assertions || []) {
      node(list,"p",`${claim.claim_id} · ${L(claim.status)} · ${claim.claim_text}`);
      renderAssessmentEffect(list, claim, L, options.sha && details.source_report_sha256 === options.sha);
      if(claim.local_error)node(list,"p",claim.local_error);
      const button=node(list,"button",L("source"));button.type="button";button.disabled=!claim.source_quote;
      button.addEventListener("click",()=>onSelect({kind:"quality-diagnostic",id:claim.claim_id,value:claim},button));
    }
  }
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
    renderAssessmentEffect(item, issue, L, options.sha && details.source_report_sha256 === options.sha);
    if (issue.local_error) node(item, "p", L(issue.local_error) === issue.local_error ? issue.local_error : `${L(issue.local_error)} [${issue.local_error}]`);
    if (issue.source_quote) node(item, "blockquote", issue.source_quote);
    for (const span of issue.support_validation?.spans || []) {
      node(item,"small",`${L("basis")} · ${span.evidence_id} · ${span.source}`);
      node(item,"blockquote",span.quote);
    }
    const button=node(item,"button",L("source"));button.type="button";button.disabled=!issue.source_quote;
    button.addEventListener("click",()=>onSelect({kind:"quality-diagnostic",id:issue.statement_id || issue.claim_id || "diagnostic",value:issue},button));
  }
  const automaticPolicy = ["strict_auto_v1", "strict_auto_v2"].includes(acceptance?.policy || details.quality_policy || options.qualityPolicy);
  const independent = node(container,"section",null,"quality-section");
  node(independent,"h3",verification ? (options.locale === "en" ? "Judge · optional final report review" : "Judge · 可选终稿复核") : (acceptance?.policy || details.quality_policy || options.qualityPolicy) === "strict_auto_v2" ? (options.locale === "en" ? "Judge · report delivery gate" : "Judge · 正文交付门禁") : L(automaticPolicy ? "autoAudits" : "audits"));
  node(independent,"p",verification ? optionalHelp : (acceptance?.policy || details.quality_policy || options.qualityPolicy) === "strict_auto_v2" ? (options.locale === "en" ? "Report gate: complete Evaluation and Judge. Jev was run before writing." : "正文门禁：完整 Evaluation 与 Judge；Jev 在写作前核验提取。") : L(automaticPolicy ? "autoAuditHelp" : "diagnosticOnly"));
  if (verification && !verification.judge) node(independent,"p",`Judge · ${L("disabled")}`);
  if (verification && !verification.jev) {
    const disabledJev = node(container,"section",null,"quality-section");
    node(disabledJev,"h3",`Jev · ${L("disabled")}`);
  }
  const extraction = options.extractionVerification;
  if (options.writerContract || options.writerFailure) {
    const writer = node(container,"section",null,"quality-section");
    node(writer,"h3", options.locale === "en" ? "Writing protocol and outcome" : "正文协议与执行结果");
    if (options.writerContract) node(writer,"p", options.writerContract);
    if (options.writerFailure) {
      if (options.sha && options.writerFailure.source_report_sha256 && options.writerFailure.source_report_sha256 !== options.sha) node(writer,"p",options.locale === "en" ? "This run-level failure is not bound to the selected report version." : "此运行失败不绑定当前选择的报告版本。");
      else node(writer,"p",`${options.writerFailure.family} · ${options.writerFailure.code}`);
    }
    node(writer,"p", options.locale === "en" ? "Chapter recommendations do not restrict the frozen source pool. Scientific support is verified independently by Evaluation." : "章节名单仅作推荐，引用限于冻结总证据；科学支持由 Evaluation 独立核验。");
  }
  const history = options.evaluationHistory;
  const cleanup = history?.editorial_cleanup;
  const editorialBound = cleanup?.version === "assessed-repair-repetition-v1"
    && cleanup.requires_full_reassessment === true
    && cleanup.result_sha256 === options.sha
    && cleanup.candidate_sha256 === (history.parent_report_sha256 || history.source_report_sha256)
    && history.prior_history?.source_report_sha256 === cleanup.candidate_sha256
    && history.final_evaluation?.faithfulness?.details?.source_report_sha256 === options.sha;
  if (history && history.version === "version-bound-evaluation-history-v1" && (editorialBound || (!cleanup && history.source_report_sha256 === options.sha))) {
    const section=node(container,"section",null,"quality-section");const en=options.locale === "en";
    node(section,"h3",editorialBound ? (en ? "Source-version repairs and current Evaluation" : "来源版本修复与当前核验") : (en ? "Evaluation attempts · this report version" : "本版本 Evaluation 与修复过程"));
    const show=(label,evaluation,reason,committed)=>{
      const box=node(section,"details",null,"quality-diagnostic");node(box,"summary",label);
      const d=evaluation?.faithfulness?.details;
      node(box,"p",reason || "");
      node(box,"p",en ? (committed ? "Committed report" : "Diagnostic record; not a committed report") : (committed ? "已提交正文" : "诊断记录；不代表已提交正文"));
      if(!d){node(box,"p",en ? "Full Evaluation not run" : "未运行完整 Evaluation");return;}
      node(box,"p",`${en ? "Unit coverage" : "单元覆盖"}: ${display(d.coverage?.processed_units)} / ${display(d.coverage?.source_units)}`);
      node(box,"p",`${L("supported")}: ${display(d.supported_count)} · ${L("unsupported")}: ${display(d.unsupported_count)} · ${L("pending")}: ${display(d.unverified_count)}`);
      node(box,"p",`${en ? "Global check" : "全局检查"}: ${L(d.stages?.global_check || "unverified")}`);
      for(const issue of d.global_issues || d.stages?.global_issues || [])node(box,"p",issue.explanation);
    };
    const earlier = editorialBound ? history.prior_history : history;
    show(editorialBound ? (en ? "Source-version initial Evaluation" : "来源版本首次 Evaluation") : (en ? "Initial Evaluation" : "首次 Evaluation"),earlier.initial_evaluation,"",false);
    let candidateNumber = 0;
    for(const attempt of earlier.attempts || []) {
      if(attempt.attempted === false) {
        show(en ? "Repair preflight · no request sent" : "修复预检 · 未发送请求",null,attempt.reason_code,false);
      } else {
        candidateNumber += 1;
        const recovery = attempt.kind === "inventory_recovery";
        show(`${recovery ? (en ? "Inventory re-extraction · prose unchanged" : "断言清单重抽取 · 正文未修改") : editorialBound ? (en ? "Source-version repair candidate" : "来源版本修复候选") : (en ? "Repair candidate" : "修复候选")} ${candidateNumber}`,attempt.candidate_evaluation,attempt.reason_code,attempt.committed);
      }
    }
    if(editorialBound)node(section,"p",en ? "Local cleanup removed one confirmed redundant sentence. Earlier evaluations belong to source versions; the current report was assessed in full again." : "本地整理移除一条已确认重复句。前序评估属于来源版本；当前正文已重新完整核验。");
    show(en ? "Final report Evaluation" : "最终正文 Evaluation",history.final_evaluation,"",true);
  }
  const repairs = options.targetedRepairState;
  if (repairs) {
    const section=node(container,"section",null,"quality-section");const en=options.locale === "en";
    node(section,"h3",en ? "Bounded targeted repairs" : "有限定向修复");
    if(repairs.version !== "shared-targeted-repair-limit-v1" || repairs.limit !== 2 || !Array.isArray(repairs.attempts))node(section,"p",L("unavailable"));
    else {
      node(section,"p",`${en ? "Attempts consumed" : "已消耗次数"}: ${repairs.attempts.length} / 2`);
      if(budgetPlan?.envelope?.closeout_profile === "engineering-closeout-v1") {
        const inventory=repairs.attempts.filter(a=>["bounded-inventory-recovery-v1","bounded-inventory-recovery-v2"].includes(a.purpose)).length;
        node(section,"p",`${en ? "Inventory rechecks" : "清单独立复核"}: ${inventory} / 1 · ${en ? "Prose repairs" : "正文修复"}: ${repairs.attempts.length-inventory} / 2`);
        node(section,"p",en ? "Rechecking unchanged content does not prove it was split. Current full Evaluation determines acceptance." : "同一清单重新核验不代表已完成拆分；最终以当轮完整 Evaluation 为准。");
      }

      node(section,"p",verification ? (en ? "A completed repair call does not establish scientific support; complete Evaluation and any enabled reviews must still pass." : "修复调用完成不代表科学支持；仍需完整 Evaluation 及已启用的复核通过。") : (en ? "A completed repair call does not establish scientific support; full Evaluation and Judge are still required." : "修复调用完成不代表科学支持；仍需完整 Evaluation 和 Judge。"));
      for(const attempt of repairs.attempts)node(section,"p",`${["bounded-inventory-recovery-v1", "bounded-inventory-recovery-v2"].includes(attempt.purpose) ? (en ? "Inventory re-extraction" : "断言清单重抽取") : attempt.cause === "capacity" ? (en ? "Body capacity refinement" : "正文容量精简") : attempt.stage === "native_initial" ? (en ? "Initial native unit" : "初稿单元") : (en ? "Evaluation repair" : "正文定向修复")}: ${L(attempt.status)}`);
    }
  }
  if (hasPipeline) {
    const section=node(container,"section",null,"quality-section");
    node(section,"h3",options.locale === "en" ? "This run · pipeline stages" : "本轮流程阶段");
    if(options.pipelineVerification.assessment_scope === "generation_only")node(section,"p",options.locale === "en" ? "Generation diagnostics belong to rejected draft candidates, not Evaluation scores." : "生成诊断属于被拒草稿，不是 Evaluation 评分。");
    for (const [stage, status] of Object.entries(options.pipelineVerification.stages || {})) node(section,"p",`${stage}: ${L(status)}`);
    for (const [service, cost] of Object.entries(options.pipelineVerification.costs || {})) node(section,"p",`${service}: ${cost.tokens == null || !cost.usage_complete ? L("unavailable") : cost.tokens} tokens · ${cost.requests} calls`);
    if(options.pipelineVerification.reason)node(section,"p",String(options.pipelineVerification.reason),"notice-error");
    for (const issue of options.pipelineVerification.writing_diagnostics || []) {
      const detail=node(section,"details",null,"quality-diagnostic");
      node(detail,"summary",`${options.locale === "en" ? "Rejected draft unit" : "被拒绝的初稿单元"}: ${issue.location} · ${issue.subreason || issue.reason_code}`);
      node(detail,"p",issue.candidate_text);
      node(detail,"p",`${options.locale === "en" ? "JSON location" : "JSON位置"}: ${issue.raw_json_path}`);
      if(issue.phase)node(detail,"p",`${options.locale === "en" ? "Generation stage" : "生成阶段"}: ${issue.phase}`);
      if(issue.candidate_sha256)node(detail,"p",`SHA ${issue.candidate_sha256}`,"audit-binding");
    }
  }
  if (extraction) {
    const section=node(container,"section",null,"quality-section");
    const en=options.locale === "en";
    node(section,"h3",en ? "Jev · extraction verification" : "Jev · 提取核验");
    node(section,"p",en ? "Verifies extracted claims against original sources; this is not report support or accuracy." : "检查提取 claim 与原文的支持关系；不代表正文支持率或准确率。");
    node(section,"p",`${extraction.status} · ${en ? "accepted" : "通过"} ${(extraction.accepted_evidence_ids || []).length} / ${Object.keys(extraction.candidate_ledger || {}).length} · ${en ? "rejected" : "拒绝"} ${Object.keys(extraction.rejected || {}).length}`);
    for (const [eid, reasons] of Object.entries(extraction.rejected || {})) {
      const candidate=extraction.candidate_ledger?.[eid];
      const button=node(section,"button",`${eid}: ${reasons.join(", ")}`);button.type="button";
      button.addEventListener("click",()=>onSelect({kind:"quality-diagnostic",id:eid,value:{source_quote:candidate?.supporting_text,claim_text:candidate?.text,evidence_ids:[eid],explanation:reasons.join(", ")}},button));
    }
    node(section,"p",`${extraction.jev?.actual_model || extraction.jev?.requested_model || "jev-1.13.0"} · ${extraction.jev?.decisions?.length || 0} / ${extraction.jev?.planned_question_count ?? "—"} ${en ? "questions completed" : "问题已完成"}`);
    for (const batch of extraction.jev?.batches || []) if (batch.reason_codes?.length) node(section,"p",`${batch.batch_id}: ${batch.reason_codes.join(", ")}`,"notice-error");
    if(extraction.reason_code)node(section,"p",extraction.reason_code,"notice-error");
  }
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
