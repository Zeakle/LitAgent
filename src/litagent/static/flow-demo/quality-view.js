/* Version-bound diagnostics never alter the original report or gate. */
function scoreText(result) {
  if (result?.skipped || result?.score == null) return "—";
  const numeric = Number(result.score);
  return Number.isFinite(numeric) ? numeric.toFixed(3) : "—";
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
  const v2 = ["survey-evaluation-v2", "survey-evaluation-v3"].includes(details.evaluation_version);
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
  node(gate, "p", details.evaluation_version === "survey-evaluation-v3" ? `${options.locale === "en" ? "Evaluation v3 · complete batched assessment" : "评估 v3 · 完整分批核验"}` : v2 ? `${L(details.support_contract === "host-support-passages-v2" ? "current" : details.support_contract === "cited-support-spans-v1" ? "previousSupport" : "historicalV2")} · ${details.evaluation_version}` : options.pipelineVerification && !Object.keys(evaluation || {}).length ? (options.locale === "en" ? "Evaluation was not run; no report scores were produced." : "Evaluation 未运行；尚未产生正文评分。") : L("historical"));
  if (options.revision) node(gate, "p", `r${options.revision}${options.sha ? ` · SHA ${options.sha}` : ""}`, "audit-binding");
  if (v2) node(gate,"p",`${L("support")}: ${details.support_contract || L("unavailable")}`);
  if (options.bodyDiagnostics || details.body_analysis_contract) {
    const section=node(container,"section",null,"quality-section"),en=options.locale === "en";
    node(section,"h3",en ? "Academic body · analytical depth" : "学术正文 · 分析质量");
    const d=options.bodyDiagnostics;
    if(d)node(section,"p",`${en ? "Academic characters / appendix and bibliography" : "学术正文字符／附录与书目"}: ${d.academic_characters} / ${d.non_body_characters} · ${en ? "Repeated units" : "重复单元"}: ${d.repeated_units}`);
    node(section,"p",en ? "Length and counts are diagnostics, not a quality score. Preparation candidates are not verified findings." : "长度与数量仅用于诊断；准备表中的候选材料不等于已核验科学发现。");
    for(const row of details.analysis_check?.sections || []) {
      node(section,"p",`${row.section_key}: ${L(row.adequate ? "passed" : "failed")} · ${row.explanation}`);
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
  const history = options.evaluationHistory;
  if (history && history.version === "version-bound-evaluation-history-v1" && history.source_report_sha256 === options.sha) {
    const section=node(container,"section",null,"quality-section");const en=options.locale === "en";
    node(section,"h3",en ? "Evaluation attempts · this report version" : "本版本 Evaluation 与修复过程");
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
    show(en ? "Initial Evaluation" : "首次 Evaluation",history.initial_evaluation,"",false);
    for(const [i,attempt] of (history.attempts || []).entries())show(`${en ? "Repair candidate" : "修复候选"} ${i+1}`,attempt.candidate_evaluation,attempt.reason_code,attempt.committed);
    show(en ? "Final report Evaluation" : "最终正文 Evaluation",history.final_evaluation,"",true);
  }
  const repairs = options.targetedRepairState;
  if (repairs) {
    const section=node(container,"section",null,"quality-section");const en=options.locale === "en";
    node(section,"h3",en ? "Bounded targeted repairs" : "有限定向修复");
    if(repairs.version !== "shared-targeted-repair-limit-v1" || repairs.limit !== 2 || !Array.isArray(repairs.attempts))node(section,"p",L("unavailable"));
    else {
      node(section,"p",`${en ? "Attempts consumed" : "已消耗次数"}: ${repairs.attempts.length} / 2`);
      node(section,"p",verification ? (en ? "A completed repair call does not establish scientific support; complete Evaluation and any enabled reviews must still pass." : "修复调用完成不代表科学支持；仍需完整 Evaluation 及已启用的复核通过。") : (en ? "A completed repair call does not establish scientific support; full Evaluation and Judge are still required." : "修复调用完成不代表科学支持；仍需完整 Evaluation 和 Judge。"));
      for(const attempt of repairs.attempts)node(section,"p",`${attempt.stage === "native_initial" ? (en ? "Initial native unit" : "初稿单元") : (en ? "Evaluation repair" : "正文定向修复")}: ${L(attempt.status)}`);
    }
  }
  if (options.pipelineVerification) {
    const section=node(container,"section",null,"quality-section");
    node(section,"h3",options.locale === "en" ? "This run · pipeline stages" : "本轮流程阶段");
    for (const [stage, status] of Object.entries(options.pipelineVerification.stages || {})) node(section,"p",`${stage}: ${L(status)}`);
    for (const [service, cost] of Object.entries(options.pipelineVerification.costs || {})) node(section,"p",`${service}: ${cost.tokens == null || !cost.usage_complete ? L("unavailable") : cost.tokens} tokens · ${cost.requests} calls`);
    if(options.pipelineVerification.reason)node(section,"p",String(options.pipelineVerification.reason),"notice-error");
    for (const issue of options.pipelineVerification.writing_diagnostics || []) {
      const detail=node(section,"details",null,"quality-diagnostic");
      node(detail,"summary",`${options.locale === "en" ? "Rejected draft unit" : "被拒绝的初稿单元"}: ${issue.location} · ${issue.subreason || issue.reason_code}`);
      node(detail,"p",issue.candidate_text);
      node(detail,"p",`${options.locale === "en" ? "JSON location" : "JSON位置"}: ${issue.raw_json_path}`);
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
