/* Audits bind raw report versions, independently of citation-style projections. */
export function bindAutomaticAuditPreferences({ capabilities, t }) {
  const $ = selector => document.querySelector(selector);
  const mode = $("#auto-audit-mode");
  const budget = $("#auto-audit-max-versions");
  function render() {
    const caps = capabilities() || {};
    for (const option of mode.options) {
      option.disabled = option.value === "judge" ? !caps.judge_available
        : option.value === "judge_jev" ? !(caps.judge_available && caps.jev_available) : false;
    }
    if ([...mode.options].find(option => option.value === mode.value)?.disabled) mode.value = "off";
    budget.disabled = mode.value === "off";
    const limits = caps.limits || {};
    $("#auto-audit-budget-note").textContent = mode.value === "off" ? t("autoAuditOffHelp") : `${t("autoAuditBudgetHelp")} ${budget.value} · ${t("autoAuditPerVersion")} Judge ${limits.judge_calls ?? 2} × ${limits.judge_max_output_tokens_per_call ?? 16384} output tokens / ${limits.judge_timeout_seconds ?? 180}s${mode.value === "judge_jev" ? ` · Jev ${limits.jev?.max_requests ?? 20} requests / ${limits.jev?.max_input_tokens ?? 200000} input tokens` : ""}`;
    const reasons = [];
    if (!caps.judge_available) reasons.push(`${t("autoAuditJudgeUnavailable")}: ${caps.judge_reason || t("autoAuditNotConfigured")}`);
    else if (!caps.jev_available) reasons.push(`${t("autoAuditJevUnavailable")}: ${caps.jev_reason || t("autoAuditNotConfigured")}`);
    $("#auto-audit-capabilities").textContent = reasons.join(" · ");
    $("#auto-audit-capabilities").hidden = !reasons.length;
  }
  function options() {
    const selected = mode.value;
    const caps = capabilities() || {};
    if (!["off", "judge", "judge_jev"].includes(selected)) throw new Error(t("autoAuditInvalidSelection"));
    if (selected !== "off" && (!caps.judge_available || (selected === "judge_jev" && !caps.jev_available))) throw new Error(t("autoAuditUnavailableSelection"));
    const maxVersions = selected === "off" ? 1 : Number(budget.value);
    if (!Number.isInteger(maxVersions) || maxVersions < 1 || maxVersions > 3) throw new Error(t("autoAuditInvalidBudget"));
    return { auto_audit_mode: selected, auto_audit_max_versions: maxVersions };
  }
  mode.addEventListener("change", render); budget.addEventListener("change", render);
  render();
  return { render, options };
}

export function bindAuditControls({ store, api, setNotice, onEvidence, onSource, onSnapshot = () => {} }) {
  const $ = selector => document.querySelector(selector);
  const labels = {
    title: ["独立核验", "Independent audit"], start: ["核验当前版本", "Audit this version"],
    empty: ["此版本尚无独立核验结果。", "No independent audit for this version."],
    scope: ["Judge 核验事实、矛盾、结构和方法；Jev 核验引用支持关系。结果不改变评分或批准。", "Judge checks facts, contradictions, structure and methodology; Jev checks citation support. Results preserve scores and approval."],
    begin: ["开始核验", "Start audit"], cancel: ["取消核验", "Cancel audit"], close: ["关闭", "Close"],
    jev: ["同时执行 Jev 引用核验", "Include Jev citation audit"], retry: ["创建新的付费尝试（不复用结果）", "Create a new paid attempt (do not reuse results)"],
    topics: ["未提供人工主题标准", "No human topic standard provided"], methods: ["检索方法核验", "Runtime-method audit"],
    structure: ["结构检查", "Structure check"], facts: ["科学断言支持", "Scientific claim support"],
    costs: ["调用与 tokens", "Calls and tokens"], usage: ["usage 不完整，无法确定费用", "Incomplete usage; cost is unknown"],
    source: ["定位原文", "Locate source"], evidence: ["查看证据", "Open evidence"],
    unknown: ["暂无结果", "No result"], confidence: ["置信度不是准确率；没有人工金标准。", "Confidence is not accuracy; no human gold labels are supplied."],
    queued: ["排队中", "Queued"], running: ["核验中", "Running"], completed: ["已完成", "Completed"],
    partial: ["部分完成", "Partial"], failed: ["失败", "Failed"], cancelled: ["已取消", "Cancelled"],
    interrupted: ["已中断，不自动重发", "Interrupted; no automatic retry"], blocked: ["依赖门禁阻塞", "Dependency blocked"],
    skipped: ["无适用断言，已跳过", "No applicable claims; skipped"],
    not_requested: ["未请求", "Not requested"], unverifiable: ["无法核验", "Unverifiable"],
    verified: ["记录支持", "Verified against records"], unsupported: ["未获支持", "Unsupported"],
    passed: ["通过", "Passed"], not_applicable: ["不适用", "Not applicable"],
    noComparison: ["生成模型记录缺失，无法比较", "Generation model record unavailable"],
    unsupportedRate: ["未获支持比例", "Unsupported rate"], contradictionRate: ["矛盾比例", "Contradiction rate"],
    completion: ["完成率", "Completion"], comparable: ["可比较断言", "Comparable claims"], agreement: ["一致率", "Agreement"],
    localErrors: ["本地错误", "Local errors"], uncertainCount: ["低置信度问题", "Uncertain questions"], claims: ["已核验断言", "Audited claims"],
    calls: ["次调用", "calls"], input: ["输入", "input"], output: ["输出", "output"],
    limits: ["调用上限", "Call limits"],
    uncertain: ["低置信度", "Low confidence"], local_error: ["本地证据错误", "Local evidence error"],
    supported: ["支持", "Supported"], contradicted: ["矛盾", "Contradicted"], insufficient: ["证据不足", "Insufficient evidence"],
    automatic: ["Evaluation 通过后的自动核验", "Automatic audit after Evaluation passes"],
    autoSkipped: ["未发起自动核验", "Automatic audit was not submitted"],
    autoFailed: ["自动核验调用失败；不是报告质量判定", "Automatic audit failed; this is not a report-quality verdict"],
    autoBudget: ["自动核验版本预算", "Automatic version budget"],
    autoPending: ["已预留预算，等待提交", "Budget reserved; awaiting submission"],
    autoNoRetry: ["失败或重启中断不会自动重发；可使用手动入口创建新尝试。", "Failure or restart interruption does not retry automatically; a new attempt can be created manually."],
    automatic_quality_not_passed: ["Evaluation 尚未通过", "Evaluation has not passed"],
    automatic_report_partial: ["报告仅部分完成", "The report is partially complete"],
    automatic_report_rejected: ["报告已被拒绝", "The report was rejected"],
    automatic_report_cancelled: ["报告运行已取消", "The report run was cancelled"],
    automatic_version_budget_exhausted: ["自动核验版本预算已用尽", "The automatic version budget is exhausted"],
    report_sha_mismatch: ["报告内容指纹不匹配", "Report SHA mismatch"],
    automatic_submit_failed: ["提交核验任务失败", "Failed to submit the audit task"],
    automatic_submission_cancelled: ["自动提交已取消", "Automatic submission was cancelled"],
    automatic_reservation_interrupted_no_retry: ["重启中断了预算预留，未自动重发", "A restart interrupted the reservation; no automatic retry"],
    automatic_bound_audit_missing: ["关联核验任务缺失", "The bound audit is missing"],
    automatic_bound_audit_mismatch: ["关联核验任务与版本不匹配", "The bound audit does not match this version"],
    missing_key: ["服务端缺少密钥", "Server key missing"],
    audit_offline_or_disabled: ["核验服务处于离线模式或已禁用", "Audit service is offline or disabled"],
    jev_disabled: ["Jev 已禁用", "Jev is disabled"],
  };
  const L = key => labels[key]?.[store.get().locale === "zh-CN" ? 0 : 1] || key;
  const reasonLabel = value => {
    const [code, ...details] = value.split(":");
    return `${L(code)}${details.length ? `: ${details.join(":")}` : ""}`;
  };
  let key = ""; let epoch = 0; let timer = null; let data = null; let selected = null; let target = null; let modalTarget = null;
  function currentTarget() {
    const view = store.get().view;
    if (!view?.report?.survey) return null;
    return { runId: view.identity.runId, revision: Number(view.displayRevision || view.versions?.[0]?.revision || 1), source: view.identity.source, sha: view.reportSha || null };
  }
  const identity = value => value ? `${value.runId}:${value.revision}` : "";
  const path = value => `/survey/${encodeURIComponent(value.runId)}/versions/${value.revision}/audits`;
  function text(parent, tag, content, className) {
    const item = document.createElement(tag); item.textContent = content; if (className) item.className = className; parent.append(item); return item;
  }
  function sourceButton(parent, claim) {
    if (!claim) return;
    const button = text(parent, "button", `${L("source")} · ${claim.source_start}–${claim.source_end}`);
    button.type = "button";
    button.addEventListener("click", () => {
      if (onSource) { onSource(claim); return; }
      const normalize = value => value.replace(/\[E:[^\]]+\]/g, "").replace(/\[\d+(?:[,–-]\d+)*\]/g, "").replace(/[*_`]/g, "").replace(/\s+/g, " ").trim();
      const needle = normalize(claim.source_quote);
      const report = $("#report-document");
      report.querySelectorAll(".audit-source-highlight").forEach(item => item.classList.remove("audit-source-highlight"));
      const match = [...report.querySelectorAll("p,li,td,blockquote")].find(item => {
        const clean = item.cloneNode(true); clean.querySelectorAll(".citation-button").forEach(citation => citation.remove());
        return normalize(clean.textContent).includes(needle);
      });
      (match || report).scrollIntoView({ block: "center", behavior: "smooth" });
      match?.classList.add("audit-source-highlight");
    });
    text(parent, "blockquote", claim.source_quote);
    for (const id of claim.evidence_ids || []) {
      const evidence = text(parent, "button", `${L("evidence")} ${id}`); evidence.type = "button";
      evidence.addEventListener("click", () => onEvidence(id, evidence));
    }
  }
  function render() {
    if (data?.audits?.length && !data.audits.some(item => item.audit_id === selected)) selected = data.audits[0].audit_id;
    onSnapshot({ target, data, selected, select: id => { if (data?.audits?.some(item => item.audit_id === id)) {selected = id; render();} } });
    const panel = $("#audit-panel"); panel.hidden = !target;
    $("#audit-title").textContent = L("title"); $("#audit-start").textContent = L("start");
    $("#audit-scope").textContent = L("scope");
    const caps = data?.capabilities;
    $("#audit-start").disabled = !caps?.judge_available || target?.source === "sample";
    $("#audit-capabilities").textContent = caps ? `Judge: ${caps.judge_model}${caps.self_judged ? " · self_judged" : caps.self_judged == null ? ` · ${L("noComparison")}` : ""} · ${caps.judge_reason || ""} · Jev: ${caps.jev_model} ${caps.jev_reason || ""}` : "";
    const results = $("#audit-results"); results.replaceChildren();
    if (data?.automatic) {
      const automatic = data.automatic;
      text(results, "h3", L("automatic"));
      text(results, "p", `r${automatic.source_report_revision} · ${automatic.mode} · ${L("autoBudget")}: ${automatic.max_versions}`);
      text(results, "p", automatic.status === "skipped" ? L("autoSkipped") : automatic.status === "failed" ? L("autoFailed") : automatic.status === "reserved" ? L("autoPending") : L(automatic.status));
      if (automatic.reason_codes?.length) text(results, "p", automatic.reason_codes.map(reasonLabel).join("; "));
      if (["failed", "interrupted"].includes(automatic.status)) text(results, "p", L("autoNoRetry"));
    }
    if (!data?.audits?.length) { text(results, "p", L("empty")); return; }
    const select = document.createElement("select"); select.setAttribute("aria-label", L("title"));
    for (const result of data.audits) {
      const option = text(select, "option", `r${result.source_report_revision} · ${L(result.status)} · ${result.audit_id.slice(-8)}`); option.value = result.audit_id;
    }
    if (!data.audits.some(item => item.audit_id === selected)) selected = data.audits[0].audit_id;
    select.value = selected; select.addEventListener("change", () => { selected = select.value; render(); }); results.append(select);
    const audit = data.audits.find(item => item.audit_id === selected);
    text(results, "p", `r${audit.source_report_revision} · SHA ${audit.source_report_sha256} · ${audit.judge_model}${audit.self_judged ? " · self_judged" : ""}`, "audit-binding");
    if (["queued", "running"].includes(audit.status)) {
      const cancel = text(results, "button", L("cancel")); cancel.type = "button";
      cancel.addEventListener("click", async () => {
        const captured = { ...target }; cancel.disabled = true;
        try { await api.json(`${path(captured)}/${audit.audit_id}/cancel`, { method: "POST", mutate: true }); if (identity(captured) === key) await load(); }
        catch (error) { if (identity(captured) === key) setNotice(error.message, "error"); }
      });
    }
    text(results, "h3", `Judge · ${L(audit.judge?.status || audit.status)} · ${(audit.judge_actual_models || []).join(", ")}`);
    text(results, "p", audit.judge?.topic_coverage == null ? L("topics") : `Topic coverage: ${audit.judge.topic_coverage}`);
    text(results, "p", `${L("facts")}: ${audit.judge?.faithfulness ?? "—"} · ${L("unsupportedRate")}: ${audit.judge?.unsupported_claim_rate ?? "—"} · ${L("contradictionRate")}: ${audit.judge?.contradiction_rate ?? "—"}`);
    text(results, "p", `${L("structure")}: ${L(audit.structure?.status)} ${(audit.structure?.issues || []).join("; ")}`);
    text(results, "h3", `${L("methods")} · ${L(audit.method_status)}`);
    for (const assessment of audit.method_assessments || []) {
      const details = document.createElement("details"); text(details, "summary", `${L(assessment.status)} · ${assessment.explanation}`);
      sourceButton(details, audit.method_inventory.find(item => item.claim_id === assessment.claim_id)); text(details, "p", assessment.record_paths.join(", ")); results.append(details);
    }
    const claims = audit.judge?.inventory_snapshot?.claims || [];
    for (const assessment of audit.judge?.claim_assessments || []) {
      const details = document.createElement("details"); text(details, "summary", `${L(assessment.supported ? "supported" : "unsupported")} · ${assessment.claim_text}`);
      text(details, "p", assessment.explanation); sourceButton(details, claims.find(item => item.claim_id === assessment.claim_id)); results.append(details);
    }
    for (const contradiction of audit.judge?.contradictions || []) text(results, "p", `${L("contradicted")}: ${contradiction.claim_a} / ${contradiction.claim_b} · ${contradiction.explanation}`);
    text(results, "h3", `Jev · ${L(audit.jev?.status || "unknown")}`); text(results, "p", L("confidence"));
    const summary = audit.jev?.summary;
    if (summary) text(results, "p", `${L("claims")}: ${summary.audited_claims}/${summary.planned_claims} · ${L("completion")}: ${summary.protocol_completion_rate ?? "—"} · ${L("comparable")}: ${summary.comparable_claims} · ${L("agreement")}: ${summary.agreement ?? "—"} · ${L("localErrors")}: ${summary.local_error_questions} · ${L("uncertainCount")}: ${summary.uncertain_questions}`);
    for (const decision of audit.jev?.decisions || []) {
      const details = document.createElement("details"); text(details, "summary", `${decision.claim_id} · ${L(decision.choice)} · ${decision.confidence}${decision.uncertain ? ` · ${L("uncertain")}` : ""}`); sourceButton(details, claims.find(item => item.claim_id === decision.claim_id)); results.append(details);
    }
    for (const input of audit.jev?.inputs || []) if (input.status === "local_error") text(results, "p", `${input.claim_id} · ${L("local_error")} · ${(input.reason_codes || []).join(", ")}`);
    text(results, "h3", L("costs"));
    for (const [name, usage] of [["Judge", audit.judge_usage], ["Jev", audit.jev?.usage]]) if (usage) text(results, "p", `${name}: ${usage.call_count ?? "—"} ${L("calls")} · ${usage.prompt_tokens ?? usage.input_tokens ?? "—"} ${L("input")} / ${usage.completion_tokens ?? usage.output_tokens ?? "—"} ${L("output")}${usage.usage_complete === true ? "" : ` · ${L("usage")}`}`);
    text(results, "p", [...(audit.reason_codes || []), ...(audit.jev?.reason_codes || [])].join("; "));
    if (audit.trace?.trace_url) { const link = text(results, "a", "LangFuse trace"); link.href = audit.trace.trace_url; link.target = "_blank"; link.rel = "noopener noreferrer"; }
  }
  async function load() {
    const captured = target; if (!captured) return;
    const generation = ++epoch; clearTimeout(timer);
    try {
      const response = await api.json(path(captured));
      if (generation !== epoch || identity(captured) !== key) return;
      if (response.source_report_revision != null && response.source_report_revision !== captured.revision) throw new Error("audit_version_mismatch");
      if (captured.sha && captured.sha !== response.source_report_sha256) throw new Error("audit_input_binding_mismatch");
      if ((response.audits || []).some(item => item.source_report_revision !== captured.revision || item.source_report_sha256 !== response.source_report_sha256)) throw new Error("audit_input_binding_mismatch");
      data = response; render();
      if (response.audits.some(item => ["queued", "running"].includes(item.status)) || response.automatic?.status === "reserved") timer = setTimeout(load, 3000);
    } catch (error) { if (generation === epoch && identity(captured) === key) { data = null; selected = null; render(); onSnapshot({target, data: null, error: error.message}); $("#audit-capabilities").textContent = error.message; } }
  }
  function sync() {
    const next = currentTarget(); const nextKey = identity(next);
    if (nextKey !== key || next?.sha !== target?.sha) {
      epoch += 1; clearTimeout(timer); key = nextKey; target = next; data = null; selected = null;
      $("#audit-dialog").close(); modalTarget = null; render(); if (next) load();
    } else { target = next; render(); }
  }
  $("#audit-start").addEventListener("click", () => {
    if (!target || !data?.capabilities?.judge_available) return;
    modalTarget = { ...target, sha: data.source_report_sha256 };
    $("#audit-dialog-title").textContent = L("title"); $("#audit-dialog-target").textContent = `r${target.revision} · SHA ${modalTarget.sha} · Judge ${data.capabilities.judge_model}${data.capabilities.self_judged ? " · self_judged" : ""} · ${L("limits")}: Judge 2 × 16384 tokens / 180s · Jev 20 × requests / 200000 tokens`;
    $("#audit-jev").checked = false; $("#audit-jev").disabled = !data.capabilities.jev_available;
    $("#audit-jev-label").textContent = L("jev"); $("#audit-jev-reason").textContent = data.capabilities.jev_reason || "";
    $("#audit-new-attempt").checked = false; $("#audit-new-label").textContent = L("retry");
    $("#audit-submit").textContent = L("begin"); $("#audit-close").textContent = L("close"); $("#audit-submit").disabled = false;
    $("#audit-dialog").showModal();
  });
  $("#audit-close").addEventListener("click", () => $("#audit-dialog").close());
  $("#audit-submit").addEventListener("click", async () => {
    const captured = modalTarget; if (!captured || identity(captured) !== key) return;
    $("#audit-submit").disabled = true;
    try {
      const response = await api.json(path(captured), { method: "POST", mutate: true, body: { report_sha256: captured.sha, include_jev: $("#audit-jev").checked, new_attempt: $("#audit-new-attempt").checked } });
      if (identity(captured) !== key) return;
      selected = response.audit_id; $("#audit-dialog").close(); await load();
    } catch (error) { if (identity(captured) === key) { setNotice(error.message, "error"); $("#audit-submit").disabled = false; } }
  });
  store.subscribe(sync); sync();
  return { refresh: load };
}
