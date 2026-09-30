const dictionaries = {
  "zh-CN": {
    product: "LitAgent 研究工作台", runs: "运行历史", newResearch: "新建研究", searchRuns: "搜索运行",
    report: "报告", execution: "执行", review: "人工审核", evidence: "证据", quality: "质量", benchmark: "基准",
    benchmarkResults: "Survey 基准结果", benchmarkArchives: "结果归档", refresh: "刷新",
    noBenchmark: "暂无有效基准归档", selectBenchmark: "选择一份归档查看结果", observations: "次运行", benchmarkRuns: "运行记录",
    benchmarkSample: "样本", profileSummary: "Profile 汇总", pairedComparisons: "配对比较",
    noComparisons: "此归档没有配对比较", comparable: "可比", notComparable: "不可比",
    sharedCohorts: "共有样本", evidenceFlow: "证据流向", negativeVerdicts: "负面判断", openReport: "打开报告视图",
    citationAuditExperiment: "引用核验试验", noCitationAudits: "该运行暂无引用核验结果", currentArtifact: "当前报告",
    staleDiagnostic: "过期诊断", auditClaims: "有效问题", protocolCompletion: "协议完成率", uncertainRate: "不确定比例",
    inputTokens: "输入 Tokens", usageComplete: "Usage 完整", auditTrace: "打开核验 Trace", auditDecisions: "核验判断",
    claim: "原子结论", auditTarget: "核验对象", auditDecision: "判断", confidence: "置信度", citationSet: "引用集合",
    uncertain: "不确定", reasonCodes: "原因码", sourcePosition: "原文位置", evidenceScope: "证据范围",
    correctedDiagnostic: "离线修正版诊断，非新 Survey", legacyMetricsWarning: "历史统计协议未校验；原配对差值已隐藏。",
    noRun: "选择一条运行查看结果", noReport: "尚未生成报告", outline: "目录", close: "关闭",
    source: "来源", status: "状态", elapsed: "耗时", tokens: "Tokens", delivery: "交付", language: "报告语言",
    start: "开始运行", cancel: "取消运行", question: "研究问题", humanReview: "人工审核",
    inspect: "检查详情", timeline: "事件时间线", graph: "Agent 执行图", allEvents: "全部事件",
    approve: "批准终稿", revise: "反馈并修订", reject: "拒绝", confirm: "确认论文选择", feedback: "修订反馈",
    export: "导出", citationStyle: "引用格式", numberedCitations: "编号引用", evidenceCitations: "原始证据", emptyEvidence: "当前报告没有可展示的证据", emptyEvents: "尚未记录执行事件",
    operationFailed: "操作失败", offline: "离线模式仅支持回放", queryPlaceholder: "例如：few-shot learning in computer vision",
    runArtifact: "回放", interactive: "交互运行", sample: "示例", trace: "打开 Trace", download: "下载 Artifact",
    submittedSyncFailed: "决定已提交，但状态同步失败。请重新同步。", retrySync: "重新同步",
    executionFlow: "Agent 执行流程", stageOverview: "阶段总览", taskDag: "任务 DAG", callTimeline: "调用时间轴",
    stagePlanning: "问题规划", stageDiscovery: "论文发现", stageEvidence: "证据处理", stageWriting: "撰写与审阅",
    stageEvaluation: "评估与修订", stageHumanReview: "人工审核", stageDelivery: "质量与交付", stageOther: "其他任务",
    taskPlanner: "研究规划", taskSearch: "外部检索", taskRecall: "语料召回", taskDedup: "论文去重",
    taskRelevance: "相关性筛选", taskExtraction: "证据提取", taskGraphAnalysis: "论文关系分析",
    taskSynthesis: "报告撰写", taskReviewer: "报告审阅", taskAdversarial: "对抗撰写与审阅", taskOther: "执行任务",
    papersFound: "发现论文", papersRetained: "保留论文", papersExtracted: "已提取论文", papersAnalyzed: "分析论文",
    evidenceGenerated: "生成证据", evidenceSelected: "选入证据", reviewRounds: "审阅轮次", outputNotRecorded: "产物数量未记录",
    records: "项记录", statusCompleted: "已完成", statusRunning: "运行中", statusQueued: "等待中", statusPending: "未开始",
    statusFailed: "失败", statusSkipped: "已跳过", statusCancelled: "已取消", statusUnknown: "未知", statusAwaitingHuman: "等待人工审核",
    emptyExecution: "尚未记录执行流程", eventCategory: "事件类别", showAll: "显示全部", role: "职责", error: "错误",
    resultSummary: "结果摘要", input: "输入", output: "输出", rawRecord: "原始记录",
    executionView: "执行视图", zoomOut: "缩小", zoomIn: "放大", fitGraph: "适配画布",
    focusSelected: "定位选中节点", toggleFullscreen: "切换全屏", timelineCategory: "时间线类别", secondsShort: "秒",
  },
  en: {
    product: "LitAgent Research Workbench", runs: "Run history", newResearch: "New research", searchRuns: "Search runs",
    report: "Report", execution: "Execution", review: "Human review", evidence: "Evidence", quality: "Quality", benchmark: "Benchmark",
    benchmarkResults: "Survey benchmarks", benchmarkArchives: "Archives", refresh: "Refresh",
    noBenchmark: "No valid benchmark archives", selectBenchmark: "Select an archive to inspect", observations: "observations", benchmarkRuns: "Run records",
    benchmarkSample: "Sample", profileSummary: "Profile summary", pairedComparisons: "Paired comparisons",
    noComparisons: "No paired comparisons in this archive", comparable: "Comparable", notComparable: "Not comparable",
    sharedCohorts: "Shared cohorts", evidenceFlow: "Evidence flow", negativeVerdicts: "Negative verdicts", openReport: "Open report view",
    citationAuditExperiment: "Citation audit experiment", noCitationAudits: "No citation audit is attached to this run", currentArtifact: "Current artifact",
    staleDiagnostic: "Stale diagnostic", auditClaims: "Valid questions", protocolCompletion: "Protocol completion", uncertainRate: "Uncertain rate",
    inputTokens: "Input tokens", usageComplete: "Usage complete", auditTrace: "Open audit trace", auditDecisions: "Audit decisions",
    claim: "Atomic claim", auditTarget: "Audit target", auditDecision: "Decision", confidence: "Confidence", citationSet: "Citation set",
    uncertain: "Uncertain", reasonCodes: "Reason codes", sourcePosition: "Source position", evidenceScope: "Evidence scope",
    correctedDiagnostic: "Offline corrected diagnostic, not a new Survey", legacyMetricsWarning: "Historical metric protocol is unverified; original paired deltas are hidden.",
    noRun: "Select a run to inspect", noReport: "No report has been generated", outline: "Outline", close: "Close",
    source: "Source", status: "Status", elapsed: "Elapsed", tokens: "Tokens", delivery: "Delivery", language: "Report language",
    start: "Start run", cancel: "Cancel run", question: "Research question", humanReview: "Human review",
    inspect: "Inspect details", timeline: "Event timeline", graph: "Agent execution graph", allEvents: "All events",
    approve: "Approve", revise: "Revise from feedback", reject: "Reject", confirm: "Confirm paper selection", feedback: "Revision feedback",
    export: "Export", citationStyle: "Citation style", numberedCitations: "Numbered", evidenceCitations: "Evidence IDs", emptyEvidence: "No evidence is available for this report", emptyEvents: "No execution events recorded",
    operationFailed: "Operation failed", offline: "Offline mode only supports replay", queryPlaceholder: "e.g. few-shot learning in computer vision",
    runArtifact: "Replay", interactive: "Interactive", sample: "Sample", trace: "Open trace", download: "Download artifact",
    submittedSyncFailed: "The decision was submitted, but state refresh failed. Sync again.", retrySync: "Sync again",
    executionFlow: "Agent execution flow", stageOverview: "Stage overview", taskDag: "Task DAG", callTimeline: "Call timeline",
    stagePlanning: "Planning", stageDiscovery: "Paper discovery", stageEvidence: "Evidence processing", stageWriting: "Writing and review",
    stageEvaluation: "Evaluation and revision", stageHumanReview: "Human review", stageDelivery: "Quality and delivery", stageOther: "Other tasks",
    taskPlanner: "Research planning", taskSearch: "External search", taskRecall: "Corpus recall", taskDedup: "Paper deduplication",
    taskRelevance: "Relevance screening", taskExtraction: "Evidence extraction", taskGraphAnalysis: "Paper graph analysis",
    taskSynthesis: "Report synthesis", taskReviewer: "Report review", taskAdversarial: "Adversarial synthesis and review", taskOther: "Execution task",
    papersFound: "Papers found", papersRetained: "Papers retained", papersExtracted: "Papers extracted", papersAnalyzed: "Papers analyzed",
    evidenceGenerated: "Evidence generated", evidenceSelected: "Evidence selected", reviewRounds: "Review rounds", outputNotRecorded: "Output count not recorded",
    records: "records", statusCompleted: "Completed", statusRunning: "Running", statusQueued: "Queued", statusPending: "Pending",
    statusFailed: "Failed", statusSkipped: "Skipped", statusCancelled: "Cancelled", statusUnknown: "Unknown", statusAwaitingHuman: "Awaiting human review",
    emptyExecution: "No execution flow has been recorded", eventCategory: "Event category", showAll: "Show all", role: "Role", error: "Error",
    resultSummary: "Result summary", input: "Input", output: "Output", rawRecord: "Raw record",
    executionView: "Execution view", zoomOut: "Zoom out", zoomIn: "Zoom in", fitGraph: "Fit graph",
    focusSelected: "Focus selected node", toggleFullscreen: "Toggle full screen", timelineCategory: "Timeline category", secondsShort: "s",
  },
};

export function translator(locale) {
  const selected = dictionaries[locale] || dictionaries["zh-CN"];
  return (key) => selected[key] || key;
}

export function applyTranslations(root, t) {
  for (const element of root.querySelectorAll("[data-i18n]")) element.textContent = t(element.dataset.i18n);
  for (const element of root.querySelectorAll("[data-i18n-placeholder]")) element.placeholder = t(element.dataset.i18nPlaceholder);
  for (const element of root.querySelectorAll("[data-i18n-label]")) {
    const label = t(element.dataset.i18nLabel);
    element.setAttribute("aria-label", label);
    if (element.hasAttribute("title")) element.setAttribute("title", label);
  }
}
