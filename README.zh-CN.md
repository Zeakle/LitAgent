# LitAgent

[![English](https://img.shields.io/badge/lang-English-1769aa.svg)](./README.md)
[![简体中文](https://img.shields.io/badge/lang-简体中文-c5221f.svg)](./README.zh-CN.md)

LitAgent 是一个基于动态多 Agent 编排、对抗审查、混合检索和明确质量交付策略的
证据约束型学术文献综述系统。

系统会把研究问题拆解成运行时 DAG。外部搜索和本地召回共同构建带来源定位的证据账本，
Synthesis 生成的草稿需要经过 Adversarial Reviewer 挑战，最终报告必须通过 Evaluation、
Quality 和 Delivery 策略后才能标记为可发布。每次执行都可以通过本地 RunArtifact 和可选
LangFuse trace 进行回放。

## 项目状态

| 模块 | 状态 | 仓库证据 |
| --- | --- | --- |
| 动态 DAG 和对抗式 Survey 流程 | 已实现 | [`runner.py`](./src/litagent/runner.py)、[`test_phase14_4_e2e.py`](./tests/test_phase14_4_e2e.py) |
| 持久化 Paper Corpus 和 PDF ingestion | 已实现 | [`rag/`](./src/litagent/rag)、[`test_phase14_1.py`](./tests/test_phase14_1.py) |
| Retrieval benchmark | 已完成 | [`RESULTS.md`](./benchmarks/rag/RESULTS.md) |
| 端到端 Survey benchmark | 诊断性 pilot 已完成，尚未形成正式 baseline | [`2026-08-25 pilot`](./benchmarks/survey/results/pilot-2026-08-25.md)、[`2026-09-08 修复后 pilot`](./benchmarks/survey/results/post-repair-pilot-2026-09-08.md)、[`test_phase14_5.py`](./tests/test_phase14_5.py) |
| 可复现流程回放 | 已实现 | [`demo.py`](./src/litagent/demo.py)、[`test_phase14_6.py`](./tests/test_phase14_6.py) |
| 交互式研究工作台 | 离线实现完成；真实双语验收待完成 | [`interactive_service.py`](./src/litagent/interactive_service.py)、[`test_interactive_runner.py`](./tests/test_interactive_runner.py) |

公开能力声明与证据的完整对应关系位于
[`docs/design-truth.md`](./docs/design-truth.md)。

## 快速开始

### 离线回放与研究工作台

离线模式不需要 API key、Docker、模型下载或网络访问。报告优先的研究工作台内置
`ready`、`blocked` 和 `partial` 三种交付状态样例；同一页面还能查看真实运行的 DAG、
事件、节点输入输出、证据、质量与交付判定、报告版本及导出结果。

```powershell
pip install -e ".[api]"
litagent demo
```

浏览器会自动打开 `http://127.0.0.1:8000/flow-demo`。只启动服务而不打开浏览器：

```powershell
litagent demo --no-browser
```

内置样例只是确定性的契约 fixture，不能作为模型质量或 benchmark 证据。

真实运行可以输入自定义研究问题，将报告语言指定为 `English` 或 `简体中文`，并在
论文选择和终稿审核两个边界暂停。人工批准不能绕过自动质量门和交付门。交互控制状态
保存在本地 SQLite，大型 checkpoint 与完整回放 artifact 继续以 JSON 写入
`artifacts/`。

新建弹窗默认使用 `academic_v2`“学术综述 · 检索记录附录”，保留两种 v1 结构，默认引用样式为 IEEE。正文采用七章学术结构，参考文献之后显示默认折叠的附录 A；证据页读取同一冻结版本的检索与筛选记录。Markdown/PDF/Word 完整保留附录，核验仍覆盖整个报告。结构修改需要
新建运行；已有报告可切换 IEEE、APA 7、GB/T 7714-2025 顺序编码或著者—出版年样式，
保留简单编号和原始证据。样式切换不调用模型、不改变源 SHA、评分和批准状态。
Markdown/PDF/Word 使用同一引用投影；BibTeX/RIS/CSL JSON 只导出所选版本实际引用的论文。
元数据缺失、未分段的完整姓名会提示，不自动补齐，也不代表完整投稿格式认证。

报告页提供绑定版本/SHA 的“独立核验”：默认 Judge，可选 Jev。结果独立保存，不改变
正文、评分、批准或交付；没有人工主题标准时不提供主题覆盖分数。服务端通过
`report_audit` 配置模型，Judge 默认沿用生成凭证；更换 host 必须显式配置独立凭证变量。
Jev 使用 `jev-1.13.0`，需 `TYPESAFE_API_KEY`，缺 key 时界面禁用。终态结果（含中断）
默认复用，显式“新的尝试”才重跑；重启不自动重发请求。接入已通过离线验收，真实
核验另行验收；Jev confidence 不代表准确率。

新建研究可选择 Evaluation 通过后的自动核验：关闭（默认）、Judge、Judge + Jev。
仅在最终报告版本保存后触发；人工终稿审核前可查看结果。整个运行默认最多自动核验
1 个版本，可选 1–3；失败、取消及重启中断也占名额，且不会自动重发。修订版本需
重新通过 Evaluation 并有剩余名额；历史报告不会补跑。手动核验入口继续保留。

新运行使用 `survey-evaluation-v2`：最多两次模型调用，逐原文单元提取断言并核验分类，
科学断言仅使用实际引用的证据；运行方法与可信记录核对，报告组织与正文核对。
材料上限为 120 条／80,000 字符；超限、缺材料或响应不完整保持未核验。
faithfulness 门槛仍为 0.8，严重实体、数值或实验条件归属错误单独阻止交付。
`academic_v2` 的附录和旧方法透明结构的完整查询、时间、来源结果及论文／证据计数由程序组装，在审阅、评估和冻结前完成，修订后重新核对。

质量页按版本和 SHA 展示实际阈值、支持／不支持／未核验数量、结构方法检查与原文定位。
Judge/Jev 共用报告页的加载与尝试选择，独立保存，不参与交付门禁或混合总分。
历史记录明确标识旧口径，缺少统计时显示不可用。2026-10-03 离线回归 1327 passed，
桌面／390px、中英文及键盘验收通过；真实通过率提升尚待同输入、同模型、同预算、
同评估版本的受控对照，不能用新口径直接对比历史 0.62。

固定材料预检进一步发现原始摘要重复导致上下文截断，现按同论文／同原文共享支持，
保留独立证据绑定并拒绝不完整材料。审阅证据层为 16,000 tokens，总上下文仍为
32,768，运行预算不变。2026-10-04 首组真实对照使用 5 次请求／62,798 tokens；
抽取响应的别名和原文引用不符合契约，结果保持未核验，第二组未启动。已明确裸别名、
完整引用摘录及宿主允许清单，补齐冻结来源的归档定位与完整评估停止门禁；
本轮离线回归 1354 passed，19 项新测试覆盖真实失败形态与归档消费链；
随后 pair-002 首组实际使用7次请求／90,422 tokens，别名契约已满足，
但一条非连续摘录仍使核验停止；第二组未发，不宣称质量增益，历史失败结果不修改。
随后改为宿主连续片段 ID，模型选择已有片段；共享材料保留逐断言引用归属。
该阶段离线回归1376 passed，预算超限继续未核验；该轮付费0，真实质量效果待验证。

2026-10-07 最新离线回归为1618 passed。`host-support-passages-v2` 由模型选择宿主原文片段ID，
程序还原文本和位置，并检查实际引用归属、来源明确的缩写、数值与所需支持方面。
保守关系检查在缺依据时保持未核验；Reviewer 区分已有引用与语义支持，只纠正反馈、不提升评分。
历史报告和失败尝试保持不变。run-004 的独立协议／语义验收失败；run-005 仅完成预检，
没有新增付费调用。离线通过不代表真实通过率已提高。

学术样式需要 PATH 中可用的本地 `node`。处理器、CSL 模板、locale、来源哈希及许可随
[`src/litagent/csl/`](./src/litagent/csl/) 分发，运行时不下载模板。Word 为格式化文本，
不含 Zotero 动态字段；查看中文 Word 需安装中文字体（macOS 使用 Arial Unicode MS，
Windows 使用 Microsoft YaHei）。方法透明结构只披露实际运行记录，不宣称完整 PRISMA 合规。

![LitAgent 流程回放桌面视图](./docs/images/flow-demo-desktop.png)

### 真实 Survey

将 `.env.example` 复制为 `.env`，配置 OpenAI-compatible LLM key，并启动项目基础设施：

```powershell
pip install -e ".[dev]"
docker compose up -d
litagent demo --live
```

LangFuse 是可选组件。配置完成后，回放页面会显示当前运行的 trace 链接。本地 artifact
保存在 `artifacts/runs/`，其中可能包含 prompt、论文片段和模型输出，因此不会进入 Git。

工作台目前是本地、单进程控制面，不包含身份认证、租户隔离或分布式任务队列；补齐这些
能力前不应绑定到公网。

## 系统概览

```mermaid
flowchart LR
    Q["研究问题"] --> P["Planner 任务规划"]
    P --> D["TaskGraph + Scheduler"]
    D --> S["外部搜索"]
    D --> R["本地 Corpus 召回"]
    S --> G["去重 + 相关性门"]
    R --> G
    G --> E["信息提取 + Evidence Ledger"]
    E --> Y["Synthesis"]
    Y <--> V["Adversarial Reviewer"]
    V --> A["Evaluation + Evidence Rewrite"]
    A --> L["Quality + Delivery"]
    L --> O["Report + RunArtifact + LangFuse"]
```

运行时采用全异步实现。Search provider 和本地 Recall 可以并行执行，完成相关性过滤后，
Extraction 与 Graph Analysis 也可以并行。出现失败、取消或超时时，Scheduler 会把所有
任务收敛到明确终态，避免 artifact 把未完成任务表现为成功。

公开的数据流、信任边界、持久化和降级契约见
[`docs/system-overview.md`](./docs/system-overview.md)。

## 从问题到交付

1. **Planner**：把研究问题拆成规范化 sub-query 和依赖 DAG。
2. **Search 与 Recall**：并行查询外部学术来源和持久化本地 Corpus。
3. **Dedup 与 Relevance Gate**：合并论文身份、排序候选并拒绝弱相关结果。
4. **Extraction 与 Graph Analysis**：构建 Evidence、Claim、locator 和论文结构。
5. **Synthesis 与 Adversarial Review**：初稿和每轮修改共享同一份 run-scoped evidence selection。
6. **Evaluation 与 Rewrite**：评估引用、内部一致性和 faithfulness；修复候选只有通过校验和复评才会提交。
7. **Quality 与 Delivery**：根据执行完整性和内容质量输出 `ready`、`needs_review`、`blocked` 或 `partial`。
8. **Observability**：本地保存完整 artifact，可选向 LangFuse 上传递归脱敏后的事件投影。

## 已验证能力

| 能力 | 实现边界 | 验证证据 |
| --- | --- | --- |
| 动态 DAG 调度 | 依赖调度、重试、超时、取消和下游 skipped | [`test_orchestrator.py`](./tests/test_orchestrator.py)、[`test_phase14_4_e2e.py`](./tests/test_phase14_4_e2e.py) |
| 混合论文检索 | Named dense vector、BM25 sparse、RRF、可选 CrossEncoder | [`retriever.py`](./src/litagent/rag/retriever.py)、[`RESULTS.md`](./benchmarks/rag/RESULTS.md) |
| Corpus ingestion | Manifest、本地或 allowlisted arXiv PDF、质量门、section-aware chunking | [`ingest.py`](./src/litagent/rag/ingest.py)、[`test_phase14_1.py`](./tests/test_phase14_1.py) |
| 增量索引 | 稳定 point ID、embedding/payload hash、stale delete、CorpusState | [`corpus.py`](./src/litagent/rag/corpus.py)、[`test_phase14_3.py`](./tests/test_phase14_3.py) |
| Evidence 约束生成 | Synthesis 与 Reviewer 共享不可变的 Evidence selection | [`evidence.py`](./src/litagent/evidence.py)、[`test_phase13_9.py`](./tests/test_phase13_9.py) |
| 事务式 Evidence repair | 候选校验、复评以及 survey/evaluation/quality 原子提交 | [`runner.py`](./src/litagent/runner.py)、[`test_phase13_9.py`](./tests/test_phase13_9.py) |
| Trace 与回放 | 完整本地 artifact、SSE、LangFuse 脱敏 observation | [`recorder.py`](./src/litagent/observability/recorder.py)、[`test_run_recorder.py`](./tests/test_run_recorder.py) |
| Tool 与 MCP 控制 | 名称/类别 allowlist、JSON Schema、capability mapping、写操作 fail-closed | [`executor.py`](./src/litagent/tools/executor.py)、[`test_tools.py`](./tests/test_tools.py)、[`test_mcp.py`](./tests/test_mcp.py) |

## Retrieval Benchmark

Phase 14.3 使用 25 篇论文、15 条经过来源核对的 query，每条 query 重复运行三次。以下数据
来自仓库内冻结结果，不是估算值。

| Profile | Recall@5 | MRR@10 | nDCG@10 | p50 latency |
| --- | ---: | ---: | ---: | ---: |
| MiniLM + RRF baseline | 0.8867 | 0.8056 | 0.8238 | 14 ms |
| BM25 | 0.8778 | 0.9167 | 0.8817 | 4 ms |
| RRF + CrossEncoder rerank | 0.9089 | 1.0000 | 0.9406 | 647 ms |
| Selected-fulltext section-aware | 0.9222 | 0.8167 | 0.8426 | 61 ms |
| SPECTER2 + RRF | 0.9089 | 0.9333 | 0.8942 | 32 ms |

生产默认仍为 abstract + MiniLM + RRF。Reranker 能提高质量，但增加了明显延迟。
Section-aware 是当前优先的全文策略，但只有端到端 Survey benchmark 证明其产品收益后，
才考虑调整默认配置。

完整结果和解释边界见
[`benchmarks/rag/RESULTS.md`](./benchmarks/rag/RESULTS.md)。

## Survey Benchmark

版本化 Survey benchmark 包含 Few-shot、Vision Transformer 和 NeRF 三个领域共 15 个
case，以及 54 篇经过 judgment 的论文 Corpus。比较对象为：

- 仅使用 external search
- external search + abstract RAG
- external search + selected-fulltext RAG

实现和确定性测试已经完成。一次成本受控的 9 次真实 pilot 已落盘全部 observation 和
Judge，校验了 9 份 RunArtifact 哈希，并确认 18 条 Survey/Judge LangFuse trace 都有
input/output。结果仍为 `partial`：没有任何 profile pair 具备相同的外部输入，核心论文
citation recall 全部为 0，运行时 faithfulness evaluation 全部跳过，且一条 selected-fulltext
Survey 在 synthesis 阶段失败。详见
[`脱敏 pilot 结果`](./benchmarks/survey/results/pilot-2026-08-25.md)。因此当前项目不声明胜出
profile，也不声称全文 RAG 已经改善最终 Survey。
该 pilot 暴露的问题已完成可验证修复：benchmark profile 复用同一份冻结的
Planner/Search 快照，paper flow 按阶段度量，运行时 evaluator 复用已配置 provider，
长上下文生成使用有界紧凑重试，且 observation/token 预算会在下一次付费调用前
停止。随后的一条
[`修复后真实 pilot`](./benchmarks/survey/results/post-repair-pilot-2026-09-08.md)
最终 runtime faithfulness 为 0.8958、delivery 为 `ready`；但它使用同模型 Judge 和 dirty
worktree，只验证修复链路，不构成跨 profile 收益或正式 baseline。

## 质量、安全与失败语义

- `ready`：执行完整、质量通过、允许发布。
- `needs_review`：执行完成，但质量未验证或存在关键降级。
- `blocked`：执行完整但质量失败，保留草稿且禁止发布。
- `partial`：因失败、取消、超时或预算终止造成执行不完整。

外部论文内容会经过 prompt injection 检测和 provenance 边界。Tool 调用必须满足已注册名称、
允许类别和合法 JSON 参数。未知 MCP capability 以及 write/destructive 操作默认拒绝。
API 使用全局并发限制和协作式 CancellationToken。

默认测试会阻止未标记网络访问，Integration 和 Live 测试使用显式 marker 和隔离存储身份。

```powershell
D:\miniconda3\envs\litagent\python.exe -m pytest tests -q -p no:cacheprovider --basetemp=.pytest-tmp-readme
```

## 项目结构

```text
src/litagent/
  agents/          Planner 和 DAG Worker
  orchestrator/    TaskGraph、Scheduler、重试、超时与取消
  rag/             Corpus ingestion、向量存储、检索和 Claims
  memory/          Working、Episodic、Semantic、Procedural Memory
  eval/            Citation、Consistency、Faithfulness Evaluation
  observability/   LangFuse、脱敏、RunArtifact 和 Archive
  benchmark/       Ingestion、Retrieval 和端到端 Survey Benchmark
  static/          Flow Replay 页面和离线样例
benchmarks/         版本化 Dataset、Profile 和审核结果
docs/               公开架构、ADR 和 Design Truth
tests/              默认离线、Integration 和 Live 测试
```

## 持久化边界

- **PostgreSQL**：Semantic/Procedural Memory 和可恢复的 CorpusState。
- **Qdrant**：Paper chunk、Trusted Claim 和 Episodic Memory，不同用途使用独立 collection。
- **Redis**：具有 TTL 的短期 Working Memory。
- **RunArtifact JSON**：完整本地执行回放。
- **Manifest 与原始资产**：派生向量索引的重建来源。

Docker Compose 为 PostgreSQL 和 Qdrant 使用 named volume。Working Memory 的临时语义不会
被描述成长任务断点恢复。

## 已知限制

- PDF 流程不包含 OCR、公式理解、表格抽取或图片理解。
- Benchmark Corpus 有意保持小规模和领域化，不能外推到任意学术检索任务。
- 独立 LLM Judge 是确定性指标的补充，不能替代领域专家审核。
- 当前是本地工程系统，不是具备认证、计费和多租户隔离的托管平台。
- 内置离线样例只验证回放契约，不衡量模型质量。

## 延伸文档

- [系统概览](./docs/system-overview.md)
- [Design Truth 与证据矩阵](./docs/design-truth.md)
- [Qdrant 与 Embedding ADR](./docs/adr/001-qdrant-and-embedding.md)
- [RAG Benchmark 方法](./benchmarks/rag/README.md)
- [Survey Benchmark 方法](./benchmarks/survey/README.md)
