# LitAgent

[![简体中文](https://img.shields.io/badge/lang-简体中文-c5221f.svg)](./README.md)
[![English](https://img.shields.io/badge/lang-English-1769aa.svg)](./README.en.md)

**基于多 Agent 协作与证据约束的学术文献综述系统。**

围绕研究问题检索和整理相关论文，生成带引用的文献综述。
支持报告阅读、原文追溯、审阅修订和多格式导出。

[系统架构](#系统架构) · [实验结果](#实验与验证) ·
[效果展示](#效果展示) · [快速开始](#快速开始)

## 项目概览

LitAgent 将文献综述流程拆分为检索、筛选、提取、写作和审阅等任务，由不同 Agent 协作完成。
每份报告都保留引用来源、审阅记录和评估结果，方便阅读、核查与后续修改。

- **输入**：研究问题、运行配置，以及可选的本地论文语料。
- **处理**：外部搜索与本地 RAG → 去重筛选 → 证据提取 → 写作审阅 → 核验修订。
- **输出**：结构化综述、参考文献、原文定位、评估结果和运行记录。

## 系统架构

![LitAgent 系统流程与四层记忆读写图](./docs/images/readme-workflow-memory-zh.svg)

左侧是运行流程。外部论文搜索与本地 RAG 并行，证据抽取与引用网络分析并行。右侧四层记忆通过同色虚线接到使用节点，箭头指向节点表示读取，指向记忆表示写入。
节点角标中，`LLM` 表示调用生成式模型，`LLM*` 表示按任务或配置调用，`代码` 表示该步骤无生成式 LLM 调用。检索和排序仍可使用向量模型或重排模型。
工作记忆按运行、Agent 和工作单元隔离，记录已采纳的决定、待解决问题和近期改动。
程序性记忆在搜索结束后累计成功、空结果、限流和超时。情景记忆在报告满足交付条件后保存调研摘要，供下一次规划召回。

语义记忆统一在 Extract 前按研究主题召回。系统核对来源和适用条件后，将符合条件的证据纳入本轮账本，供后续写作、审阅和评估使用。报告满足交付条件后，只将最终版本实际引用且核验通过的证据写入语义记忆。

## 核心能力

- **多 Agent 协作**：按任务依赖异步调度，分别完成搜索、提取、写作和审阅。
- **混合检索**：结合外部论文搜索与本地语料，支持 dense、BM25、RRF 融合和重排。
- **引用核验**：关联报告论断、引用文献和原文片段，检查引用是否支持对应结论。
- **人工审阅**：支持论文选择和终稿审核，可从检查点恢复任务。
- **运行追踪**：记录任务状态、模型调用和用量，支持历史回放与 LangFuse 追踪。

## 实验与验证

### 检索实验

使用 **25 篇论文、15 条查询，每条重复 3 次**，比较不同检索配置的召回率、排序质量和延迟。

| 配置 | Recall@5 | nDCG@10 | p50 ms |
| --- | ---: | ---: | ---: |
| MiniLM + RRF | 0.8867 | 0.8238 | 14 |
| BM25 | 0.8778 | 0.8817 | 4 |
| MiniLM dense | 0.7711 | 0.7440 | 10 |
| RRF + reranker | 0.9089 | 0.9406 | 647 |
| 全文 section-aware | 0.9222 | 0.8426 | 61 |

RRF + reranker 的 nDCG@10 达到 0.9406，p50 延迟为 647 ms；MiniLM + RRF
的 p50 为 14 ms，可根据排序质量和响应速度的需求选择。全文配置采用 section-aware 切分，
单独比较不同索引材料下的检索表现。

项目使用 pytest 覆盖任务编排、证据处理、评估修订、断点恢复及 API 行为。
检索结果和功能验证记录分别保存在 `benchmarks/rag/RESULTS.md` 和 `docs/design-truth.md`。

## 效果展示

![真实生成的 Vision Transformer 综述，显示目录、摘要与行内引用](./docs/images/readme-workbench-zh.jpg)

截图来自一次真实的 Vision Transformer 文献综述运行。复评中，561 条科学断言有 536 条获实际引用支持（95.54%），这版草稿未达到严格交付标准。

## 快速开始

### 无密钥离线演示

需要 Python 3.11。演示模式无需 API key 或 Docker。

```sh
python -m venv .venv
# macOS / Linux
source .venv/bin/activate
# Windows PowerShell: .venv\Scripts\Activate.ps1
python -m pip install -c requirements/release-constraints.txt -e ".[api]"
litagent demo --presentation
```

### 可选真实运行

```sh
python -m pip install -c requirements/release-constraints.txt -e ".[dev]"
# 复制 .env.example 为 .env，填写 OpenAI-compatible provider key。
docker compose up -d
litagent demo --live
```

真实运行需要配置模型服务，费用由所选服务商收取。
环境变量示例在 `.env.example`，默认配置在 `config/default.yaml`。

### 开发验证

```sh
python -m pytest -v
python -m black --check src tests
python -m isort --check-only --profile black src tests
```

集成测试与真实模型测试使用 `integration`、`live` 标记，按需单独运行。
