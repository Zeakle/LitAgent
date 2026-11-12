# 演示与技术讲解

## 启动与准备

按[中文首页](../README.md#快速开始)安装后运行 `litagent demo --presentation`；
[English](../README.en.md#quick-start) 提供相同启动步骤。浏览器先开
`http://127.0.0.1:8000/presentation`，再进入工作台。
演示不需要密钥或后台 Docker，也不读取私人 artifacts。依赖安装仍需网络。
现场不启动新的付费 Survey；预录屏和页面是网络不稳定时的备用材料。

本机验证环境：Python 3.11.15、Node 24.14.0、macOS。Linux 由远端 CI 验证，
未看到成功 run 前不能标成已通过。学术引用样式需要 Node；缺少时明确报错。
中文 PDF 使用可用 CJK 字体；Word 阅读器需 CJK 字体，例如 macOS Arial Unicode MS、
Windows Microsoft YaHei，Linux 可安装 Noto CJK。字体缺失不伪称导出视觉通过。
本轮真实中文草稿：PDF七页、Word十三页。隔离的headless LibreOffice默认只扫描随包字体，
首次预览缺中文字形；将本机已安装CJK字体显式加入本次FONTCONFIG_FILE后，同一DOCX
的全部十三页可见且无越界。未修改系统字体设置或DOCX；仅能提取中文文本不算视觉通过。

## 三分钟脚本

| 时间 | 操作 | 讲解重点 |
| --- | --- | --- |
| 0:00–0:30 | 打开公开依据页 | 文献助手的问题是可追溯证据和错误交付，不只是长文本生成；live 仍实验中 |
| 0:30–1:10 | sample-ready：正文、引用、证据、质量 | 此为离线夹具；展示统一数据契约和版本 SHA，不称模型质量 |
| 1:10–1:50 | 检索表 | 固定 25 篇/15 查询，RRF 与 reranker 的质量/延迟取舍；小数据集边界 |
| 1:50–2:30 | 打开「真实修订案例 377 → 380」 | 点 claim_191，看修订前后原句和实际引用；说明冻结证据、全篇复评和 9 次请求 |
| 2:30–3:00 | 导出、架构、恢复说明 | 相同版本本地导出；请求预算、未知 usage、重启不重发；下一步是受控验证 |

语言切换后核对文字和 HTML lang；390px 可打开侧栏，键盘 Tab/Space 可展开诊断。
不要把未运行指标说成零分，不要用合成 ready 页面证明当前 strict 策略成功。

## 架构讲解与追问

- 当前架构以[源码核对后的流程图](system-overview.md)为准：Survey 由自定义
  TaskGraph/asyncio Scheduler 编排，LangGraph 用于 ReAct 子流程；当前结构化写作直接使用 WriterSession。
- 独立双循环需显式选择 `config/separate-review-evaluation.yaml`：Reviewer 负责编辑审阅，
  Evaluation 负责独立核验与修复复评，两类修订预算独立。编辑接受与科学通过分别展示，
  已改善但未通过的候选只能作为工作草稿，不能自动发布。历史样例不补造双循环记录。
- 为什么多 Agent：显式任务依赖与独立审阅职责；没有为展示而新加 Agent。
- 如何限制循环：运行前锁定预算与次数，遇到无进展或候选回归就停止。本案例沿用正文五轮上限，累计两轮即通过。
- 如何处理幻觉：完整支持文本与候选 finding 分离；每条科学断言只检查实际引文，
  Evaluation 与可选 Judge 也可能误判，因此不保证任意主题绝对正确。
- 如何防错归属：稳定 evidence/paper ID、宿主定位、条件/数值校验和真实反例。
- 如何恢复：输入/配置/源码身份绑定，先落盘已发请求，未知响应不自动重发。
- 怎样理解消融：已有 BM25/dense/RRF/reranker 对比；全文和 embedding 对比单独说明。
  没有开展可宣称获胜的端到端模型对照，不推断 Pro 比 Lite 提升多少。

## 最新状态与失败复盘

[377 → 380 公开案例](../src/litagent/static/presentation/frozen-repair-377-380.json)展示同一份
few-shot 综述如何从拒绝交付变为通过。点击 `claim_191` 定位 SRL 实验比较的原句，
再沿 `E` 链接查看实际引用的论文。377 的逐条结果为 330/339 支持、9 条不支持，正式
支持率因清单检查未通过而不可用。380 重新抽取并复评，398/398 条获支持，核心覆盖与
全局检查通过；新增 9 请求、188,560 tokens。这个结果使用冻结证据，单列于正式 15 题套件。

### 历史固定收尾套件

[055 结果](../src/litagent/static/presentation/closeout.json)：计划3题，实际运行1题。
五章正文和摘要完成，但Reviewer用满4096输出tokens仍返回incomplete；Evaluation未开始，
另外两题按技术停止规则未运行。7请求、96,102已知tokens、新增unknown0，没有追加套件。

查看首页的历史收尾记录，说明生成1/3、有效核验0/3、严格交付0/3；再阅读
未审阅草稿、用键盘从引用跳到来源、展开检索附录，最后解释为何没有交付。完整用量和
原草稿SHA可查，不能把协议完成、合成ready或历史较宽松成功说成当前科学质量通过。

历史054曾完成三份正文、摘要和Reviewer，但有效整篇Evaluation也为0/3；只作此前失败
对照，不与055混算。单一Reviewer输出容量的限制留作后续工程/实验改进，本轮不再付费试跑。

### 历史协议失败回放

随包的历史公开诊断完成 50 篇提取，26 次 Lite、1 次 Pro，总计 124,448 tokens。
Pro 在 finish=stop、未耗尽 20k 输出上限时返回非法 JSON，字符偏移 8299。
系统保留原响应，阻止正文提交，Reviewer/Evaluation 未运行。

修复候选减少传输容器嵌套并无损展开本地 schema 引用；保留来源权限和所有约束。
离线 schema/消费者回归证明实现契约，真实有界协议测试单独记账；不能补括号冒充合法结果，
也不能把 schema 通过说成科学通过。协议验证结果另见发布验收记录。

## 简历项目描述（按实际职责改写）

**LitAgent｜证据约束型多 Agent 文献综述工作台｜Python / LangGraph / Qdrant / FastAPI / LangFuse**

- 构建异步 DAG 文献处理流程，串联搜索、去重、提取、生成、对抗审阅与版本化核验；
  将正文、证据、SHA 和交付状态绑定，失败候选完整复评后才允许提交。
- 实现混合检索及可复现对比：25 篇论文、15 条查询、各重复 3 次；RRF Recall@5
  0.8867，reranker 为 0.9089，同时记录 p50 从 14ms 增至 647ms 的性能取舍。
- 建立有界模型预算、取消与未知 usage 保守计账、checkpoint 恢复和 trace 回放，
  提供无密钥演示、原文定位、质量诊断和六类本地导出。

不能写：15/15 已通过、科学准确率 100%、所有 Survey 都可交付、人工金标准已校准、
Pro 已带来统计显著提升。独立完成/主导等归属表述应由本人按实际投入确认。

## 保存的演示材料

- **当前首页截图**：[中文工作台与证据检查](images/readme-workbench-zh.jpg)。
  2026-11-12 从源码 `bdfecbd` 启动 `litagent demo --presentation` 后采集，
  选择合成 `sample-ready` 并打开引用检查面板。截图展示当前界面；样例内容不证明模型质量，
  也不用于证明新双循环已完成真实运行。
- **历史演示素材**：[25 秒截屏讲解视频](images/demo-walkthrough.mp4)和
  [字幕](images/demo-walkthrough.srt)，归档于 2026-10-20 提交 `19cd514`。
  这是当时的页面与导出检查截屏序列，不是实时生成录屏或重构后界面实录；
  内置报告为合成夹具。[历史公开依据页截图](images/public-overview.jpg)仅作旧版参考。
