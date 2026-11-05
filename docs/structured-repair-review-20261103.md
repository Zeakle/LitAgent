# 结构化修复契约复核与051终态

2026-11-03。用户确认：仍最多两次共享修复HTTP，完整Evaluation解释保留本地，修复模型接收宿主精确定位、结构化失败项及完整原文依据，科学门禁、预算和历史不变。

核对发现，该能力已经在步骤287实现并进入当前冻结策略。本次复查真实发送和消费路径，没有重复新增协议、修改生产或改写历史结果。当前Phase 14.5仍为`in_progress`。

## 已落实的契约

- 当前稿与已认证失败候选的诊断分别绑定SHA、原文位置、实际引用和授权目标。所有可定位失败项保留；未知类型或无法定位时明确不可修复。
- 自由解释完整保留本地，模型消费结构化类别、状态、失败aspects、完整claim和冻结`source_quote`。事实卡的压缩提取不能代替原始支持文本。
- 正文、摘要、Reviewer、本地契约及容量入口使用同一选择。摘要依赖不授予新科学事实；未核验清单不伪装成不支持判定。
- 清单恢复与正文修复共享最多两次。失败候选不成为正式原稿，只有完整复评及全部严格门禁通过后才原子提交。
- 选择和身份进入冻结计划、请求及checkpoint；旧None/V1读取和身份保留，普通API不公开内部selector。

实际接口见[投影入口](../src/litagent/writing_flat.py#L1211)、[请求组装](../src/litagent/writing_v2.py#L2464)和[当前稿绑定](../src/litagent/writing_v2.py#L3303)。原实现及最大资源证明见[287验收](structured-repair-input-20261102.md)。

## 本次离线证据

实际pytest PID14499，退出0：**311 passed，5 warnings，108.52秒**。覆盖4个structured文件、task-view与真实消费者链、当前V2修复资格。13个相关源/测试文件执行前后SHA一致。

当前V2资格链与旧V1长解释链的测试范围分别记录，不能把它们合称所有组合已穷尽。没有可复现的新生产缺陷，因此本次没有新增重复实现或只镜像实现的测试。

完整625源文件仍为`7c185d4bf78615321702310e56a326592a20ce0b210e2c8a23131e59eb9fb1af`。默认离线8015通过继承同源码306的实际退出0证据，本次没有重跑全量或最大26/9资源矩阵。原42664历史与Plan.md逐SHA保持。新增付费请求为0，未恢复051。

证据：[独立审阅](../artifacts/acceptance/structured-repair-contract-review-20261103/review/review.md)、[实际进程](../artifacts/acceptance/structured-repair-contract-review-20261103/review/process-001.json)、[历史保护](../artifacts/acceptance/classification-agreement-live-20261103/terminal-protection-proof.json)。

## 最新真实运行仍未通过

051已终态，controller9641/child9706实际退出0仅表示控制器正常结束。套件停止于ViT的`writer_capacity_source_map_invalid`，**生成2/3、Reviewer完成2/3、完整Evaluation 0/3、严格交付0/3**，NeRF未运行且保留计划分母。

| 题目 | 实际失败 | 修复与核验状态 |
| --- | --- | --- |
| few-shot | `evaluation_inventory_invalid` | 两次机会均用于清单恢复；未变断言的原子性判断出现漂移。最终102支持、33不支持、15未核验仅为诊断，整体无有效科学分数；未发生Pro正文修复或候选提交。 |
| ViT | `writer_capacity_source_map_invalid` | 容量精简返回48条patch，其中37条来源属于其他目标，宿主正确拒绝；186科学条目未完成支持核验，无候选提交。 |
| NeRF | `not_started_after_stop` | 未运行，不能记为科学零分或从分母删除。 |

ViT原JSON及native schema合法，原system也已明确同目标来源要求。不能自动改编号后追认成功。few-shot清单失败和科学支持未达标是两道不同门禁，不能仅解决原子性标签就宣称科学通过。

本轮真实新增52请求、794339已知tokens、0新增未知；累计1403请求、14141938已知tokens、1283047历史未知占用。旧Jev67请求/756769保守输入、usage不完整及038未知占用继续保留。失败响应计费，预留不作为实费。

诊断：[few-shot](../artifacts/acceptance/classification-agreement-live-20261103/partial-case-review/few-shot.md)、[ViT归属重放](../artifacts/acceptance/classification-agreement-live-20261103/terminal-map-review/review.md)、[终态费用](../artifacts/acceptance/classification-agreement-live-20261103/terminal-cost-review/review-002.md)。

## 展示、导出与未验边界

- 同源JSON/CSV/Markdown/HTML核对计划分母、终态、各题数量和usage；HTML嵌入数据等于JSON，实际浏览器显示2生成、0完整核验、0交付。
- 8020限定页面桌面1280×900、移动390×844、中英文和键盘操作完成，宽表在可滚动容器中，临时viewport已reset。
- 8000返回归档not found。本次未改服务，原文定位、引用跳转及折叠附录UI不能记为通过。
- 两份原稿实际导出Markdown/PDF/Word及BibTeX/RIS/CSL JSON，保留原文SHA和未核验草稿标识。PDF共15页、Word共27页，42页均实际逐页检查，未见截断、重叠、缺字。Word默认蓝标题及元数据缺字段警告如实保留，未为美化重写正文或补元数据。
- NeRF未生成，不存在其报告导出验收。导出和视觉通过不能替代科学核验。

浏览器持久化标签操作曾被自动审批拒绝，原因是授权仅含读取、交互检查及截图。未执行，也未通过其他方式绕过；普通结果链接仍可打开。

## 后续工作

先收敛清单原子性与支持判断的职责、全局输入分类表示，以及容量精简的目标/来源关系表达。任何新输出契约都需保留原051拒绝、旧身份和来源血缘反例，并重新做离线真实消费者与完整资源验收。本次不隐式实施新协议、不新增修复机会或付费套件；严格科学1.0、完整覆盖和全局门禁不变。
