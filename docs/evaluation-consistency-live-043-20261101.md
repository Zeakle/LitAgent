# Phase 14.5：043 固定三域真实验收结果

043已真实执行并保存协议失败终态。本轮计划3题、运行1题、生成1份完整草稿、完整Evaluation 0题、严格交付0/3。ViT与NeRF因技术停止规则未运行，仍留在三题分母；不恢复043，不重复请求或追加到通过。Phase14.5继续in_progress。

## 执行、身份与预算

- 套件：frozen-writer-three-domain-20261101-043；manifest身份78e11bf4ad72b0276555032db3c18603bc3fc1715a3fdab983c2f011311870d5。
- 源码584文件，摘要50447d08f2e6d9a642d7f2c70282fc69c5f57244e4337f1646930da99cb57ae8；本步未修改源码、配置或测试。继承的6785默认测试与最大12＋4验证均为同源真实退出0，不重复把测试绿值当科学证明。
- 根唯一工具session77309，wrapper44366／child44405，真实终端chunk993d61／进程退出0，2220.17秒。退出0表示正确保存停止终态，不表示科学通过。
- Pro负责正文、摘要和修订，Lite负责Reviewer／Evaluation，均为Doubao Seed2.1 260915；thinking关闭、零传输重试、两次共享机会，Judge/Jev关闭，无人工审批节点。
- 三题启动锁定444／448／448万tokens、合1340万／192请求。本轮未耗尽总预算、请求数或上下文；预留不是实际费用。

## 结果与未完成环节

| 题目 | 正文／Reviewer | Evaluation | 严格交付 |
|---|---|---|---|
| few-shot | 五章46单元，摘要1／结论1，Reviewer完成 | 第三轮第4支持批返回内部矛盾，未完成 | 否 |
| ViT | 未运行 | 未运行 | 否 |
| NeRF | 未运行 | 未运行 | 否 |

正文SHA始终为c44d2a2b5d46cdc44f33a6a1e4c23b5b41c6faea6302ce6e56aae83b678a843c。两次共享机会均用于清单恢复，正文改写0次，未提交失败候选。

三轮完整抽取分别169／185／175条，科学分类分别162／177／168。前两轮完成全部支持批与global，但仍有3／4条原子性诊断，清单整体无效；第三轮只接受前三批48条，第4批拒绝，未执行第三轮global。不能把前两轮raw支持数量129／33或147／30当有效分数，更不能将最终0／0／168占位值解释为科学零分。最终应显示未完成核验。

### 直接停止原因

第三轮第4批的claim_60与claim_61同时返回：

- status为unsupported，reason为attribution_mismatch；
- atomic、classification_valid与source_faithful全部为true；
- 全部必需方面predicate、conditions均supported=true，并提供权限内支持p3。

现有_validate_decision_consistency因此按negative_status_with_complete_positive_diagnostics拒绝整批，终态为evaluation_decision_consistency_invalid。原始10554字符响应保存在checkpoint，并与同operation归档逐字一致，SHA为c7af628e7150a9f488aa9c0798c634e66f2254de18f5081e3f9bae7c58ef4967。该次用量8106＋4240已结算，没有将unsupported自动改为supported。本次诊断只证明输出字段互相冲突，不证明论文事实正确。

### 另外两个真实闭环缺口

1. 清单谱系不稳定：同一报告、原文片段、偏移和引用集合没有变化；一组原复合断言在第二轮拆为三条，第三轮又逐字合回原句。整份清单hash不同不能阻止局部回归；fragment覆盖也不能替代逐命题保留证明。另有同一因果链的atomic=false解释同时肯定原文一致，不能按标点机械裁判。
2. 当前显式策略优先修复无效清单，并与正文修复共用两次机会。两次都用完后，正文问题未进入科学改写；这与冻结控制流一致，不是计数错误。展示中的repairs=2不能称正文修好了两次，不增第三次机会。

## 用量与历史保护

本轮新增39次模型请求、599976已知tokens、新增未知占用0，全部settled。43个费用父链累计1111次模型请求、9675343已知tokens、1283047历史未知占用；038的78205未知占用保留。历史Jev另计，本轮没有Judge/Jev请求。

独立终态审计真实退出0：42旧认证前缀、全部新增wire与snapshot闭合，584源文件和23763保护文件逐SHA不变。Plan.md不改；未commit、push或tag；未停止、重启或替换8000／8020服务。

## 展示与导出验收

公开包：benchmarks/survey/results/frozen-writer-three-domain-20261101-043/benchmark.{json,csv,md,html}。四格式共同字段、失败原因与费用一致，HTML完整内嵌JSON与原JSON相同。Reviewer=true在JSON及HTML原始诊断中存在，但CSV／Markdown／HTML主矩阵缺少独立Reviewer列；这项呈现要求未完成，不能以原始JSON替代可读列。

8020真实浏览器已检查1280×720和390×840、中英文document.lang、无页面横向溢出、Enter展开／收起诊断。临时viewport已reset回1280×720，保存6张实际截图。浏览器的标签持久标记遭自动审批拒绝后已去掉，未使用替代标记绕过，报告交付状态未变。

8000现有只读回放服务未收录043，实际页面显示run not found；新报告原文和证据的工作台定位仍未验收。没有更换端口或重启服务绕过。

few-shot完整草稿经原生产导出器生成Markdown／PDF／Word和BibTeX／RIS／CSL JSON，均绑定同一SHA、revision1，publishable=false。PDF7页由根逐页原尺寸检查；Word13页由独立代理逐页原尺寸检查；中文、引用、书目、附录可读，未见裁切或重叠。Word附录密集、末页留白较多如实保留，不改原文。三份实际报告导出不等于科学验收通过。

独立六格式一致性审计002实际退出0：140条原始证据引用映射保持，Markdown／PDF／Word的103个显示编号逐项同序；9篇书目、标题／URL及正文→书目→附录顺序一致，三种文献数据均可解析。54条原元数据警告完整保留，未知作者、年份和卷期页码没有补写。BibTeX／RIS按既有单行字段契约将标题换行转为空格，保留文字；RIS未单独映射genre／archive字段，不能宣称所有snapshot扩展字段均无损交换。此项限制与科学验收状态分别记录。

## 本轮结论与下一步

- 生成、摘要和Reviewer真实完成，目标内索引修复本轮未触发，因此不能将其科学修复效果记为已验证。
- 扩批、结算及失败保留真实执行，账本可核对；没有预算耗尽。完整正文科学修复与最终复评仍未闭合。
- 严格质量目标未达到，不推广默认、不启动15题，也不追加同组样本。

下一步先离线审查判定契约与清单恢复的稳定性：用043两条矛盾判定及三组跨轮原文配对建立真实消费者正反例，明确来源忠实性、原子性、科学支持及归属的责任；保留全部科学门禁和两次共享上限，不自动改分、不只加一条提示就再次付费。任何生产修改须先同步当前步骤指导，完成同源相关、默认、最大资源与历史保护验收。本轮不预先声称已修好。

## 证据入口

- artifacts/acceptance/repair-local-indices-live-20261101/launch043/actual-tool-exit-001.json
- artifacts/acceptance/repair-local-indices-live-20261101/terminal043-review/terminal-cost-review-001-proof.json
- artifacts/acceptance/repair-local-indices-live-20261101/diagnostics/consistency-002/proof.json
- artifacts/acceptance/repair-local-indices-live-20261101/nonatomic-lineage-review/diagnostic-001.json
- artifacts/acceptance/repair-local-indices-live-20261101/inventory-priority-review-001.json
- artifacts/acceptance/repair-local-indices-live-20261101/presentation-audit/four-format-audit-001.json
- artifacts/acceptance/repair-local-indices-live-20261101/launch043/ui/browser-observations.json
- artifacts/acceptance/repair-local-indices-live-20261101/launch043/exports/few-shot-metric/exports-receipt.json
- artifacts/acceptance/repair-local-indices-live-20261101/launch043/exports/few-shot-metric/pdf-visual-review-root.json
- artifacts/acceptance/repair-local-indices-live-20261101/launch043/exports/visual-review/docx-visual-review-001.json

- artifacts/acceptance/repair-local-indices-live-20261101/export-audit/six-export-audit-002.json
