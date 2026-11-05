# 摘要契约与完整链路验收：034（2026-10-31）

## 结论

摘要数量契约已通过真实首次生成和正文修订后的摘要更新。034 首题完成五章、摘要、Reviewer、一次容量精简和五批候选核验，但完整链路仍未通过：宿主组装第二次修复输入时复制了完整评估记录，导致衍生输入超限。后两题按技术故障规则未启动，严格交付仍为 **0/3**，Phase 14.5 保持 `in_progress`。

不能将本次解释为总预算耗尽、摘要数量再次冲突或科学质量已经合格。旧 031 结论 3/2 仍被拒绝；034 没有恢复 031–033，没有额外付费摘要试跑，也没有在失败后追加套件。

## 冻结与离线依据

独立套件身份为 `6381b28c97c35d58ec53e6216a92743963c754ba3f6c89f7e7819f03d01fee2e`。544 份源码、原三域材料、完整 34 父账本及 8768 份保护文件至终态保持。实际预算计划为 443/447/447 万 tokens，合 1337 万；每题硬上限 450 万/64 请求，三题最多 192 请求。Pro 写作、摘要和修订，Lite Reviewer/Evaluation；思考、Judge/Jev 关闭，零传输重试，两次共享修复，科学支持率 1.0 不变。

034 绑定最终默认离线 `pytest -v` 的 **5633 通过、25 排除、8 既有 warnings、实际退出 0**；90 组最大请求、两类各三域最大资源路径、旧六份计划、031 原响应回放、相关 Black/isort/diff 均已通过。当前验收期间没有更改这组受测源码，没有用未执行的新测试冒充证明。

这些离线最大路径没有覆盖失败候选的完整 canonical 诊断再次进入修复的实际 producer/consumer 组合，034 因此暴露新的工程缺口。不能以原有测试全绿否定该缺口，也不能继续宣称完整预算保障已成立。

## 实际流程与版本边界

| 阶段 | few-shot | ViT | NeRF |
|---|---|---|---|
| 五章正文 | 50 单元，完成 | 未运行 | 未运行 |
| 首次摘要/结论 | 各 2 单元，完整编译 | 未运行 | 未运行 |
| Reviewer | 完成 | 未运行 | 未运行 |
| 首次抽取 | 177 项，其中 165 科学断言；完整工作流容量不足 | 未运行 | 未运行 |
| 共享修复 | 已使用 1/2，容量精简 | 未运行 | 未运行 |
| 候选摘要更新 | 完成，仍各 2 单元 | 未运行 | 未运行 |
| 候选复评 | 77 项、5/5 批返回、全局检查返回；清单有效性失败 | 未运行 | 未运行 |
| 第二次修复 | 请求构造前检查失败，未发送、未计第二次修复 | 未运行 | 未运行 |
| 严格交付 | 不可交付 | 未运行、不可交付 | 未运行、不可交付 |

正式保留原稿 SHA：

`095c9eeb3b1cb82b583b253894aaceeaccbd706b67c4ba3610ab8ecf5273e50e`

未提交候选 SHA：

`8e3ff33307bc61bbac05d9d00986104940693a09afdac394e318b38a24e966c4`

候选 73 条科学断言中，局部结果为 **49 支持、9 不支持、15 未核验**，另有 4 条其他类别。清单有效性失败，候选没有提交；这些数量只解释该候选，不能写回原稿评分或当作最终有效支持率。正式原稿仍显示 165 条未核验，不能把不同 SHA 的统计混在一起。

## 根因与实际重放

使用原始完整响应、真实请求构造器和消费者，只在本地重放，新增模型请求为零。

1. `body_quality.bound_repair_diagnostics` 复制完整 canonical 失败记录，包含支持校验、原文引用、语义检查及来源身份。
2. 第一次候选失败后，25 条记录保存到 `journal.failure`；下一次修复以未提交原稿为基线，将该记录完整作为 `previous_local_failure` 传入。
3. `writing_flat.repair_projection` 共享目标解释，却未投影整份前次失败记录。仅 `previous_local_failure` 已占 92,264 字节，整个指导材料 **96,996 > 12,288 字节**，确定性复现原 `writer_envelope_exceeded`。
4. 该检查发生在后续预留、次数增加和发送之前，没有第 18 次付费请求。17 份原始 SDK 响应均为完整 JSON/stop；最大单诊断字符串 535 字符，没有超过 2000 字符限制。
5. 仅在私有内存中绕过已证字节守卫进行计量、不接受也不发送，该完整请求预留为 **182,780 > 117,964**。因此仅抬高 12 KiB 限制仍会超单请求上下文。

原归档保留 `output_scope / writer_envelope_exceeded`，不改写历史。独立诊断将其归因为**宿主衍生修复表示与完整预算规划契约缺口**，不是模型返回了 97 KB 非法修复输出。守卫正确拒绝了请求，但剩余修复可完成的工程保证尚未兑现。

## 账本与可观测性

实际 **17 请求、249,496 已知 tokens、0 新增未知占用、0 思考 tokens**；Pro 8 次、Lite 9 次。17 个 operation、start、complete、账本条目及 usage 增量逐项对应并各结算一次。443 万计划额度不是实际费用，未推算精确人民币成本。

包含历史累计 908 请求、6,540,832 已知 tokens、1,204,842 旧未知占用。032 等未知占用不释放、不当零费用；历史 Jev 67 请求/756,769 保守输入 tokens 单列，usage 不完整。新套件没有 Judge/Jev 请求。

LangFuse 实际只读核对 17 条 generation，模型/API/usage 一致，科学输入、输出及完整消息为空。Trace：

[034 Trace](http://localhost:3000/project/litagent-local-project/traces/b3b99342928f35a9b7e1125bfa6f4759)

## 展示与实际导出

原 JSON/CSV/Markdown/HTML 统计同源，三题均保留在计划分母。原稿和未提交候选在说明包中分开；原始 `benchmark.*`、归档及评分不覆盖。

- 桌面 1280×900、移动 390×844、中英文 `document.lang`、键盘语言切换/菜单 Escape/附录 Enter、IEEE/APA7、两条真实来源 Inspector 和附录顺序均已实际检查。最终移动抽屉 `data-open=false`、scrim 隐藏、右边界 -6.20，不能把关闭动画首帧当终态。
- 034 实际只有一个正式 revision，不冒充实际多版本验收；旧多版本隔离由既有离线消费者测试覆盖。
- 原工作台主要呈现正式原稿及其首次容量诊断；没有将未提交候选的五批局部诊断完整汇总到正式质量页。独立说明包披露候选结果，完整工作台候选展示仍是后续缺口，不能把它称为已修复。
- 六类真实草稿导出完成：Markdown/PDF/Word 与 BibTeX/RIS/CSL JSON。PDF 6 页、Word 11 页共 17 页全部看图；中文、引用、附录与完整文本一致。148 条证据出现记录、54 引用簇、117 可见编号字符串、9 篇书目是不同单位，不能混加。54 条元数据警告保留，没有补写作者年份。
- 导出绑定正式原稿 SHA，保留不可交付草稿标识，不使用失败候选。没有为未生成的后两题伪造报告或导出验收；原033/260 导出不变。原8000/8020服务不重启，不新增生成或核验入口。

## 后续最小修复与门禁

下一步仅离线收敛修复反馈传输：完整 canonical 审计留本地；按精确目标/来源共享已有正文与原文，保留每条失败解释、条件和支持判定。必须同时绑定候选 SHA、原稿 SHA 与目标映射，不能用相同文本搜索或错误 SHA 借用诊断。

加入034真实失败候选→剩余修复构造链和最大规模反例，证明所有失败完整保留、完整来源与两次修复及最终复评都进入计划。不能通过删除诊断、扩大上下文、提高总预算或降低科学门禁换通过。新表示必须显式版本化、旧身份保持；输出大小合法仍超锁定计划时应归因为宿主规划违约。

本轮不恢复034、不额外试跑、不新开035、不自动进入15题或推广默认。三域完整链路与科学质量目标尚未达成，不能以摘要完成、导出可读或测试数量代替。未commit/push，`Plan.md` 未修改。

## 可核查证据

- `artifacts/acceptance/report-final-readiness-20261031/034-execution-proof.json`
- `artifacts/acceptance/report-final-readiness-20261031/terminal-audit/terminal-proof.json`、`trace-proof.json`
- `artifacts/acceptance/report-final-readiness-20261031/failure-review/034-root-cause-review.json` 及原构造器重放/计量证明
- `artifacts/acceptance/report-final-readiness-20261031/ui034/`
- `artifacts/acceptance/frozen-writer-three-domain-20261031-034/readiness/artifact-tools/exports-v1/verification-summary.json`

稳定入口：[034 终态说明](http://127.0.0.1:8020/frozen-writer-three-domain-20261031-034/review.html)、[034 原始 Benchmark](http://127.0.0.1:8020/frozen-writer-three-domain-20261031-034/benchmark.html)、[034 正式草稿](http://127.0.0.1:8000/flow-demo?run=archive%3Afrozen-writer-three-domain-20261031-034-few-shot-metric&view=report)。

新增 `review.json/csv/md/html` 仅为原终态的独立说明，不发新请求或重评。97 个公开 leaf 字段逐值一致，HTML 使用同一 JSON；已实际检查桌面/390px、中英文与键盘切换，viewport 恢复，控制台无 error。原 `benchmark.*` 保持不变，未提交候选的局部结果不冒充正式稿评分。下一步离线设计复核另存 `failure-review/034-followup-design-review.json`，尚未实现或声称通过新反馈协议。
