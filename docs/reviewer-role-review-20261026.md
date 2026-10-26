# Lite Reviewer独立思考配置与协议停止保真

[015真实记录](../benchmarks/survey/results/frozen-writer-three-domain-20261025-015/README.md)已经编译五个正文章节及摘要、结论，但Lite Reviewer返回`incomplete / length`：4096输出tokens全部为reasoning，没有审阅答案。输入32853tokens、完整预留104365合法，运行总计149128低于500000；这不是上下文、总预算、超时或JSON解析器问题。Evaluation未运行，严格交付仍为0/3。

## 本次修复

- 新增仅用于服务端冻结验收的固定配置`writer_low_reviewer_disabled_responses_reemit_v2`。Pro正文、摘要及修订保留enabled/low；Lite Reviewer使用disabled并显式清空effort；Evaluation保持disabled。
- 不修改共享AppConfig默认值。Reviewer使用独立配置副本；新参数进入现有manifest、身份校验、实际执行及四格式展示。旧配置名称、字段及默认API身份保持原语义，不能用新模式恢复旧套件。
- 输出上限仍为Pro 20000、Reviewer 4096、Evaluation 16384；每题500000 tokens/64请求，两次共享修改、零重试。原文、输出schema、科学支持率1.0及完整覆盖等门禁保持。
- 修复另一项离线确认的缺陷：两次恢复额度用尽后，第三个协议错误曾被改记为`writer_repair_limit / quality`，可能让套件继续下一题。现在保留原协议错误和技术停止，只增加次数耗尽诊断，不发送第三次修复。普通科学修订的质量终态不变。015没有触发此缺陷，不能把它写成015停止原因。

## 验证与解释边界

[SDK与完整消费者回归](../tests/test_native_responses.py)实际执行Synthesis→Reviewer→唯一报告出口→Evaluation→自动判定，只替换HTTP。测试同时保留科学支持正例、不支持反例和缺少方法记录反例；新模式的Pro请求仍有low，Reviewer请求没有effort，旧模式保持。

[角色身份和调度测试](../tests/test_writer_role_profile.py)检查配置副本、非法参数、重新hash后的篡改拒绝、四格式一致、实际调度参数及终态不重发。[恢复边界测试](../tests/test_native_reemission.py)通过实际SDK先消耗两次共享恢复，再发送第三个坏正文或摘要，确认原协议根因、次数、费用及重启不重发。

另将015全部九份真实原包只读重放：八个Pro请求字节完全一致，编译中间SHA仍为`651a8942fdd9a6632a65d7b5ef64b40e6d25536490a0763d834f94e28b0dcea0`；Reviewer请求只改变thinking及删除effort，完整正文、原文和schema保持。把原来的不完整响应送入新配置消费者，仍然拒绝，不伪造一个真实成功答案。重放149128是历史用量，不是新增账单；本步外部模型请求为0。

私有证明、未核验草稿、完整响应和截图留在本机`artifacts/acceptance/reviewer-role-split-20261026/`；60项历史保护及015的486份源码副本保持。离线配置预览明确标识无模型请求，桌面和390px、中英文、键盘与无溢出已核对。没有新最终报告，不宣称PDF/Word科学报告验收。

默认离线验收最终**4633 passed、25 deselected、8既有warnings**（222.93秒），7个改动Python文件Black/isort及git diff --check通过。首轮旧分类预期失败和本机回环权限问题保留记录，修正测试契约并允许本机测试服务器后完整重跑；没有放宽科学门禁。

该修复保证角色配置和失败分类按契约执行，不能保证模型在4096额度内一定返回合格审阅，也不证明科学质量提升。015不恢复、不覆盖旧结果，累计771请求、4483702已知tokens及1142672旧未知占用保持。Phase14.5继续`in_progress`，后续真实验收必须另建源码与材料冻结身份。

## 016真实验收补记

[016](../benchmarks/survey/results/frozen-writer-three-domain-20261026-016/README.md)冻结fa760bf及原三域材料。few-shot分类、方法两章通过独立逐字恢复；随后综合比较章第三个非法JSON使共享额度耗尽。原writer_json_invalid/protocol根因被保留，未发送第三次恢复，另外两题按规则未运行。这项分类修复获得了真实停止行为证据。

本轮仅6次Pro请求/83749完整已知tokens/0新增未知，没有完整正文、摘要、Reviewer或Evaluation。**Reviewer角色解耦仍未被本轮真实调用验证**；不能把配置出现在清单里当作调用完成。六原包实际SDK→Synthesis逐字重放同一终态，0新增外部请求。累计777请求/4567451已知/1142672旧未知，科学严格结果仍0/3。后续先离线审现有生成契约，不继续付费调参或增加新的输出编码，Phase14.5保持in_progress。
