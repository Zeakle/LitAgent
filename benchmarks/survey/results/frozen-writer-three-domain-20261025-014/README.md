# 三域真实验收014：引言段落层级错误

**严格交付0/3，质量目标未完成。** 用户确认账户恢复后执行本轮；few-shot首请求HTTP200/completed，原始JSON可解析，但引言直接输出7个`text`/`fact_card_refs`单元，省略规定的`sentences`段落层。原schema拒绝它，恢复器因无法证明段落归属在追加请求前拒绝。没有补JSON、猜分段或放宽schema。ViT和NeRF按技术停止规则未请求，保留在计划三题分母。

- 冻结源码`a2d9f0586e88f55d945fe1a63cbf5a3f5b2e1c50`，486份源码；identity`d06c62b021ee5aef4a01a3b3154bca3f2377beef13e282a3a96946eda6c8d39a`。
- 原三域20/20/18卡、9/12/11篇、完整问题/原文/有效配置保持；15完整父账本和52历史保护。profile为`writer_reviewer_low_responses_reemit_v2`，Lite/Pro、Judge/Jev关闭。
- 每题500000 tokens/64请求、总1500000/192、两次共享修改、零重试、科学1.0与完整覆盖/全局门禁不变。本次实际修复0次。
- 1次Pro请求、11184已知tokens（9208输入＋1976输出）；其中1389思考tokens已含在输出内。新增未知占用0，用时约69.1秒，未触20k输出或300秒超时。
- 原生schema及strict=true实际发送，供应商回显相同。完整请求与012本地契约通过的引言请求逐值相同，本次却违反层级；不能以严格包装声明或一次成功推断响应总能遵守schema，也不能断言供应商内部根因。
- 原响应1098字符，SHA`c2fff5eb9617f19062a4b81821629852eaf4cf4df86cd123d8282a4059001b6c`；7个引言条目均缺`sentences`。没有完整正文、摘要、Reviewer或Evaluation，科学支持分数不可用，不计成科学零分。
- 累计762请求、4334574已知tokens、1142672保守未知占用；旧013欠费请求及其56129占用完整保留，Jev67、Judge0不变，没有退款或重置。
- 完整原包经安装SDK→Synthesis离线逐字重放，同一归属拒绝、0修复、完整11184历史用量保持；重放1次模拟HTTP、0新增外部请求，不计作新费用或科学核验。
- LangFuse读回1个generation、11184tokens/思考明细及空科学IO；不声称完整业务链覆盖。

[HTML](benchmark.html) · [JSON](benchmark.json) · [CSV](benchmark.csv) · [Markdown](benchmark.md) · [脱敏协议诊断](protocol-diagnostic.json) · [恢复边界与离线证据](../../../../docs/native-reemission-review-20261025.md)

014已终止，不追加到通过、不自动15题。本次只证明账户已能接收请求和本地拒绝机制有效，未证明正文科学可靠。后续先离线审查现有段落/句子指令与原生schema的接线，不新增业务输出协议或猜测原文段落。Phase14.5保持`in_progress`。
