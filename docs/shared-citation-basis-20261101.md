# 共享引用集合与支持依据：040回归修复

日期：2026-11-01；Phase 14.5 / 步骤278。状态：离线修复与验收完成，正文质量目标未完成。

## 问题与最小修复

040真实运行完成了正文、摘要和Reviewer。清单恢复后的首个支持批次有三条响应同时给出unsupported、attribution_mismatch和全部必需aspect为true，且有合法basis，宿主因此拒绝。原文单元的复合事实共享实际引用集合；模型解释要求集合中其他论文也分别支持当前原子断言，混淆了许可集合与充分支持依据。

本次只改 `src/litagent/eval/scoped.py` 的显式 `SUPPORT_VALIDATION_INSTRUCTION`：每条断言可从其allowed_support_ids中选取足够basis子集；集合中未用于该断言的共享来源，本身不构成错误归属。没有增加可用来源，也不许可借用别的片段或同论文其他记录。

schema、validator、一致性拒绝、来源归属、条件/数值/百分比与百分点、科学分类及不猜拼写别名的规则保持。原三条矛盾响应仍拒绝；拼写守卫仍阻止追认原记录。单独controlled positive只证明宿主可接受充分子集，不是人工金标准或真实模型正确率。

提示全文继续派生support identity并进入fingerprint、envelope、账本和checkpoint；040旧身份不能恢复为新版本。旧None、v5/v6及未启用该规则的v7语义不变。

## 实际离线证据

最终封存572文件，source digest `ecf518ae043ff3fe22327ed8bce440dedadf1b95978d991b833f36af531f2ddb`；支持语义identity `7e8f6ca1cf1f1d0b16867a69cd8da7fb6ccd630a1a97330e9b9719f5beb0f5d0`。

- 最终相关回归92项实际exit0，包含真实三条回放、充分子集、错论文/实体/条件/数值/拼写/引用权限反例，以及SDK发送与新旧身份恢复。
- 12项完整资源verifier已全部实际exit0，90组合、54次完整Evaluation预检及6条最大合法资源路径完成；69个独立超限反馈组合仍按预期拒绝，不伪称所有材料组合均可发送。
- 原失败144条ViT向量完整保留，最终9批、最大请求预留99995≤assessment100000。global最大114988，output16384与上下文限制保持。
- 三域仅重新构造离线计划：444/448/448万，合1340万，分别低于450万/64请求；仍含两共享修复及完整复评。预留不是实际费用，本步0新增模型调用，无新套件或新正文。
- 最终默认 `pytest -v`：6467 passed / 25 external deselected / 8 existing warnings，实际PID7737、exit0，pytest 413.16秒、控制器418.536秒，前后572源码一致。Black/isort及diff检查实际exit0。

## 过程失败与保护

首轮提示649→716字节跨过单请求硬限，自动拆批后144条向量剩余完整流程不够，maximum001真实exit1。之后仅缩短没有协议消费者的重复标题到682字节，所有科学及子集正文逐字保留；重新封存后完整重跑，未扩大上限或删原文/输出。

default001的6466项通过，另1项因沙箱禁止socket.bind而在PDF loopback夹具setup报错，实际exit1。原测试在允许loopback绑定后单项实际通过，最终默认回归使用同一测试，不跳过、不改断言。回放helper的wire/native映射、私有审计的canonical digest和input_identity误判也保留原失败，并记录到SUGGESTION；不能据此修改产品或来源SHA迁就审计。

16394个旧保护文件及首轮708个新证据逐SHA不变，包含040原报告、评估、费用、SHA和原失败。界面/导出代码未改，无新报告；277桌面/390px、中英文、键盘、引用、附录及六格式证据只适用于原040草稿，不能当成新科学验收。

## 结论边界与下一步

本步不能宣称真实正文支持率提高或三域通过。040仍是有效Evaluation 0、严格0/3；本次不恢复040、不覆盖旧评分或自动追加付费样本。Phase 14.5保持in_progress。完成全部离线收尾后，下一步另行固定同源版本和原三域材料，沿用已有有界授权执行独立真实验证；科学失败如实保留。

私有可复核证据位于 `artifacts/acceptance/shared-citation-basis-20261101/`：原始fixture provenance、两轮source seal、过程日志/实际exit、资源矩阵、独立review及历史保护清单。

最终闭合校验实际PID9090、exit0：572最终源码、16394旧保护及708首轮证据保持；默认、资源及四后处理均绑定同一源码。`final-002-proof.json`明确科学目标未完成，`final-verifier-002-process.json`保存真实退出。未commit/push。
