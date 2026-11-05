# 跨阶段费用凭证离线验收（2026-11-01）

Phase14.5 / 步骤272仅完成离线工程验收；正文质量目标未完成，最近真实结果仍严格0/3。本步没有网络或模型调用，没有创建039、恢复038、补发未知请求或新UI/报告导出。

## 问题与修复边界

把038加入历史父链时，旧writer-only认证在ViT第8个已知inventory请求即报frozen_accounting_planned_writer_binding_ambiguous_or_missing。实际有7writer/Reviewer和7Evaluation已发送请求，末assessment因HTTP403/AccountOverdueError缺少usage，78205预留必须保留；未发送的futureglobal不是费用。

增加显式carry_planned_mixed_unknown_v2，仅用于现有验收CLI的ROUND_PROFILE。prepare依赖检查发生在读取材料/创建目录前，live不能覆写；旧None、v1、38父收据和预算/生成身份保持。新认证闭合真实archive start/terminal、writer journal、Evaluation精确checkpoint、完整发送wire、ledger、round/recovery。原文、候选或反馈未持久化时不虚构身份前像。

runner仅为本地checkpoint和evaluation.batch事件增加session_id；相同冻结输入的writer identity可能跨run相同，不能将成本哈希当强归属。未来v2父强制两处owner一致；038旧initial仅凭已有完整survey精确重算旧identity，不追填旧文件或借其他run。LLM请求与原identity公式不改。

## 同源实际验收

- 最终567源码，digest `76d070e65d4350c113e127864c5226d32474e127e49223594f32ebb5d8a77b53`，前后逐文件一致。271完整566和四plan已保存。
- 默认`.venv/bin/python -m pytest -v`：6298 passed、25外部deselected、8既有warnings，PID72690，368.469秒，实际exit0；本机已有导出fixture获允许监听127.0.0.1临时端口，不跳过测试。
- 新56测试与旧72相关测试共128 passed；新测试使用tmp_path与公开两论文材料、真实SDK/producer/checkpoint/consumer，仅外部HTTP Mock，不依赖私人038目录。未额外实际建立clean checkout。
- 独立真实038复制件12反例和入口7反例actual0，配置/fingerprint/owner/wire/progress/usage/round等篡改拒绝；旧v1函数逐字保持并继续拒绝mixed。
- 本次3个Python文件Black/isort、git diff --check实际0。全量源码未因格式扩散。
- 最大12 verifier、summarize和verify_final全部实际exit0：90向量、54Evaluation预检、6最大资源路径、18旧计划及038三计划保持。三新计划444/448/448万、合1340万，各41计划请求/两次共享修复；这是预留，不是实际费用。69个聚合repair子例仍正确资源拒绝，不宣称全部可发送。
- 最终同源39父链纯文件重算actual0：1010请求、8082678已知tokens、1283047未知保守占用；旧38收据逐值相同。Jev历史67请求/756769保守输入、usage_complete=false，Judge0，单列保留。038仍48请求/721310已知/78205未知，其中mixed ViT14请求，不退款不重发。
- 14444保护文件SHA零变化。

## 证据与范围

根目录：`artifacts/acceptance/support-receipts-live-readiness-20261101`。`completion-proof.json`及`default-001-proof.json`记录实际退出；`source-seal-001.json`与`source-after-001.json`精确同源；`maximum-001/execution-status.json`保存全部PID/真实退出。最终账本重放为`review-tools/mixed-replay-002.json`，不是仅两文件中间source的001。独立审计见`mixed-review-002/review.json`和`entrypoint-review-004/proof.json`。所有私有首错保留，未改成产品通过证据。

成本认证和离线绿不等于科学通过；本步不更改coverage.complete=false时不得科学改写的门禁，不新做UI/导出验收。账户恢复事实仍待明确；持续有界授权保持。下一步可仅准备039冻结清单并独立审计，不能据此自动启动真实请求、追加到通过或推广工作台默认。
