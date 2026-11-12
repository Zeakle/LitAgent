# Responses 三域真实验收011

**严格交付0/3，质量目标未完成。** few-shot引言通过JSON和本地schema，方法分类章返回合法JSON，但`taxonomy`为对象，而请求要求段落数组。宿主拒绝该响应，未生成完整正文；摘要、Reviewer、Evaluation和修复均未执行。ViT、NeRF按技术停机规则未发送，仍保留在三题分母中。

- 源码：`0b900250265a01d5a4fb0a0d1fa9ec269c9409f3`，483源码文件；冻结身份：`ae30c535a7b041bcd682054c82fcfa0996123d84929ca9fd12df93445510c3d5`。
- 使用原三域完整20/20/18卡、9/12/11篇，不重新检索或提取。`writer_reviewer_low_responses_v2`，Pro生成、Lite审阅和Evaluation；Judge/Jev关闭。
- 每题500000 tokens/64请求、总1500000/192、共享两次修复上限、零重试，科学支持率1.0门槛未降低。
- 两次HTTP均200，provider状态均`completed`，实际JSON/schema仍由宿主独立检查。请求带`strict=true`，完整HTTP原包与本地响应逐字一致；不能把HTTP成功当作契约或科学通过。
- 新增2请求、22014已知tokens、0新增未知占用，其中1431思考tokens已包含在输出计数内，不重复计费。约99.5秒；未触输出或超时上限。
- 累计755请求、4260395已知tokens、1086543旧未知保守占用，Jev67、Judge0保持。旧010的56129未知占用没有退款或清零。
- 40个历史保护文件及483个冻结源码副本保持。实际预算快照、原始请求、响应和全文仅留本地。
- LangFuse读回实际2个generation、22014 tokens、思考用量元数据及科学输入输出为空；不声称完整执行树验收。

[HTML](benchmark.html) · [JSON](benchmark.json) · [CSV](benchmark.csv) · [Markdown](benchmark.md) · [脱敏协议诊断](protocol-diagnostic.json) · 故障审计

当前证据证明已移除010的未知字段阻塞，但仍有服务返回schema不符的问题。尚无Evaluation结果，科学支持率不可用，不能计为零分或质量改善。后续先离线重放、核对供应商契约及真实消费者，不恢复011、不补数组后接受、不自动追加付费到通过。Phase14.5保持in_progress。
