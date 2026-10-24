# Responses 三域真实验收010

**严格交付0/3，质量目标未完成。** 本轮只有few-shot引言发出1次请求，服务返回HTTP400：未知字段 `background`。没有生成正文，也没有Reviewer、Evaluation或修复；ViT、NeRF按技术停机规则未发送。该结果不能用于判断模型科学能力或新接口生成质量。

- 源码：`690778d88e1f6b830deb0224b814c36252022f9b`；冻结身份：`ad23cfa8bf70b85a8842b804a8c417682f1a71d47d8f15eb01cdc660ec216797`，481源码。
- 同一grounded-writer-v1、完整三域20/20/18卡和9/12/11篇；writer_reviewer_low_responses_v1。
- 原每题500000 tokens/64请求、总1500000/192、两共享修复、零重试，Judge/Jev关闭。未降低门槛或截断材料。
- 服务未返回usage。账本显示新增已知0且保留56129未知保守占用；不能声称实际不计费。旧1030414未知也没有清零。
- 累计753请求、4238381已知tokens、1086543未知保守占用。Jev67、Judge0不变。
- 原36保护文件、009终态及所有旧费用不变；原请求、响应只保留本地，不公开完整论文或HTTP原包。
- LangFuse只读核对实际1个generation、usage不可用、输出字段和科学IO为空；不声称完整链路或精确成本。

[HTML](benchmark.html) · [JSON](benchmark.json) · [CSV](benchmark.csv) · [Markdown](benchmark.md) · [脱敏传输诊断](transport-diagnostic.json)

下一步先校对火山方舟实际支持字段与SDK兼容范围，并验证未知占用的连续继承；不能通过直接去掉字段后重发010、清空未知额度或额外请求到通过来处理。Phase14.5保持in_progress。
