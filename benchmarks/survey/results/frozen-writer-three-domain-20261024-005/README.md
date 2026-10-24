# 三域冻结正文验收

计划3题，运行1题，生成1题，Evaluation完成0题，严格交付0/3。状态：stopped。

源码 `a8cb7f725b2de37aa3bb894a6f7a51040bd753c9`，协议 `grounded-writer-v1`；Lite审阅/评估，Pro写作，Judge/Jev关闭。固定证据到交付，不包含新检索或提取，不是端到端检索Benchmark。人工校准未验证，不报告科学准确率。

|题目|生成|Evaluation|严格交付|支持/不支持/未核验|原因|
|---|---|---|---|---|---|
|few-shot-metric|Yes|not_complete|No|—/—/—|writer_final_output_unavailable|
|vit-data-efficient|No|not_run|No|—/—/—|not_started_after_stop|
|nerf-foundations|No|not_run|No|—/—/—|not_started_after_stop|

新增已知用量76146 tokens，未知保守占用0，6请求。历史用量独立保留，不重复计算提取费用。失败和未运行题不移出分母，不追加到通过。

[JSON](benchmark.json) · [CSV](benchmark.csv) · [HTML](benchmark.html)

## 真实停止位置

五章正文与摘要/结论共103个写作单元（94+9）已通过JSON、schema和本地编译；这不是103条已核验科学断言。Reviewer完整请求预留151,731，超过含余量的117,964上下文上限，在发送前拒绝；Lite审阅、Evaluation、修复和最终交付均未执行。

原state仍保存笼统writer_final_output_unavailable。本视图额外展示版本绑定的worker失败和完整输入只读重放诊断，不改原状态、费用或正文。编译草稿SHA为`34e8fca38d54b469c098281a4e320798ac9ae6ccf75801991659adaab6002e4f`，最终报告SHA仍不可用。

请求记录为6次Pro、76,146已知tokens、零重试/新增未知；关闭的Judge/Jev未调用。六份输入逐字绑定后在实际Synthesis Worker离线重放，草稿SHA一致，未增加模型请求。Reviewer payload中的draft为41,820 bytes，units为66,032 bytes，包含重复正文；来源完整保留，没有截断输入或降低输出预留。

同一真实正文的下游请求也做了离线构造检查：假设每个写作片段对应一条断言，并使用全部不支持的脚本响应，仅用于预算审计；独立抽取和七个支持批次均能构造，全局请求预留163,075仍超限。这不是实际科学抽取或核验结果，也不能据此计算支持率。后续修复必须同时覆盖Reviewer与全局材料，不再只修一个局部就追加付费。

468份执行源码和16个当前保护路径保持；LangFuse读回6个generation/76,146 tokens且科学I/O为空，不声称完整trace树已验证。累计738模型请求/4,032,613已知tokens，历史1,030,414未知用量保守占用继续单列。Phase14.5保持in_progress，0/3目标未完成。完整草稿、来源与响应保留本地，不公开未经核验正文。
