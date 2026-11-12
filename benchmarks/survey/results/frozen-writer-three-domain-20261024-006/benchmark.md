# 三域冻结正文验收

计划3题，运行1题，生成0题，Evaluation完成0题，严格交付0/3。状态：stopped。

源码 `2c1b4bb3dd6b56ca51188404db7c174876a72e46`，协议 `grounded-writer-v1`；Lite审阅/评估，Pro写作，Judge/Jev关闭。固定证据到交付，不包含新检索或提取，不是端到端检索Benchmark。人工校准未验证，不报告科学准确率。

|题目|生成|Evaluation|严格交付|支持/不支持/未核验|原因|
|---|---|---|---|---|---|
|few-shot-metric|No|not_complete|No|—/—/—|writer_json_invalid|
|vit-data-efficient|No|not_run|No|—/—/—|not_started_after_stop|
|nerf-foundations|No|not_run|No|—/—/—|not_started_after_stop|

新增已知用量76416 tokens，未知保守占用0，6请求。历史用量独立保留，不重复计算提取费用。失败和未运行题不移出分母，不追加到通过。

[JSON](benchmark.json) · [CSV](benchmark.csv) · [HTML](benchmark.html)


## 停止位置与解释边界

五章响应均通过本地 schema 和完整正文检查，但摘要／结论请求返回非法 JSON，未形成最终报告。summary_initial 原响应 1,085 字符，offset 1,085／column 1,086 报缺少分隔符，finish=stop；实际只使用 555 输出 tokens，未达到 20,000 上限。conclusion 键出现在尚未结束的 abstract 容器内；不接受补括号或删约束修复原响应。原始文本与账本保留本地。

Reviewer、Evaluation、修复、Judge/Jev 均未调用，支持数量和真实原子断言数量不可用，不是科学零分。另两题按技术停机规则未启动，严格交付0/3，失败和未运行题保留在分母。

源码2c1b4bb/469文件，20个历史保护路径一致；6次Pro/76,416 tokens，零重试／新增未知。累计744模型请求／4,109,029已知tokens，旧1,030,414未知保守占用、Jev67及Judge0继续独立保留。此前输入预算修复只有离线验证，本组尚未实际进入Reviewer，不宣称其真实效果已通过。

[原响应位置诊断](protocol-diagnostic.json)。Phase14.5保持in_progress；本组终止，不续跑，不追加到通过。

LangFuse只读核验6个generation／76,416 tokens、科学I/O为空；不声称完整trace树验收。[本机trace](http://localhost:3000/project/litagent-local-project/traces/389f6ebb15521a522a46283bc680f2f6)。
