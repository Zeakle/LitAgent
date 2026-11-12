# 三域冻结正文验收

计划3题，运行1题，生成1题，Evaluation完成0题，严格交付0/3。状态：stopped。

源码 `53a821a4f83ce0ad4e04001ebf49affeb9566f76`，协议 `grounded-writer-v1`；Lite审阅/评估，Pro写作，Judge/Jev关闭。固定证据到交付，不包含新检索或提取，不是端到端检索Benchmark。人工校准未验证，不报告科学准确率。

|题目|生成|Evaluation|严格交付|支持/不支持/未核验|原因|
|---|---|---|---|---|---|
|few-shot-metric|Yes|not_complete|No|0/0/116|evaluation_assertion_capacity_exceeded|
|vit-data-efficient|No|not_run|No|—/—/—|not_started_after_stop|
|nerf-foundations|No|not_run|No|—/—/—|not_started_after_stop|

新增已知用量89534 tokens，未知保守占用0，4请求。历史用量独立保留，不重复计算提取费用。失败和未运行题不移出分母，不追加到通过。

[JSON](benchmark.json) · [CSV](benchmark.csv) · [HTML](benchmark.html)

## 本次真实终态

few-shot 首次完成本版完整正文、摘要/结论、Reviewer 及独立断言抽取。正文与摘要均通过原 JSON/schema 和本地来源绑定检查；这只能证明该次协议完成，不能证明一般稳定性或科学正确。

独立抽取处理了 62 个宿主正文片段，产生 **132 条模型断言，超过既定 128 条上限**。其中 116 条暂标为 scientific、14 条 runtime_method、2 条 report_structure；分类忠实性尚未经过第二阶段复核，后 16 条不能被当作已通过或为了进入容量而删除。另有 68 个宿主披露单元单独处理，不是这 132 条的超量来源。

没有开始证据支持批次或全局一致性检查，116 条暂标科学断言均未核验，支持率不可用。Reviewer 提出 13 项问题，不能代替完整 Evaluation。修复请求为 **0**；`writer_assessment_incomplete` 是未发送的预检终态，不是发生过一次付费修复。

已按技术停止规则保留 few-shot 草稿、原评分、原始响应和失败；ViT、NeRF 未发送请求，没有虚构另外两份正文。严格交付仍为 **0/3**，没有补充样本、重发或自动启动十五题。Judge/Jev 均关闭。

本次新增 4 请求 / 89,534 已知 tokens，用量完整、未知新增 0；连同所有历史累计 **731 模型请求 / 3,943,288 已知 tokens**，历史未知保守占用 1,030,414 保持不变。上限和预留不当作实际费用，历史提取不重复计费，冻结写作观察不虚增完整 Survey 初始轮次。

报告 SHA-256：`92604f969eac3500e88c0e17ca1427202596bcfb8543427491a6b9d62db590b3`。执行源码为上方 `53a821a`，464 个冻结源码文件已核对。只读展示后来升级为 `frozen-writing-view-v3`，从该 SHA 的归档读取真实原因；原 `state.failure_code=evaluation_not_completed` 保持，不追溯修改执行结果。

执行前默认离线 4,226 项通过；后续诊断/导出修复默认 4,233 项通过（25 deselected，8 既有 warnings），只读展示不会触发模型。JSON、CSV、Markdown、HTML 使用同一结果行；公开包不包含原始响应、完整来源或凭证。人工准确率及误放行率不可用。

本机 [LangFuse trace](http://localhost:3000/project/litagent-local-project/traces/02eaff77990f4a94fda901c529bc2f3a) 读回 4 个 generation / 89,534 tokens，科学输入输出为空；不声称已验证完整 trace 树。正文、完整评估输入和原始响应仅保存在本地。

当前剩余问题是正文复合断言与核验容量不对齐，以及尚未验证的事实/分类质量。不能提高容量、裁掉断言、删去核心问题或把缺材料声明当作科学答案来使报告通过。Phase 14.5 保持 `in_progress`。
