# Benchmark清单质量失败分流：步骤248

029实际响应已通过新锚点定位，但Evaluation清单存在非原子断言。调度脚本三个旧v1限定分别控制诊断字段记录、质量失败继续执行与四格式诊断面板，漏接新v2后会误记为protocol并中止后续题。此前只覆盖src及报告worker，没有覆盖最终CLI消费者，这是测试缺口。

本次只修改 `scripts/run_frozen_writing.py`：三个分支统一调用既有 `located_findings_enabled`，不新增协议、模型、Agent、预算或免检类别。合法host_validated清单质量失败可继续后续计划题；非法JSON、伪造身份和未知marker仍停止。失败报告仍不可交付，不改正文、评分或批准。

新增 `tests/test_benchmark_inventory_dispatch.py` 消费真实SDK→Synthesis→Reviewer→Evaluation产生的报告，再通过实际CLI调度、状态落盘和四格式导出。host manifest/生命周期与费用使用隔离夹具，缓存同一真实生产链结果用于测试分流；这些零host计数不作为真实费用证据。覆盖v1/v2继续、非法JSON、未知身份、未知marker停止，以及CSV字段和HTML诊断可见性。

最终默认 `pytest -v`：**5470通过、25排除、8项既有warnings，实际退出0**；535受验源保持，47相关通过。2个Python文件Black/isort、git diff --check通过；54最大实际请求向量通过，六份旧v5/v6计划不变。4204条既有路径保持，029冻结534源码副本与原终态、报告SHA、费用不变。

源码修改发生在029终态验证之后，不能把本修复描述为029运行时已生效。原029仍0/3，16请求251733已知tokens，修复后没有新付费。额外诊断视图明确不是新运行，没有改变原protocol记录。

证明目录：`artifacts/acceptance/benchmark-inventory-dispatch-20261030/`，主要为 `completion-review.json`、默认pytest证明、54组探测、`step248.diff` 与历史保护清单。真实运行证明在029目录；工程与科学结果分开。

Phase14.5继续in_progress。分流已修复，但非原子抽取的有界恢复与剩余科学问题仍未解决，不以测试通过替代正文质量达标。
