# 029真实验收与终态

## 结果

029身份为 `83f4ef602c4fdf4512a1d8aa17be49b4f0bde43d27358baa8b3c723774febe77`。按既有授权固定few-shot→ViT→NeRF各一次，原公开问题、20/20/18卡及完整来源与028逐值一致，534份源码冻结，29个费用父链与3642历史保护发送前通过。计划443/446/446万tokens，合计1335万/192请求，不恢复028。

首题完成48正文单元、4摘要/结论单元、Reviewer、106项断言清单（100科学）、7批支持核验及全局检查；107个原文单元均处理。新锚点响应JSON/类型/身份合法，2个finding正确恢复为3个原文位置，旧a41双父定位问题没有重现。

全局检查发现3条非原子断言，整体Evaluation无效、不可交付，修复0/2。CLI仍把新v2的清单质量失败记作protocol并停止，ViT/NeRF未运行，保留在三题分母，严格0/3。本次原终态不改写。

## 失败原因与责任边界

失败发生在Evaluation的报告断言抽取，不是证明论文证据账本的所有事实错误。被指出的复合断言包括：

- a80同时包含K-tuplet、Proto-Triplet/ICNN的机制、设计目标和边界，还合并实验主张；
- a85同时包含MSENet的动机、多阶段机制与额外部件代价；
- a90合并三种方法的训练源、目标域和覆盖差异。

这些内容需要在保持主体、条件和限定语的前提下分别核验，不能靠删除正文或改成组织语句让其通过。完整输入、原始响应和定位保存在029的 `inventory-findings.json`、`global-response-review.json` 与 `inventory-proof.json`。

局部宿主诊断为50支持、10不支持、40未核验；由于清单无效，这不是最终评分、准确率或人工金标准。支持率门槛没有降低；不能据这组局部数直接断言正文只有50%正确。

分流失败是工程漏接线：246只审查src，漏了Benchmark脚本中两个执行分支与一个导出筛选的v1限定。029后的离线步骤248统一使用既有 `located_findings_enabled`，保留协议/身份故障停止规则；原029仍记录protocol，不回填为成功或假装后两题已运行。

## 用量与保护

新增16请求、251733已知tokens、0新增未知占用、0思考、零传输重试；没有预算耗尽。累计867请求、5947517已知tokens，原1142672未知占用单列保留，历史Jev费用不变。tokens不是货币成本估算。

534运行源与3642保护在终态修改源码前已核对；原冻结副本、正文SHA `15160b8fb578e18226eb482d4b6787db292a0f74b6940623a82c173e9633ce43`、归档、原始state和费用均保持。LangFuse读回16个generation、同用量、科学input/output为空；API为7次Responses与9次Chat。没有宣称整个trace tree均已审查。

## 可浏览产物

- 原始结果：[029](http://127.0.0.1:8020/frozen-writer-three-domain-20261030-029/benchmark.html)。
- 补显示清单状态的[只读诊断视图](http://127.0.0.1:8020/frozen-writer-three-domain-20261030-029-view-v2/benchmark.html)，明确标注无新运行，状态/次数/费用和0/3结论保持。
- 正文及证据：[质量工作台](http://127.0.0.1:8000/flow-demo?run=archive%3Afrozen-writer-three-domain-20261030-029-few-shot-metric&view=quality)。只读viewer保留025–028并加入029，不启用live、独立核验或任务恢复。
- 本地IEEE草稿：`artifacts/acceptance/frozen-writer-three-domain-20261030-029/draft-exports/few-shot-draft.md`，192处引用、9条书目、54条原有元数据警告；原始SHA保持，明确不可交付。未新生成PDF/DOCX。

JSON/CSV/Markdown/HTML统计一致；诊断视图四格式另带来源状态SHA，原始包不覆盖。桌面1280/实际390中英、键盘展开、claim_79原文及其实际引用来源验收通过，console错误0，语言与视口恢复。

后续没有新增付费请求。Phase14.5保持in_progress。下一步应为Evaluation的非原子断言清单建立有界重抽取和完整复评能力；不把清单缺陷交给正文改写，也不绕过完整核验。
