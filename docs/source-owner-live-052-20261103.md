# 052 固定三域真实验收：评估协议停止，质量目标未完成

2026-11-03，Phase 14.5 / 步骤309。唯一 controller 的 wrapper PID24785、子进程 PID24845 实际退出0；套件终态为 `stopped / evaluation_decision_consistency_invalid`。进程正常收束不代表科学验收通过。完整 Evaluation **0/3**、严格交付 **0/3**；ViT、NeRF 因技术门禁未运行，继续保留在计划分母。

## 固定输入与实际进度

生产源码630项，digest `ec6e6b125a846b87d677c0a06b69d4605fbf24377013dd41610698fbdc46d360`；manifest identity `cbbbdffc50404ca474ae3caf047a813cd49944d07cc8263c2ae31aaa4ee3fdb6`。原三域公开材料、模型、关闭思考、两次共享修复、零重试、Judge/Jev关闭均保持。未重新搜索或提取，未恢复051或追加到通过。

| 题目 | 正文及摘要 | Reviewer | 完整 Evaluation | 严格交付 |
|---|---|---|---|---|
| few-shot-metric | 已生成 | 已完成 | 第4个支持批次被拒绝 | 否 |
| vit-data-efficient | 未运行 | 未运行 | 未运行 | 否 |
| nerf-foundations | 未运行 | 未运行 | 未运行 | 否 |

few-shot 完整抽取157条：142条科学断言、15条报告组织陈述。全部106个原文单元已处理；动态计划由128条/8批扩为157条/10批，保留两次修复，含余量需求391万 tokens、47请求，低于启动时锁定的444万/64。**本次停止不是预算或容量耗尽。** 这是实际安全边界扩批的证据，不代表两次修复、全局检查或全部科学核验已完成。

## 原始失败定位与科学结果边界

第4批（index3）`claim_48`、operation `e7bfd4d49cb54aa6b79276d69b975fcc`，raw SHA `cc4caa0f39a340ed3b2dfa626b7749fafbfc3ea0c50e7cba8ddf30063a44a7b5`。原JSON和schema合法，原文忠实性及分类字段可消费；失败来自以下互相冲突的语义字段：

- `status=unsupported / reason_code=insufficient_support`；
- 必需的 `predicate`、`conditions` 均为 `supported=true`，且完整引用合法原文；
- 总解释指出“并非重新定义推理规则”的否定边界未获支持，predicate解释却只确认正半句。

宿主复现 `negative_status_with_complete_positive_diagnostics`，整批拒绝。既不能按布尔值自动改成支持，也不能根据自由解释自动翻转检查项。原拒绝及原始响应保持。

前三批48条仅是局部诊断：36条科学断言中27支持、6不支持、3未核验，另12条组织陈述。3条未核验分别涉及原子性与多余aspect；原不支持项继续保留。整体评估无效，因此**科学支持率不可评分**；归档兼容包装中的 `score=0 / skipped=true` 不能显示为科学零分或通过。最终142条科学断言均标为未完成核验，不能用局部27/36代替整体结果。

本次共享修复HTTP、候选、提交均为0；尚未到达新V4来源派生修复或V2全局分类分组调用，不能宣称这些真实分支已通过。它们的现有证据仍是308离线真实消费者与资源验证。

## 预算、用量与历史保护

新增12请求、170032已知 tokens、0未知占用，全部有效结算且无重试；prompt139779、completion30253。累计1415请求、14311970已知 tokens，历史1283047未知占用及038的78205 hold原样保留。历史Jev67请求/756769保守输入且usage不完整，Judge0；本次均未调用。预留391万、锁定444万不是实际费用。

费用审 PID26488、科学原文重放 PID26493 均实际退出0。费用当前父只含052，历史50父已认证前缀未在终态审重新执行。源码、原报告/证据/评分/费用、Plan与历史保护须在最终关闭凭证再逐SHA核对。未向远端LangFuse读回；本地trace有12个开始及完成事件与170032 tokens对齐，不冒称远端服务验收。

## 展示与导出

[Benchmark 页面](http://127.0.0.1:8020/frozen-writer-three-domain-20261103-052/benchmark.html)展示计划3、运行1、生成1、完整核验0、交付0；失败与未运行题均保留。桌面1163×818、移动390×844、中英文、键盘切换及诊断展开已实际检查，没有页面级横向溢出；表格使用自身滚动。临时viewport已恢复。首原生面板裁切截图及错误role定位保留，最终QA使用实际标签截图和真实summary元素。

8000工作台对新run显示not found，未重启或改变服务。因此该页的原文/证据点击定位、折叠附录UI **未验收**，不能以Benchmark静态展示替代。

唯一已生成few-shot草稿实际导出Markdown/PDF/Word/BibTeX/RIS/CSL JSON。PID26596导出、26640渲染、26932内容审、27006最终审均实际退出0。PDF7页、Word12页共19页逐页看图，无可见裁切/重叠/缺字；9条参考文献解析、ID/标题/URL/顺序一致。54条来源元数据缺失警告保留，未猜补。原报告SHA `922a7c7b54ad2e009dad29895ed4420f5af3b45d55159cea01b004fa4880cd7d` 不变，所有文件仍明确为不可交付草稿；另外两题没有虚构正文或导出。

## 验证范围与下一步

本步未修改产品或测试源码，默认8123通过/25排除/8既有warnings、15Python Black/isort/diff及最大请求/完整两修复预算证明继承相同630/ec6e的308实际结果，未冒称重新运行。新observer的首红均保留：执行后格式化脚本、路径/费用模块归一、digest helper混用、RIS换行及parser字段猜测；只修观察器并以新实际进程复验，未清洗原raw。

下一步先审评估状态与aspect的重复表达及消费者契约，准备可验证的最小方案。必须保留科学负例、完整核验、旧终态和身份隔离；不能直接放过矛盾字段、将未核验记为通过，或再次付费碰运气。若方案改变既定数据流或技术失败分类，按项目架构确认规则处理。Phase 14.5 继续 `in_progress`。

主要证据位于 `artifacts/acceptance/source-owner-live-20261103/`：`terminal-science-review/`、`terminal-cost-review/`、`export-qa-052/`、`ui-final-052/`及最终关闭凭证。
