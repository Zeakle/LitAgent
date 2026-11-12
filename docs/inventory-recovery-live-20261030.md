# 清单恢复三域真实验收 030（2026-10-30）

本轮确实调用了方舟服务。三题计划仅首题few-shot运行；在容量精简响应违反映射契约后按技术停机规则停止，ViT和NeRF未发送。严格交付为0/3，Evaluation未完成，Phase14.5保持`in_progress`。不能据此计算事实支持率或宣称新清单恢复已真实成功。

## 冻结和准入

- 套件：`frozen-writer-three-domain-20261030-030`，身份`185fdf17d4059d515b13ea3fa10a028c78a60d4e367fc93c761a9a59f49287f5`。
- 最终537源文件与5503默认测试实际退出0的版本一致；72最大请求、六旧计划、三域两次最大恢复模拟和14Python格式证明逐项核对。原材料与029逐值一致，30完整费用父链，4319旧路径保持。
- 显式启用`bounded-inventory-recovery-v1`，旧默认不变；`planned-budget-v7`、`body_summary_v3`、`flat-writer-v1`、`strict_auto_v2`、`academic_v2`保持。
- Pro `doubao-seed-2-1-pro-260915`写作／修订，Lite `doubao-seed-2-1-lite-260915`审阅／Evaluation；端点为`https://ark.cn-beijing.volces.com/api/v3`。思考关闭、零重试、Judge/Jev关闭，无审批节点。
- few-shot／ViT／NeRF计划443／446／446万tokens，合1335万／192请求；单题450万／64硬上限、两次共享修复不变。预留不是费用。

## 首题实际经过

| 环节 | 实际结果 |
|---|---|
| 整章生成及摘要结论 | 58正文单元＋4摘要结论单元，完成 |
| Reviewer | 完成 |
| 独立断言抽取 | 110/110原文单元处理，194条清单，其中189科学断言 |
| 容量分配 | 首次完整支持需要13批；保留后续完整路径后选择192条／12批精简目标 |
| 容量精简 | 使用第1次共享机会，返回合法JSON但跨字段映射矛盾，候选拒绝 |
| 支持核验与全局检查 | 未运行，0条支持判断，未核验 |
| 新非原子清单恢复 | 已启用但尚未走到该分支，真实效果未证实 |
| 最终交付 | 阻止；其余两题未运行，均留在3题分母 |

报告SHA为`c12964218d57202789ed4a966176e46f1915d075dd324c86a2b8dbe4ba1062a7`。失败候选没有替换原稿，历史029及其科学失败、原始正文和费用不变。

## 两层原因

先触发`capacity_remaining_workflow_unaffordable`：当前194条清单不能在冻结额度内同时锁定本轮及剩余修复后的完整复评。程序保留全部清单，进入已设计的精简分支，没有直接裁剪证据或用部分核验宣布通过。当前有效容量192不是配置中新增的硬上限，而是该轮资源计划的结果。

最终停机原因为`writer_capacity_new_unit_unaccounted`。Pro返回38个目标的38个替换单元及58个原单元处理记录。在`r8`（分类章节第二段）返回一个新单元，同时把两个原单元的`replacement_indexes`都置为空，等价于“旧内容全部删除，新内容没有任何原始映射”。真实`normalize_edit_map`只读重放仍拒绝，原始响应SHA为`125a513363ff9b2fa81dee9f87a70cfc7fd2c3d1f4eab1c510e0b98cc6c2423f`。不能凭内容相似猜测应映射到哪个单元，也不能跳过此检查提交候选。

这里是未完成科学核验，不是189条事实被判不支持。预算没有被实际费用耗尽，但完整流程的资源可行性与容量恢复仍限制本轮执行；不能将其表述成预算保障和正文质量均已通过。

## 用量与保护

新增9请求、151470已知tokens、0未知占用、0思考tokens；有效usage均已结算。耗时约409.5秒。累计876请求、6098987已知tokens，历史1142672未知保守占用继续单列。历史Jev67请求／756769估算输入、usage不完整保持，新增Judge/Jev均为0。

537源码副本、4319历史路径、源报告SHA及账本独立复核通过。LangFuse读回9个generation：8 Responses＋1 Chat，usage同为151470，科学输入输出为空；仅核对generation层，未宣称完整trace树验证。原030不恢复，不追加新样本或真实调用，不进入15题。

## 展示与限制

[本轮Benchmark](http://127.0.0.1:8020/frozen-writer-three-domain-20261030-030/benchmark.html) 已在已有浏览器标签页打开；[草稿质量页](http://127.0.0.1:8000/flow-demo?run=archive%3Afrozen-writer-three-domain-20261030-030-few-shot-metric&view=quality) 使用只读缓存，不初始化模型或历史任务恢复。

JSON／HTML对象、CSV计数与usage、Markdown计划和计数一致。发现展示缺口：Markdown及HTML行详情优先展示上游Evaluation容量原因，Markdown没有最终修复协议错误；HTML页首与JSON仍保留最终停机码。该缺口明确记录在`public-export-proof.json`，不修改原导出补成功。

桌面1280、实际390px中英与键盘语言切换、正文实际引用→完整原始支持、附录展开、质量未核验／blocked状态已检查。英文Quality需等待真实标题可见；早期误拍Report的截图另存为中间结果，不作质量页证据。最终语言和视口已恢复。

草稿仅生成本地Markdown检查，167引用位置、9书目、54条既有元数据警告，正文SHA不变。此步没有新业务源码变更，沿用同源5503／72离线证明；未为失败草稿额外生成PDF／Word或重跑整套pytest，也不冒称重新完成这两项导出视觉验收。

## 下一步离线修复范围（251待开始）

1. 使用030真实响应建立精简映射反例，审查扁平patch顺序、目标局部索引及原单元处置，保留当前拒绝行为；不自动补映射，不把未核验内容判成错误事实。
2. 校验58单元／194断言的写作密度与容量目标传递，减少重复展开的同时保留核心方法、条件、数字和具体局限；不提高预算或改变分母。
3. 区分最终停机原因与上游Evaluation原因，四格式及质量页同时呈现阶段和根因。
4. 修复前先同步当前GUIDE，新增真实producer→consumer正反例；若改业务源码，重新封存并执行相关及默认离线验收。此次030保持终态，不在本轮追加付费。

## 本机证据

`artifacts/acceptance/frozen-writer-three-domain-20261030-030/` 中的`manifest.json`、`independent-preflight-review.json`、`acceptance-proof.json`、`terminal-quality-proof.json`、`repair-map-review.json`、`repair-map-replay-proof.json`、`public-export-proof.json`、`trace-proof.json`、`ui-proof.json`以及原始运行归档。原始科学材料和响应只留本地，不打入公开展示包。
