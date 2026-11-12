# Phase 14.5：042 固定三域真实验证

本轮实际完成一次新套件启动，随后按协议错误停止。结果为计划3题、运行1题、生成1份草稿、完整Evaluation 0题、严格交付0/3。没有恢复041、追加试跑或重新搜索、提取；Phase14.5保持`in_progress`。

## 固定身份与实际执行

- 源码582份，摘要`c3cd9314db5f8b777e1f6dcd5c87318ca83a9810fdfc2b4366ea11e0a68bb79e`。
- 套件`frozen-writer-three-domain-20261101-042`，manifest身份`0820bc06bba0588e580a4f8bd7b3e18555eab062cda60c9eeb58c310d64c5a1b`。
- 真实wrapper32789／child32831／工具session40624，最终chunk`7c9b47`，进程退出0，耗时520.529秒。进程退出0表示套件正确保存失败终态，不表示质量通过。
- Pro写作、摘要和修订，Lite Reviewer／Evaluation，思考关闭、零传输重试、两次共享修复；Judge/Jev关闭。
- 三题计划444／448／448万tokens，合1340万／192请求；沿用已批准最高每题450万／64请求。
- 同源默认离线6765通过、实际退出0；最大12验证器与4后处理实际全0。真实启动前，准备、独立复核和真实启动器预检均实际退出0。

## 真实结果

| 题目 | 正文与Reviewer | Evaluation | 严格交付 | 结果 |
|---|---|---|---|---|
| few-shot | 48正文单元、摘要1、结论1；Reviewer完成 | 完整抽取182条，支持核验0批，未完成 | 否 | 第一次容量精简的位置映射非法 |
| ViT | 未运行 | 未运行 | 否 | 按技术错误停止规则未发请求 |
| NeRF | 未运行 | 未运行 | 否 | 按技术错误停止规则未发请求 |

首次清单包含175条科学断言与7条组织陈述；110/110原文单元和50/50模型片段均已处理。尚无论文支持核验结果，175项应解释为未核验，不能解释为175项错误，也不能用skipped指标中的占位值生成科学分数。

原稿SHA为`7e9b0fbee145e02945d23ab77ba4ffe1b1b2986abff2478729917591470aec37`，没有提交精简候选，没有正文改写。已消耗1/2共享修复机会。

## 容量处理与直接停止原因

182条清单需要12批。当前完整支持核验与global请求预留873258，已结算129769；两次未来修复及完整复评各1595928。原公式得到保守需求4194883，含10%余量整档为4620000，超过启动锁定的4440000；请求总数53没有超过64。因此进入目标176条／11批的有界精简是正确资源处理。refine计划344万是预留上界，不是实际费用。

精简响应是完整JSON，native schema也接受其形状。模型返回38个target、每个1个替换单元，但48条旧单元映射使用全局patch序号，47条越界。首错`r2 / writer:introduction:1:0`只允许局部`[0]`，却返回`[1]`。实际系统提示已经要求“WITHIN target, not global patches”；parser没有改号。`normalize_edit_map`正确抛出`writer_capacity_edit_position_invalid`，候选未提交。

这是模型编辑元数据的跨字段契约错误。不能据此判断正文科学质量不合格，也不能通过改号、扩大范围或改写042原结果追认成功。原始响应、实际请求、schema、canonical和失败均已保存；离线原guard重放真实退出0，复现相同拒绝。

## 用量和保护

- 本轮新增9次请求，157121已知tokens，新增未知占用0，全部已结算。
- 累计1072次模型请求、9075367已知tokens；历史未知占用1283047完整保留，其中038的78205未释放。
- 本轮准备读取41个历史费用父链，终态只认证新增042凭证，不执行旧科学恢复器。
- 当前582源码、582冻结副本、583冻结材料文件及20762保护文件逐SHA不变；`plan/Plan.md`不改。
- 未commit、push、tag；未停止或替换8000服务，8020原静态服务保持。

## 展示和导出验收

JSON、CSV、Markdown、HTML同源保留3题计划分母、1运行、2未运行和0交付。8020真实浏览器完成1280×720与390×844、中英文、`document.lang`、Enter开闭诊断与Tab焦点检查；表格在自身容器横向滚动，移动页面没有横向溢出，临时viewport已reset。

few-shot完整草稿经原生producer导出Markdown、PDF、Word、BibTeX、RIS、CSL JSON，全部仍为`publishable=false`。PDF8页、Word14页均逐页打开检查；109处正文数字引用逐项同序，书目1–9一致，三种文献数据与同一snapshot一致，54条原元数据警告保留。标题精确文本核对完整，未改原文、评分、SHA或原始元数据。

8000的既有只读缓存未刷新，新042工作台原文链接的真实浏览器定位未验收。此前自动审批拒绝服务切换，具体切换授权仍未收到明确答复；没有更换端口或服务绕过。此项不冒充已完成。

## 结论与下一步边界

1. 生成、摘要和Reviewer本次完成；280状态配对修复尚未进入真实支持批次，不能宣称已获真实验证。
2. 容量规划与有界精简入口实际触发，费用和保护闭合；完整两次修复及最终复评的真实路径仍未完成。
3. 科学质量仍未核验，严格交付目标未达到。

下一步先离线收敛精简编辑映射的目标内位置契约，覆盖至少两个不同target均从局部0开始的实际发送例，继续拒绝原042全局序号响应；不自动改号或放宽候选提交门禁。生产修改前另行同步当前GUIDE与同源验收要求，不在042中途改源码，不追加本轮付费调用。

## 本地证据

- `artifacts/acceptance/evaluation-status-live-20261101/launch042/controller-result.json`
- `artifacts/acceptance/evaluation-status-live-20261101/terminal042-budget-audit/audit-001-proof.json`
- `artifacts/acceptance/evaluation-status-live-20261101/terminal042-position-review/position-review-003.json`
- `artifacts/acceptance/evaluation-status-live-20261101/terminal042-position-review/capacity-chain-001.json`
- `artifacts/acceptance/evaluation-status-live-20261101/launch042/ui042/browser-qa-observations.json`
- `artifacts/acceptance/evaluation-status-live-20261101/launch042/exports/few-shot-metric/exports-receipt.json`
- `artifacts/acceptance/evaluation-status-live-20261101/launch042/exports/few-shot-metric/docx-visual-review-root.json`
- `artifacts/acceptance/evaluation-status-live-20261101/launch042/exports/visual-review/pdf-visual-review.json`
