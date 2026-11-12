# 步骤265 / 036：混合清单恢复后的三域真实验收

本次真实运行已经结束，结果为技术停止，严格交付 **0/3**。这不是科学支持率为0；本次尚未进入摘要、Reviewer或Evaluation。Phase14.5保持`in_progress`。

## 执行范围与终态

用户确认具体端点、Pro/Lite模型、公开问题与原文、可信检索记录和新生成正文外发后，启动已准备的独立036。前一次自动审批审核在创建进程前拒绝的记录保留；收到后续“同意你上述的要求”后提交同一有界启动，没有重新prepare、另建根或绕过审核。

- 清单：`b176a5b2fe480eea8edb73f0356edb3a75180c9a66bffca5507b2080960c5d3b`。
- 原三域材料、552受验源、36费用父记录及10458保护文件核对通过。
- 计划444/447/447万tokens，合1338万；逐题64、套件192请求上限，两次共享修复、零传输重试，思考/Judge/Jev关闭。
- 实际子进程27896，wrapper27878，控制句柄74346，控制器实际exit0、67.967秒。exit0表示成功保存可解释终态，不表示模型或科学通过。
- few-shot首章引言4单元完成本地接收；第二章分类收到7单元后被拒。ViT、NeRF因技术停止未发送。

| 题目 | 已运行 | 完整报告 | Reviewer | Evaluation | 严格交付 |
|---|---:|---:|---|---|---|
| few-shot | 是 | 否 | 未运行 | 未运行 | 否 |
| ViT | 否 | 否 | 未运行 | 未运行 | 否 |
| NeRF | 否 | 否 | 未运行 | 未运行 | 否 |

固定3题分母保持，未追加运行到通过。没有摘要、共享修复或混合清单恢复调用，不能称新恢复已真实走通。

## 精确根因

分类章是完整合法JSON，提供方返回`completed/stop`；第6单元的`fact_card_refs`为`["c1","c9","c18","c1"]`，其中`c1`重复。宿主完整schema的`uniqueItems=true`准确拒绝，原错误位置为`units[5].fact_card_refs`，外层根因为`protocol / writer_flat_schema_invalid`。

实际发送的提供方schema虽标记`strict=true`，但引用唯一性仅在description中描述，没有可执行`uniqueItems`关键字。实际响应能通过发送的原生schema，不能通过宿主完整schema。因此不能称非法JSON、预算耗尽、科学事实判错，或宣称提供方违反了其可执行唯一性schema。内部账本`strict_worker_failed`只是失败保护记账，不能替代外层根因。

原始分类响应SHA为`cab2d1f22fd482ced6592f269655cbbd6e4298dc7a86565dcebcd70bc4170ed7`。独立审计通过实际HTTP receipt、原始请求schema和真实host parser只读复现了同一拒绝；没有去重、改raw、追加引用、接受候选或追认成功。首章原raw SHA为`ee317856327782a0aa6af6458e12bb642de4ffb2ced2879cc8871c994f98520e`。

## 用量、预算与可观测性

| 服务 | 实际请求 | 已知tokens | 新未知占用 |
|---|---:|---:|---:|
| Pro 2.1 | 2 | 23283 | 0 |
| Lite 2.1 | 0 | 0 | 0 |
| Judge / Jev | 0 | 0 | 0 |

两次分别11163、12120tokens，均Responses、20000输出上限、思考关闭，reasoning tokens为0。operation、实际HTTP参数与wire hash、原响应usage及ledger结算逐项一致，每次只结算一次。计划额度不是实际消费，未发生预算规划违约；完整后续流程仍因提前技术停止而未验证。

旧累计931请求/6871081已知tokens增至 **933请求/6894364已知tokens**；旧1204842未知占用保留。旧Jev67请求、756769保守输入tokens及其usage不完整状态单列，未重置历史成本，也不将tokens换算成未经验证的精确货币成本。

本机LangFuse只读一次读回2条generation，模型/API/23283tokens与本地事件一致；generation输入、输出为空，完整消息未外发。没有新增模型请求用于该读回。

## 展示与导出

JSON、CSV、Markdown、HTML同源；CSV210字段、三题分母、阶段、预算、费用及原因逐项对齐，README与Markdown字节相同。结果页为：

[036结果](http://127.0.0.1:8020/frozen-writer-three-domain-20261031-036/benchmark.html)

现有8000只读查看器最初未缓存036，新增与原归档完全相同的字节后诊断入口可用，11个旧缓存文件未变，没有重启服务或启动任务。质量页显示“写作校验失败；Evaluation未运行”，科学指标为未核验横线，Judge/Jev关闭；r1占位SHA不当作完整科学报告SHA。

已实际核对桌面1280和390×844、中英文`document.lang`、可见菜单路径的Enter语言切换/Escape关闭及焦点返回、无全局横向溢出，并查看最终截图。初次移动截图的过渡尺寸和越过抽屉直接聚焦导致的无效截图保留为interim，未计通过；最终使用`*-settled.jpg`。这不是完整无障碍审计，也不冒充新报告的引用/附录验收。

本次没有完整正文；三题×Markdown/PDF/Word/BibTeX/RIS/CSL JSON均为`not_available`。未创建替代报告、未渲染PDF/Word，也未把旧035报告作为本次导出验收。

## 离线证据与完成边界

本步不改业务源码、配置或测试；沿用与当前552文件逐项相同的最终5817通过/25外部排除/8既有warnings、实际exit0及完整最大请求证明，未重复计作新测试。历史报告/费用/失败/SHA均保留。

三个结论分别为：摘要契约本次真实尚未到达；工程预算与终止记账可核对，但完整Reviewer→Evaluation→混合恢复→复评闭环未证明；科学质量未核验，严格交付0/3。正式15题、默认推广和Phase14.5完成条件均未达到。

后续离线审查应处理提供方可执行约束与宿主引用集合语义的差距，明确表示层错误与科学错误的责任边界；任何接收行为改变须版本化、保留原036拒绝及来源/预算边界，不用删项追认这次成功。未启动新的付费尝试，未commit或push，Plan.md保持。

## 证据入口

私有目录`artifacts/acceptance/mixed-inventory-recovery-live-20261031/`：

- `authorization.json`、`action-scope-confirmation.json`、`auto-review-rejection.json`：范围、明确确认和前次进程创建前拒绝。
- `036-preparation-proof.json`、`036-gates-proof.json`、`gate-review/gate-proof.json`：实际准备和独立启动门禁。
- `controller-start.json`、`controller-result.json`：真实进程和exit0终态。
- `terminal-review/terminal-proof.json`：真实HTTP/schema/host parser/唯一结算/历史保护的独立审计。
- `artifact-tools/audit-v1/terminal-proof.json`、`review.md`：四格式统计与18项导出不可用。
- `trace-proof.json`、`ui/ui-proof.json`、`ui/viewer-copy-proof.json`：脱敏trace、实际浏览器和只读副本。
