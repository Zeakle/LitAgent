# 步骤264：混合清单恢复与请求契约离线验收

本步只完成离线工程修复和验收。没有恢复035、准备036或新增模型调用，035仍为严格0/3，Phase 14.5保持`in_progress`。不能把以下Mock传输、请求构造、账本模拟和测试结果当成新版本的真实科学通过率。

## 修复内容

新增内部显式`mixed_inventory_recovery=True`，仅配合v7和原清单恢复入口使用。新envelope选择`bounded-inventory-recovery-v2`；普通Survey API不增加字段，默认False不序列化。旧v1、旧v5/v6/v7计划及checkpoint保持原语义；新规则及回复/全局提示身份进入新运行、预算和请求身份，不能用新规则恢复035。

新恢复先验证同一报告SHA、完整宿主原文/fragment范围、全部处理记录、实际引用、旧清单唯一ID以及已结算请求。支持处理混合`non_atomic`、`classification`、`source_fidelity`和`missing_assertion`线索，保留每条原始代码、解释及batch/global来源。它们只是未决线索，不是自动改稿命令；不按“无遗漏”等关键词删除或反转判定。

恢复沿用现有Evaluation完整重抽报告、摘要和结论，再执行全部实际引用支持核验及global。清单变动不修改正文或证据；相同内容、只改ID、循环、遗漏或分类逃避不能获交付。清单恢复与容量精简、Reviewer及科学改写仍最多共享两次；恢复有效但科学失败才可使用剩余机会改正文、更新摘要并完整复评，失败候选不提交。未知usage保留占用，重启不重发。

传输中补齐了原先遗漏的消费者：v2的复数`source_fragment_ids`、旧断言目录和实际EID→eN映射。pN仅为上次清单的线索，vN对应本次原文单位，不能充当新输出字段或扩大引用权限。宿主必须提供真实别名目录，不按相同文本、相邻位置或zip顺序猜映射。

正文、摘要和patch的回复布局提醒由本次实际wire schema派生，不增加模型输出字段。ViT原035完整响应中的多余空根`fact_card_refs`仍然拒绝，不删字段、不扩schema、不付费重发；新诊断给出实际多余/允许根字段、阶段和schema身份。旧035真实请求74511的消息、schema及拒绝结果逐值保持。

## 本步发现并关闭的缺陷

| 缺陷 | 实际修复与证据 |
|---|---|
| 新反馈复数字段被旧短ID投影当单数读取，恢复请求构造KeyError | 显式v2投影并检查unit/fragment/prior-claim/实际证据目录，HTTP-only完整消费者链覆盖 |
| 新无进展原因被泛化为`evaluation_contract_invalid` | 仅v2保留精确`inventory_recovery_no_progress`，不改变旧v1终态 |
| 预算拒绝诊断重新构造请求时遗漏host别名，掩盖资源原因 | 复用同一次已经构造的messages/schema；明确完整请求超限才记`inventory_recovery_request_capacity_exceeded`，身份、unknown及规划违约不降级资源 |
| 新global提示使合法176条请求118379超过117964 | 保留完整数据和下游规则，仅v2合并重复的宿主前言；实际完整请求降至117243，输出预留16384保持 |

全局前言的每项职责分别由短base和保留的HOST COMPLETENESS、BODY ANALYSIS、ownership、anchored及coverage规则承担，未删科学要求或任何原文。旧035完整global system前缀14415字符及旧GLOBAL_PROMPT1619字符逐字相同。曾建议共享的expected目录实际早已被旧producer消除，实测节省0，未把这一建议算成修复收益。

## 最终离线结果

- 最终默认`.venv/bin/python -m pytest -v`实际退出0：**5817通过、25默认外部排除、8既有warnings**，pytest 277.97秒，控制器282.637秒。
- 最终552份源码/配置/测试/资产与完整最大证明逐值同源，digest为`96c3eecfb23791d4ecd66a03414d42d4f8af44bf4146cddcb30b965b385b1044`。
- 本步10个生产文件、4个新测试模块及1个公开ViT回放fixture；14个Python文件Black/isort和`git diff --check`实际退出0。
- 12份旧计划逐值保持，旧fingerprint、默认False省略、普通API不接收新flag、新旧checkpoint隔离通过。
- HTTP-only Mock的真实链涵盖Synthesis→摘要→Reviewer→首次Evaluation→同SHA混合恢复→全部批次/global→剩余正文修复及摘要更新→完整复评→自动判定。科学反例、错误论文支持、缺单元、分类逃避、无进展、取消和未知响应仍不能交付。

第一次默认5816通过、退出0属于中间版本；同源最大请求随后失败，因此没有据此宣布离线退出。原失败、原封存和后续修复分别保留，最终使用第二次默认验收及重新构造的最大证明。

## 资源证明的准确范围

90个显式v2基线向量和6条最大合法账本路径完成，11个独立验证进程实际退出0。54个Evaluation矩阵全部通过，最大全局请求117243≤117964；完整正文、全部引用、原文及输出预留保持。账本路径中两次共享修复后仍有最终核验额度，但这是离线资源模拟，不是模型生成或科学通过。

其中**63个修复聚合子例是正确的发送前容量拒绝**，不能写成90项都可发。相较旧版本48项，15项因新回复提醒增加481保守tokens而跨过完整请求限额；这项代价明确保留，不提高上限或删诊断。给每条字段合法的独特长反馈，并不保证合计都能装入模型上下文。

真实035材料的v2反馈经实际producer、完整request builder及真实别名目录构造：20项问题、21条关联断言、6个原始global绑定，完整请求87882≤117964，包含16384输出预留。报告SHA和原归档未变；本次是捕获并计量，未占用修复机会、未预留或发送模型请求。

ASCII 32×2000说明的真实Gateway容量反例，以及独立CJK/转义补验均保留完整原始问题，在恢复attempt、预算占用和HTTP之前明确资源拒绝。CJK完整恢复预留123885、转义177452，均超过117964；原始global先通过接收，未知占用为0。它们是合成传输边界测试，不声称模型实际生成过这些最大响应或模拟usage是真实费用。

三份新预算重新准入：few-shot444万、ViT447万、NeRF447万，合1338万；各41计划请求、两次修复，仍低于450万/64及1350万/192硬上限。这些是计划额度，不是实际消费；旧035的443/447/447万记录不改。

## 历史、界面与剩余工作

10281份历史/035保护文件逐项未变，035及其父费用、旧未知占用、科学评分和批准不改变。前端及报告渲染/导出35个相关文件哈希保持；本步没有新报告，不冒称新v2的桌面、移动或PDF/Word真实报告验收。上一部035的同源界面与实际草稿导出证明仍可查。

当前明确未完成的是新版本真实科学质量验收。旧035的110支持/33不支持/16未核验仍是无效清单下的局部结果；本次没有把它们变成成功。后续验证仍需固定源码、材料、模型及有界新计划，不能恢复035、隐式追加到通过或拿离线数量代替正文目标。本步不推广工作台默认，不修改Plan.md，未commit或push。

## 证据入口

- `artifacts/acceptance/mixed-inventory-recovery-20261031/default-run-002/pytest-proof.json`：最终实际退出及552同源。
- 同目录`budget-review/maximum-release-001/summary-proof.json`、`boundary-transition-proof.json`：最大向量、资源边界、12旧计划、三份准入及真实035构造。
- 同目录`reply-agent/global-base-recheck-001/semantic-old-v1-proof.json`：规则职责与旧消息逐字对照。
- 同目录`review-agent/unicode-capacity/proof.json`：CJK/转义完整消费者容量拒绝。
- 同目录`final-source-and-preservation-proof.json`、`format-final-proof.json`、`completion-proof.json`：最终封存与完成范围。
