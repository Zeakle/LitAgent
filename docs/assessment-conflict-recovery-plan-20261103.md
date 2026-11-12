# Phase 14.5 / 下一步草案：评估语义冲突保守归为未核验

状态：**待确认，只读设计，不实施、不启动付费请求**。依据052原始响应与消费者重放；原052、费用、终态及SHA不变。该草案调整既定技术失败分类，需按AGENTS.md先确认架构数据流条款。

## 1. 已定位原因与不采用的处理

052正文、摘要、Reviewer及157条清单完成，预算扩批成立；第4支持批claim48的`unsupported`与全部必需aspect为true冲突。当前producer已明确要求一致、完整predicate含否定边界，未发现要求相反的工程约束。原JSON/schema合法不能证明科学结论正确。claim42/43额外输出causal_relation也违反已有all/only规则，现消费者已将它们保守记为未核验。

六个reason_code可唯一派生status，但删除status并不能消除claim48的负reason与全正aspect冲突。反过来仅从全正aspect推supported，会丢掉当前明确的负判信号。因此本草案**保留全部独立原判**，不靠重复提示或删字段宣称解决。

## 2. 唯一行为变更

仅新、完整冻结的flat/V2评估路径可显式选择服务端策略`semantic_conflict_policy=host-unverified-continue-v1`；普通API缺省、当前未选路径、旧身份及历史行为不变。

在以下前置条件全部通过后，对两种特定语义冲突进行逐断言保守降级：

- 完整原JSON可解析、schema/类型/必需字段/全部ID/原始来源身份合法；完整批次覆盖无遗漏、重复或超量；
- 在捕获冲突前先核原文位置、实际引用、claim-owned basis与每个support ID的存在/唯一性/归属/边界；aspect集合必须恰好等于required且无重复，所用support IDs只能来自合法basis。不能因为旧positive分支先发现布尔冲突就跳过这些验证；
- usage已完整结算，无在途或未知响应；
- 冲突精确为`positive_status_with_negative_required_aspect`或`negative_status_with_complete_positive_diagnostics`。不得以捕获所有异常方式扩大范围。

模型raw与`batch.decisions`的canonical原判逐字/逐值保存，不修改status、reason、aspect、解释或证据。宿主另保存经过独立认证的effect：冲突代码、原判、报告SHA、断言ID、原文范围和实际引用；host assertion沿用内部`status="unverifiable"`，页面计数称未核验（unverified），不混淆两种词汇。该条在科学分母中只计一次，不能贡献支持数量。global、analysis、limitations和coverage使用的supported IDs必须排除effect断言；其他合法条目继续受相同完整科学检查。

完成余批和全局检查，只表示诊断收集完成；只要存在冲突未核验项，整篇不能自动通过，也不能显示有效支持率达标。缺少任意批次/全局结果时仍不完整，不从局部结果推算整篇分数。

非法JSON、截断、schema/身份错误、遗漏/重复ID、状态与reason非法配对、认证失败、未知usage、预算规划违约及系统故障继续技术停止。引用/来源错误沿用各自原有严格拒绝或本地未核验路径，不能借两码变更错误类别或放过。atomic、fidelity、classification的既有否决也不能被冲突effect遮蔽或因此新增修复资格；不增加传输重试或评估重发。

## 3. 复用既有有界修复

仅当所有支持批与global已经完成、清单唯一且完整合法、当前run/report/checkpoint/ledger认证全部通过后，才可根据新typed effect进入既有正文修复；partial回调、atomic/fidelity/分类失败本身不授机会。新effect通过单独的当前身份/定位/原raw链认证，不伪造support_validation来源问题或扩写既有11项白名单以绕过资格检查。

冲突只构成“该原文需独立复查”的结构化目标，不能把自由解释升格成已确认科学错误。既有host-structured-repair将精确范围、完整原文和引用提供给Pro，与Reviewer/容量/科学修复共享原两次HTTP。模型可保留、改写或decline；不能额外调用Lite修字段，不能只修改评估结果以通过。完整指导或实际请求放不下时按原资源终态拒绝，不截断问题或提高字节上限。

原稿不提交失败候选。任何候选变化后更新摘要、结论、引用绑定，并对整篇重新抽取、全部支持批次及global完整复评；科学支持率1.0、核心覆盖、具体局限、比较条件和回归检查全部通过后才原子提交。未修复、重复、无进展、覆盖回退、两次耗尽或资源不足均形成不可交付终态。

新策略保存在启动时不可变的locked envelope，不来自可变context、普通公开请求或运行中开关；与实际handler规则、诊断契约一起进入完整预算计划、请求/checkpoint、候选和自动判定身份。None/未选/旧V2仍按原raise，不用新规则恢复052或将旧raw追认为新策略成功。新增策略不减少输出预留、两次修复或完整材料；最高450万/64不变。因合法诊断更完整而增大的修复请求，必须重新经过真实构造器、最大typed压力和完整两修复资源验收。

## 4. 实施顺序与离线退出

1. 保存630/ec6e及052原件和费用；获确认后同步Plan已确认条款、中文GUIDE单步、路线、状态、SUGGESTION。
2. 在现有一致性检查拆出可定位的纯分类结果；旧入口保持原raise，新显式选择仅对上述两码产生宿主effect。其余错误不捕获/不重分类，不新增Agent。
3. 打通批次partial、完整汇总、global、修复资格、结构化指导、候选复评及自动判定。科学原判与宿主effect分栏保存，不能静默写回模型decision。
4. 用实际052、041/043等原raw正反例重放，Mock仅替外部传输，执行Synthesis→Reviewer→唯一报告出口→Evaluation→共享修复→完整复评→自动判定。
5. 验证实际最大请求、全部来源、诊断字节上界、两次共享修复及最后复评预算。默认离线pytest实际exit0、改动Python Black/isort和git diff --check通过；界面和四格式对齐原判/未核验/完整性。

## 5. 必须保留的负例

- claim48原旧路径仍拒绝；新路径原unsupported和全正aspect同时保留，宿主effect为未核验，禁止自动变supported。
- 反向supported+负aspect也只能未核验，不能以多数票或更乐观字段通过。
- 在两种冲突中分别植入池外basis、重复/额外/遗漏aspect、跨断言support、坏来源SHA、partial或未结算usage，均不能通过新handler进入继续/修复路径。
- EViT/Evo-ViT、DeiT/ConViT、否定边界、平均/峰值、百分比/百分点、FLOPs/速度、条件及论文归属错误继续阻止交付。
- 多余/缺少aspect、缺原文、错引用、atomic/fidelity/classification不成立不伪装已支持；已有更保守语义保持，非法结构仍停止。
- 冲突不能通过删句、改组织分类、删核心问题或只改评估JSON消失；最终所有保留科学内容完整核验。
- 任一批缺失/取消/unknown usage或global失败，不能产生可交付分数；进程重启不自动重发，旧结果不跨身份复用。
- 两次修复HTTP跨入口共享、失败不提交；费用含失败请求，预留/实际/未知分别展示。

## 6. 授权边界

本草案请求确认的是**把两种合法结构下的评估语义矛盾从立即停止改为本地未核验并继续诊断**；这是实际失败分类和后续请求顺序的变更，不能表述成原技术停止语义不变。未获确认前不修改Plan、源码或原运行。

下一步先只做离线修复与验收，不恢复052，不增加付费协议试跑、不自动新建15题套件、不推广工作台默认。未来真实验收仍须先冻结同源源码、完整资源和固定输入，并遵守既有有界授权，不追加到通过。
