# 逐断言响应约束与全局完整性一致性验收

2026-11-03，Phase 14.5 / 步骤316。最终源码643文件，摘要 `c676518ddb860ef4f131f6ba8687265ee5121a714fe4f58d455b396843b1112e`。本轮新增真实模型调用0，原053三域严格交付仍 **0/3**；完成的是响应契约的离线修复，联合预算保障仍有待办。

## 已修复行为

原few-shot六项来源错误不是未知ID，而是评估器选择了同批其他断言的来源。输入的逐claim白名单和host拒绝正确，旧输出schema却只约束整个batch的来源集合。

新内部 `evaluation_response_contract=host-bound-evaluation-v1` 将claim ID与自身允许的来源绑定：相同权限集合可共享native schema分支，每个claim、位置、科学判断和统计分母仍独立。`support_basis`与`aspect_checks.support_ids`均受约束。空来源没有整批兜底，合法子集不等于科学支持；aspect仍必须获其实际选中basis支持，由原host继续核验。

保持原decisions数组、原始支持文本、断言抽取、科学门槛和两次共享修复。没有新增Agent、模型请求、格式重发或科学分类豁免。canonical与短alias两个候选均在全部描述/别名完成后绑定，再按最终messages及schema实际计量；未知alias、错误目录和缺失input marker不能使新契约静默退回旧schema。

NeRF首次global的三条missing又把同一单元标为complete=true。新契约仅检查这项确定性的结构冲突，保留原raw/锚点/判断并返回 `evaluation_global_completeness_conflict`，整体未核验，不自动删finding、改布尔或授权正文修复。non_atomic、报告重复和科学unsupported不因这一规则被当成遗漏。

新选择及其identity通过内部envelope、budget plan、冻结任务、Synthesis和脚本prepare-only参数锁定；公开SurveyRequest不增加参数。plan缺失、task伪造、manifest残缺及非v7计划均拒绝。纯费用reader保留历史identity作为钉住的数据，不用当前规则追认或重跑旧账本。

None和旧V2响应、envelope及计划identity按原算法保留。实际修改scoped.py仍会按原whole-file规则改变自动Judge指纹，不能声称所有源码相关指纹不变。原历史checkpoint不迁移。

## 实际执行证据

证据根目录：`artifacts/acceptance/claim-response-contract-20261103/`。

| 项目 | 实际结果 | 目录 |
|---|---|---|
| 默认离线回归 | pytest80960退出0，8539通过、25排除、8 warnings，970.98秒 | `default-001/` |
| 定向回归 | pytest79959退出0，260通过；含43项新约束和6项新完整消费者场景 | `focused-001/` |
| Python格式 | Black80654、isort80653均退出0，只检查本轮9文件 | `black-001/`、`isort-001/` |
| 冻结/任务/费用接线 | 实际81018退出0，4类边界及原六项native验证 | `wiring-consumer-001/` |
| 资源修正观察 | 实际81150退出0，真实producer完整分批；资源收束81941退出0 | `resource-review-001/` |

实际Synthesis→Reviewer→冻结计划→runner→Ragas/scoped→SDK消费者均执行，只有HTTP回复被Mock。正例可完成原科学门禁和自动判定；错误实验条件、跨claim来源及global冲突仍阻止交付。该模拟不构成人工科学金标准或真实通过率证明。

原六个完整raw通过各自旧native schema；相同原pN命名空间下，新schema逐项拒绝原越权选择，合法形状对照通过。没有借namespace不匹配制造反例，也没有改写原basis、状态、报告、响应和费用。原claim27的合法与非法混合选择一并保留，不能删错项后升级为支持。

真实无发送接线探针验证：run_frozen缺/错selector，Synthesis残缺或伪造pair，manifest strip/rehash/错误profile，以及v6套用新契约均拒绝。None配置返回原空选项；历史pinned identity可被纯费用reader认证，但当前live校验仍拒绝过期identity。

## 资源结论的适用范围

三个新初始计划实际返回准入：few-shot444万、ViT448万、NeRF448万，每题最高450万/64不变。它们由首观察器80685在后续夹具错误前真实生成；该观察器整体退出1的记录保持，不冒称整体资源全绿。

修正观察器81150使用实际 `compile_document → bound_fragments → _assessment_support_inputs → batch_payload → build_evaluation_request → plan_batches`，没有手造多余全文或裁剪原文。

| 合法有限向量 | 未拆完整预留 | 实际分批 | 每批完整预留 | 原文 |
|---|---:|---|---|---|
| 16种不同claim权限 | 148432 | 8＋8 | 96674 / 96775 | 每批12份完整来源逐字保留 |
| 全部claim引用全部来源的对照 | 113649 | 8＋8 | 80468 / 80565 | 每批12份完整来源逐字保留 |

两种向量均保留16条断言恰好一次，正文/claim字节在配置范围内，每批不超过100000锁定cap。来源完整与schema合法都不代表科学支持。

这证明了这两个有限向量的真实拆批，**没有证明未知未来清单、所有合法正文与两次完整复评的联合上界**。实际当轮exact allocation、未来100k escrow及初始8批是不同边界；不能以单次16条请求超限直接认定workflow违约，也不能以8＋8成功认定任意未来都可负担。

私有更紧凑schema原型及其失败尝试均保留，但未进入产品，也未作为本步通用资源准入依据。尚未完成供应商账户对新schema的真实强制行为验证。方舟[公开结构化输出说明](https://docs.volcengine.com/docs/ark/structured-output-beta?lang=zh)列出anyOf、enum及本地ref；该文档不替代host守卫或真实语义验收。

## 首错保留与历史保护

独立静态审发现新selected路径在缺input marker时可绕过two-candidate binder，已在发送前fail-closed并加入反例；旧路径不受影响。alias参数也必须是本批真实完整目录，不接受自造键或交换映射。

接线探针首80845因同一v6合成数据还携带另一独立semantic契约，先命中其v7-only guard而红。最终只在私有数据隔离目标，保持专属reason断言，未修改产品迎合测试。

资源首80685手工构造重复发送原文、scope_context不等价，后续对照又未同步unit引用汇总，整体退出1；194683不能作生产上界。修正为真实producer后保留完整来源并得到上述148432及安全拆分。所有首红与原型尝试保留，不混成科学失败或最终通过证据。

资源收束81941把当时仍在写的自身日志收录为空SHA；进程末尾输出后，仅这一条SHA失效，其余37条匹配。原proof和日志原样保留，独立标准库追验83249退出0，记录已结束日志的实际SHA并再次核对643源码及674份原053/材料文件。不能把原81941证据图描述为完全闭合；此追验只补充终态证据认证，不改变资源测量或科学结果。

54289份历史保护继续逐SHA核对；`Plan.md`仍为b710原件。累计1488请求、15417163已知tokens、1283047旧未知占用保持，新增付费0。未恢复053、重发未知请求、改服务、推广默认或追加正式15题。页面/导出没有新字段或新报告，本轮不冒称新的视觉与导出验收。

## 下一步

先完成联合预算边界：分别管理实际模型/API上下文、本地预留及repair stage资金，重建053 ViT完整current/prior请求，并覆盖新native权限组数、正文、来源、断言与诊断字节的联合合法极值及两次完整复评。保450万/64、完整来源、原输出预留与科学1.0，不能只提高EvalConfig或按历史123123设一个恰好通过的常量。

全部必要资源与离线门禁完成前不开始新真实套件。仍须取得固定三域的真实科学验收，才能评价正文通过率是否提高；本轮没有作此声明。Phase14.5继续 `in_progress`。
