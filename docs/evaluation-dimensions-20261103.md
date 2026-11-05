# Phase 14.5：报告忠实性与引用支持分离（2026-11-03）

步骤324目前仅做离线实现和验收。Phase14.5仍为`in_progress`，最近054的有效整份Evaluation与严格交付保持0/3。本轮没有新付费调用、旧运行恢复、服务重启、默认策略推广或提交推送。

## 本步变更

新增服务端冻结实验选择`evaluation_dimensions_contract=separated-evaluation-dimensions-v1`及独立identity。它不由论文内容、普通API或writer source view自行选择，可以与仅原文写作视图独立或共同使用。

同一次assessment响应的每条decision分为两个对象：

- `report_check`保留原`correct_kind`、`atomic`、`source_faithful`和`explanation`，只比较实际报告单元、宿主片段及其限定条件。
- `evidence_check`保留原`status`、`reason_code`、`support_basis`和`aspect_checks`，核验该断言实际引用的完整原文。科学支持理由继续保存在原aspect explanation，不新增第二份长解释。

两组字段确定性还原为原canonical决策，再经过原宿主门禁。原始nested响应另存，任何false、unsupported、缺项、重复键、额外字段、错类型或错来源均不被修补为通过。报告忠实而证据不足仍失败；论文支持不能替代报告忠实性。结构隔离不是模型已正确理解两种判断的证明。

最终canonical和alias两个native候选在原来源/方面权限绑定后分别转换，然后使用实际system、messages及native schema计算预留并选择。仅对逐值相同的完整schema对象共享本地`$ref`；不裁剪来源、正文、断言或诊断。

## 机械关系边界

旧参数规则按`parameters`或“参数”子串触发，导致“被参数化为连续5D场”也进入参数数量检查。新显式策略只针对明确数量、规模和模型大小表达执行负向检查，区分参数化、相机/材质参数、参数数量与规模。8M/8B、不同数量对象、错误数值和未明确解析的单位不能凭裸数字相同通过。

`parameter_scope_issue`只返回问题或None；None不等于科学支持。旧edge/mobile、distillation主体、比较方向及后续实体、数值、条件和因果检查继续执行。没有实现通用数字拼写换算、相似词/断字补全或任意别名。

旧None路径仍调用原关系函数；旧support-v1/v2和response-v1/v2常量、被inspect的原函数不改。新策略identity绑定新投影、解码、关系规则和真实dispatcher；原API缺省语义保持。修改scoped源码会按原有规则改变基于整个源码的Judge指纹，本步不把旧Judge结果重绑定为新结果。

## 完整消费者与恢复

冻结plan、envelope、真实预算owner、private CLI、task、writer journal、每次call和纯历史读取均绑定同一pair。None不添加空字段；伪造、缺失或错配拒绝。

正常Evaluation从锁定owner选择。部分结果验证host inputs/progress与内部owner pair一致，仍声明`overall_valid=false`。settled raw的修复资格重放使用同一decoder，并从已认证owner恢复新关系策略；外部私有context不能开启或关闭它。最终Evaluation details和错误details均保存实际pair。

## 验收进度

最终source002有665文件，摘要`89065c90646ee6507a84e6d1266375e7238df7ea0e51c7a54c3ed8e98eb10a19`；本步14个Python文件及一个自包含054片段fixture发生变化。Plan SHA保持`b85f12cec18ad0d16b0a5cd370b748afdf5e3cc15f70570df018b1fd7c93e6cb`。

- source002最终210专项实际41898退出0，309.23秒，前后源码匹配最终封源。先前source001的202专项实际39945退出0，195.81秒，原件保持，不能替代最终源码验收。外部传输受控替换，真实Synthesis、Reviewer、唯一报告出口、Evaluation、两共享修复及完整复评执行。
- 14个Python的Black实际41912、isort41910及diff41911均退出0，前后源码相同。
- 最终默认离线pytest实际41643正在运行，尚不能计为通过。source001默认40649在1904通过后因已证实的旧异常兼容缺陷被核PID中断，实际exit2；原件和源前后相同记录保留。
- source002的完整三域计划已通过，旧12计划/144阶段wire逐值匹配before657；新维度独立444/448/448万、与source-only共存439/442/442万，各62请求/两修复。最大current/prior修复及完整两native分区资源验收仍待终态，尚不能计全通过。

证据目录为`artifacts/acceptance/evaluation-dimensions-20261103/`。原657源码、五plan和前步993证据在改动前复制及核对。054三例的自包含测试fixture保留原raw、原位置、完整自身来源及SHA，全部标记`gold=false`；受控正反决策不冒人工金标准或旧报告重评。

开发首红均保留：新fixture漏依赖导入；参数解析边界；distillation合成词边界；新test误解None的既有mapping默认；最终Evaluation产物最初遗漏pair；首gate观察器误读相对路径；旧非法输入异常兼容；权限观察器读取了尚未派生的batch字段。最终源码需独立完成所有门禁，不能从开发数量或协议通过推导科学质量。

## 仍未完成

原子性恢复尚未增加逐concern的显式处置。原恢复实际已完整复评、原hash仅用于候选差异/循环检查，不能把大小写改变、数量变化或新atomic=true解释为实际拆分或旧误报已被证实撤回。未来需要明确区分结构变化、原concern独立复核撤回和仍未解决，同时保留合法单一比较及联合条件，不强迫增加断言条数。

本步没有新的UI或导出实现和视觉验收，不补记旧390px、归档定位或历史Word限制为已验。真实模型是否减少维度混用、真实正文是否严格通过仍未验证；预算工程验收也不能替代科学质量验收。
