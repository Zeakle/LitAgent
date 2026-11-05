# Phase 14.5 / 294：自动交付接收完整扩容清单

本步只做离线工程修复，无新增真实模型调用。原045报告、评分、SHA、失败与费用保持；真实完整Evaluation及严格交付仍0/3。046已完成文件准备，但在付费前发现本问题，因此保持未启动，后续不在修改源码后恢复它。

## 为什么预检停止

293真实prepare/validate actual26175退出0，独立contract validate27083为0；三份预算444/448/448万、45父费用及材料正确。最终消费者仍有遗漏：runner._automatic_review先调用freeze_automatic_inputs，再判断Judge开关；该freeze固定拒绝超过128断言，所以Judge/Jev关也会阻止完整扩容报告。

actual27332用真实消费者证明：128完整清单可以冻结，129/144/176即使strict_acceptance通过且9/9/11批完整仍automatic_inventory_invalid。该证据是合成工程回归，不代表三份真实报告已科学通过。046没有state、execution.lock、launch-intent或收费请求。

## 最小修复与边界

仅修改automatic.py、runner.py，并新增直接tests。freeze新增仅宿主local的budget参数，实际runner传自己的CostBudget；前端与LLM不能传该对象。旧调用及<=128分支保持原语义。

超过128时必须由真实PlannedBudget v7授权：用现纯数据accounting reader校完整plan、revision和slot数学；所有实际请求已结算、没有未知usage、停止或在途占用。报告SHA、完整数量、实际批数和capacity描述必须绑定同一已认证assess round，每个assessment及global授权slot均实际sent/settled且wireSHA和预留一致。失败候选未提交时，可以选原稿先前已完成的同SHA round，不能只看最新allocation。

冻结产物只增加只读capacity_authorization摘要，绑定plan/allocation/snapshot SHA及数量，不复制整个预算、不授予发送或恢复权限、不释放额度。全部原strict_acceptance、fact-card来源、source quote、位置及实际引用守卫原样保留。容量授权不代表科学判断正确。

实际report未普遍保存metadata.capacity_policy/budget_policy，不新增不存在字段作为前提。真实batcher会按材料重量平衡，跨批顺序不必等原清单；复用完整唯一ID集合覆盖，逐原文位置不合并。真正分批数可能大于ceil(count/16)，最终以已授权实际批次为准。JSON round按规范digest比，计数exact int，不能让False等0或144.0等144。

独立Judge仍沿用自己的上下文、总预算与请求上限，本步不新增Judge128常量。旧RunOptions/v5/v6预算及历史文件不改；已有frozen Judge host fingerprint因宿主验证代码变化而使新核验身份变化，不覆盖旧记录。

## 真实回归与已完成证据

最终源码613文件，digest `9cfddacbb31a64d1a4760dab1ff59bb4479643f1e830dd39ea6ea6a6ecc3b200`。对612/f6的差分严格只有两个生产文件及新test；固定三题材料、模型、生成/核验请求构造器均未改。

- 新直接test42项actual28431退出0：129/144/176、真实生产分批器/请求预检/checked wire/结算、budget代理与真实runner、旧round保原稿及snapshot不变；科学失败、缺批/global、未知/在途/stop、坏链、refine/非v7、typed错误及同quote坏位置均拒。
- 最终相关8文件actual28580/wrapper28577退出0：204 passed，613源码前后保持。
- Black/isort/diff actual28810/28817/28818全部0，只检查本次三个Python。
- 新鲜构造9份计划actual28650退出0：旧v1无指导wire、旧v1+指导wire、新v2+指导wire各三题，逐值等原认证计划；新计划合计1340万、baseline41请求/题、两共享修复。当前消费者不增加模型请求或payload。
- 原84 Evaluation最大请求、72修复几何、摘要/prior/字节边界、有限完整预算路径通过逐文件/函数AST差分和完整原proof SHA绑定继承；actual28912再核原关闭10份绑定及6组资源进程。未声称重新执行旧最大owner，也不冒充任意规模全通过。
- 独立review28439与7探针28418实际退出0：两生产SHA稳定、原科学/来源/引用循环及前置门禁AST保持，剩余scope无blocker。其余原函数AST保持，runner仅增加local budget参数。

默认离线pytest actual28745/wrapper28743均实际退出0：7585 passed、25 deselected、8既有warnings；pytest442.58秒，wrapper447.73秒，613源码前后逐SHA保持。root保护 actual29944退出0：37559历史文件、Plan与046未启动状态保持；独立最终source/resource审 actual30251退出0。最终关闭凭证 `artifacts/acceptance/automatic-capacity-acceptance-20261102/final-closure-001.json` 绑定全部实际退出、日志及五plan副本。Phase14.5保持in_progress。

## 保留的失败与范围

首24均匀fixture未覆盖生产平衡顺序，actual28228真实producer红保留；首balanced28187则是测试误对旧EvalConfig填超128/8配置而失败，不能算产品缺陷证据。预算独立28074发现的类型别名是报告描述契约弱化，不夸大成已证明科学越权。首独立预算26844在validate成功后因漏verification.version观察器断言整体exit1，保持红，纯数据补证仅声明已完成范围。

本步没有新正文或导出，不把292已验23页复用成本步新报告视觉，也没有重启8000/8020或修改网页。完整离线退出后，以新源码、新套件和原三份材料安排固定三域各一次，保持450万/64每题、1350万/192累计、Pro/Lite disabled、0retry、两共享修复及Judge/Jev关。科学质量只能由后续真实结果证明。
