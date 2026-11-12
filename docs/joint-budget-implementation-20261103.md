# 逐轮联合预留：实现与离线验收

## 改动及保证范围

步骤320新增内部可选 `joint-round-escrow-v1`，由原可信v7计划推导每轮最多请求数N、单请求上限C和整轮总预留E。assessment总预留不再等于N×C；首次完整核验与两次修复后的完整复评分别锁定。已结算后的当前轮可安全扩配，但不能使用未来修复的保护额度。

保证只覆盖冻结运营资源域内完整请求的资金和次数；不保证任意模型输出能放进单请求上下文，不代表供应商一定返回或科学内容一定通过。域外保留完整清单，重新检查剩余全路径后扩配，或进入原两次共享精简。精简可保留断言数、批数而降低请求总量；目标本身不能授权发送或代表可行。科学支持率1.0、完整原文、核心覆盖、64请求和450万硬上限保持。

None原路径不调用包装，旧envelope、模型payload和native schema保持。新选择只存在可信冻结manifest、完整expected plan及宿主task绑定中；普通API不新增任意预算、endpoint或本地路径参数。新live owner完整重建原计划及域，纯历史reader只核对冻结数据、整数合同与原父身份，不执行当前renderer。

## 固定材料的只读结果

原053材料未重新搜索或提取；本轮未发送模型请求、未恢复原任务。

|题目|原锁定额度|每轮最多核验请求|单请求上限|每轮完整核验预留|计划最多请求|
|---|---:|---:|---:|---:|---:|
|few-shot|4,440,000|15|100,000|800,894|62|
|ViT|4,480,000|15|100,000|802,594|62|
|NeRF|4,480,000|15|100,000|802,380|62|

这些是保守预留，不是预计费用。三题None完整计划及旧response/prompt选择计划与before648逐值一致，五条已知初始请求的messages/native前后逐值一致；每题来源SHA及bundle完整。新计划经live空owner和纯reader认证，没有reserve、send、record或restore。核验记录：`artifacts/acceptance/joint-budget-implementation-20261103/final-plans-002/proof.json`及`terminal-receipt.json`，两child实际退出0。首次观察器因读取seal不存在字段而失败的记录保留；002退出时profile探针的ignored warning保留，不称为产品失败。

## 回归与展示

- 实际owner按全部stage最大reservation结算，三轮assessment均真实构造最多N条、总和恰好E的请求，两次修复后仍能完成最后global；未用低usage假造额度。
- 自包含67条清单、9批完整实际构造进入新域；每批只使用其实际引用的完整原文，整体来源覆盖及逐claim权限保持。单请求100001、完整路径超金额或超64次均在发送前拒绝。
- 冻结入口→Synthesis→真实SDK（仅外部HTTP替换）→唯一报告出口→Evaluation→共享两次修复与完整复评已通过专项；新资源选择不改变模型材料或schema。
- 伪descriptor、自重hash、父身份、selector剥离、未知usage、崩溃、恢复及同轮重发受保护；历史None与pure读取保留。
- UI及JSON/CSV/Markdown/HTML区分最多次数、单请求cap与总预留，禁止虚假乘法；导出未运行案例仍为not_run，不显示科学通过。
- 浏览器8000已核当前320资源、历史038质量blocked及未核验、中英文、键盘展开、390px中英文无横向溢出；新域展示由真实Node renderer及四格式消费者验证，未假装历史报告采用新合同。默认viewport及中文恢复。

封源650文件，digest `ec1cded273ed68becc60ef411c70bc768d8e94c34c2e080e34f50e28eaaec113`。9个改动Python Black/isort及git diff --check实际0，最终新专项68通过。默认离线 `pytest -v` 实际child7157退出0：8711通过、25排除、8warnings，测试耗时1155.60秒；完整封源前后相同。日志、JUnit与实际退出保存在default-final-001。

本轮新增付费0；旧报告、评分、SHA、失败和费用不变。真实严格结果仍0/3，Phase14.5保持in_progress。没有新生成正文，不能把本轮Benchmark四格式预算展示当作Markdown/PDF/Word报告的新增视觉验收。
