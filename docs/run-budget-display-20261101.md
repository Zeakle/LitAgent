# 运行预算展示接线验收（2026-11-01）

步骤275已完成。质量页现在分别展示已知用量、未知保守占用、已发请求和用量不完整状态；预算预留及容量分配时差额单列。Phase 14.5仍为`in_progress`，最近真实038仍严格0/3，本次没有新增付费调用。

## 修复内容

原`budget.updated`事件和归档已经保存六字段预算快照，但`demo`公共投影、interactive桥接和前端丢失了它，主摘要仅显示已知tokens。现新增只读浏览器投影`run-budget-state-v1`，经过两条真实视图及adapter到达质量页。

原client、recorder、账本和费用格式未改。快照只接受严格白名单及真实类型，拒绝错误run、计划身份冲突、布尔数字、非有限或不安全整数及任意错误文本。缺失显示未记录，非法显示不可用，不推算零消费。预算属于整个run，不声称是当前报告版本成本；未知占用不计入已知用量，不构成退款、重发或余额授权。

038 ViT实际显示已知218506 tokens、未知占用78205、14请求、用量不完整及`strict_usage_unknown`；主Tokens仍为218506。原容量区483523保持“分配时已结算预留差额”语义，不变成终态费用。few-shot显示已知502804、未知0、34请求，仅说明本次快照未记录未知占用，不承诺最终科学通过或费用最终完整。

## 离线证据

最终570源码digest为`91f52e8c682922c9b4feedbf2116e17be7f58ab8b3a7db3d9e1eb221deda3b93`。本步新增129项测试，覆盖纯投影、临时repository真实interactive桥接，以及实际planned SDK→事件→RunRecorder→归档→两adapter→DOM。仅外部HTTP传输被Mock，未知响应、403、取消及未来未发送global分配均通过真实消费者验证。

默认`pytest -v`实际6451 passed、25外部测试deselected、8既有warnings，实际退出0；本次6个Python文件Black/isort及`git diff --check`退出0。相关后端123、前端138、独立消费者256均实际退出0，分组有交集，不相加冒充测试总数。

`maximum-002`的12个真实verifier、汇总及独立核对均实际退出0：90向量、54预检、6完整资源路径保持；69个聚合修复越界反例仍拒绝。旧18份计划、038三计划、039三计划逐值保持。`maximum-001`因私有入口漏复制基线JSON而11项通过、1项FileNotFound，原失败保留；补齐原fixture后重新完整运行002，没有修改产品或减少资源反例。

## 实际界面与只读预览

受控刷新8000前核验了本任务原PID33556、命令、cwd及单一端口owner；新PID82754仅监听127.0.0.1:8000，presentation模式、lifespan关闭、只加载两份原038归档。8020、Docker、029原脚本及cache未改，不初始化live、独立核验或历史恢复。独立ASGI16接口含8项变更拒绝通过；本地CSL格式化与付费服务分开审查，导入库的一次IPv6回环能力探测被私有护栏拒绝并单列，不算外部服务或模型请求。

实际完成桌面1280×720、移动390×844、中英预算区、稳定APA切换后的证据定位、Esc恢复引用焦点、展开附录及归档run切换。10张JPEG与DOM逐文件绑定，真实图像尺寸和CSS尺寸全部一致。最终停留在中文、IEEE、默认桌面视口的ViT质量页。

切run首帧确实清除了旧78205预算；完成后few-shot的502804/0/34与1/2清单恢复标签正确。加载期间过早点质量会被默认报告页覆盖，稳定加载后选择质量通过；该边界仍记录，不声称所有导航时序已验。列表加载后按钮名称增加日期、样式切换首帧仍旧值，均按新DOM和稳定状态重新确认。现有材料仅r1，没有实测同run多revision切换；该竞态依赖离线消费者测试。本次没有新报告或新PDF/Word导出验收，旧导出统计和原字节保持。

## 保留与后续

14782份历史保护文件SHA全部不变，包括原报告、评分、批准、失败、未知占用和039冻结副本。039仍已准备未运行，其旧全源码绑定已经过期，未修改manifest追认；没有准备040、恢复038或重发未知请求。

预算展示完成不等于正文质量通过。账户恢复状态仍待明确；已有持续有界付费授权保持。下一轮真实三域验证尚未启动，须以最终源码和必要启动证据独立冻结，不能复用已过期039，也不能隐藏旧失败或追加到通过。

证据根：`artifacts/acceptance/run-budget-display-20261101`。主要文件：`completion-proof.json`、`default-001-proof.json`、`maximum-002/independent-final-check.json`、`maximum-002/independent-source-history-check.json`、`pipeline/proof-001.json`、`preview-review/review.json`、`preview-refresh-proof.json`、`browser-observations.json`及`ui-files-proof.json`。
