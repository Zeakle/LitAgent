# 请求超时归因与失败观测（2026-10-31）

## 结论与边界

步骤255完成离线工程验收，Phase14.5仍为`in_progress`。032首个正文请求的300秒超时、1请求62170未知占用和0/3严格交付原样保留。本步不恢复032，不新增付费模型请求，不把离线夹具当真实Survey或科学通过。

## 改动

1. `LLMRequestTimeout`只表示宿主拥有的请求期限确实到期。`chat_with_timeout`和实际SDK请求共享最早的期限；外部取消、SDK传输超时、内部普通TimeoutError保留不同归因。现有请求/Worker/调度时限、模型、输出、预算和重试数未改变。
2. 请求失败保留API、阶段、允许秒数、时限作用域、输出上限及稳定原因。脱敏白名单、LangFuse失败generation和RunRecorder本地失败节点共同消费这些字段。科学消息、原始响应和自由错误正文没有因此进入metadata-only trace。
3. Scheduler区分自己的Worker时限和typed请求超时。Writer journal保留`llm_request_timeout`及结构化诊断；已发送且usage未知仍为`interrupted_no_replay`，不结算为零、不释放占用、恢复不重发。
4. 独立Python栈定位了API导入时加载118份约3.35GB本机历史的问题。模块级索引初始化改为空字典，历史仍由既有lifespan加载一次；未删除历史、未修改启动配置或报告接口。

## 验收证据

- 默认`.venv/bin/python -m pytest -v`：**5559通过、25排除、8既有warnings，243.85秒，实际退出0**。受测540份源码哈希保持。
- 14项新回归使用真实SDK和消费者，仅替换HTTP传输；覆盖Chat/Responses在途超时、外部取消、SDK超时、嵌套期限、Scheduler归因、Writer账本/恢复不重发、旧事件兼容，以及API导入/正常lifespan。发送的HTTP JSON不增加观测字段，SDK read timeout保持原值。
- 本步7个Python文件Black/isort、`git diff --check`均退出0。前期夹具和导入停顿记录单独保留，不算最终验收。
- 6056个历史保护文件无变化；032冻结539份源码副本不变。v5/v6六份旧预算逐值一致；v7原三题计划仍443/447/447万tokens、合1337万，完整字典一致。
- 真实本机LangFuse写入/读回一条明确标记`OFFLINE CONTRACT FIXTURE`的合成超时记录：`responses`、`synthesis`、1秒、`request`、`llm_request_timeout`完整；科学IO空，usage不可用。模型HTTP由MockTransport替代，付费调用0，夹具占用不进入真实费用账本。原032 trace未修改。

[本机离线trace夹具](http://127.0.0.1:3000/project/litagent-local-project/traces/3d2d8ca5d286451eff8efd321bd01cc9)。完整本机证据在`artifacts/acceptance/request-timeout-diagnostics-20261031/`，默认进程退出、源码、格式、兼容及历史保护分别保存proof。

## 剩余问题

本步两次无密钥HEAD连接检查没有获得方舟HTTP响应，最后一次为ConnectTimeout，约8139ms；未发送授权头或科学材料，模型调用0。这只说明检查时连接不可用，不能据此断定032当时是网络、服务排队或推理延迟。

完整真实正文、摘要、Reviewer、Evaluation及必要修复仍未跑通。摘要约束工程修复的离线证据有效，但真实摘要成功、三域科学支持率1.0和严格交付目标尚无新证明。未产生新完整报告，不新增PDF/Word导出或UI科学通过声明，不推广工作台默认策略。

下一步先核对这台机器的方舟连接和服务状态，再为后续有界验证确定独立清单；旧032不续跑，未知usage不自动重发。本步未commit/push，未修改架构`Plan.md`。
