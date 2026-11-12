# 首次写作 EOF 容器政策实现审查

2026-10-26，依据用户明确批准的[设计](writer-syntax-boundary-proposal-20261026.md)，在既有 Writer 编译边界实现 `grounded-writer-eof-containers-v1`。这是一项有意的接受规则变更，不能称作旧行为完全不变。

## 实际行为

- 仅固定服务端 `writer_low_reviewer_disabled_responses_eof_v1` 启用。Pro 写作 enabled/low，Lite Reviewer disabled、Evaluation 保持原配置；普通 Survey/API 默认、旧 profile 参数及旧身份算法保持。
- 仅首次 `body:<固定章节>` 和 `summary_initial` 可归一化。完整合法 JSON 仍走原解析器；Reviewer、Evaluation、语义修订及修订后的摘要均不适用。
- 唯一编辑为按解析栈在 EOF 追加 `]`/`}`。候选须通过完整宿主 schema，原前缀和候选须具有相同完整单元、原始位置、段落/章节归属和语义指纹。字符串未闭合、中部损坏、重复键、非法字段或引用、缺单元和错误容器继续拒绝。
- 原 raw、原 JSON 诊断、HTTP receipt 和请求身份保留；派生候选另存 raw/候选/schema/供应商 schema SHA、追加符及单元位置。一次响应不会伪造成两次模型调用。新政策进入会话及冻结清单身份，重启仍不重发。
- 本地初始编译不消耗两次语义修订；两次拒绝响应仍计次数，第三次请求被拒绝。原科学支持率1.0、核心覆盖、方法披露、全局检查、预算与零重试保持。
- trace 只发送政策、阶段、身份/hash和数量；Benchmark view-v11 的四种格式共用脱敏投影，明确“语法处理不等于科学核验”。旧记录缺值显示不可用。

## 实际失败响应重放

016的分类、方法和实验章节三份原始 HTTP 响应经过安装的 SDK→WriterSession→生产解析器重放。三次请求字节与原请求完全相同，只有 MockTransport 返回已有原包，没有外部模型调用。

|章节|原完整单元数|追加|与当时独立模型重发逐字相同|
|---|---:|---|---|
|taxonomy|22|` ]} `（无空格）|是|
|methods|42|` ]} `（无空格）|是|
|experiments|18|` ]} `（无空格）|当时未重发，无比较结果|

这三个数是语法单元数，不是已核验的科学断言数。重放用量39,573 tokens仅用于核对旧记录，不计为新费用。重放不生成完整报告、不补写余下章节、不重新评分016。历史三域严格结果仍为0/3。

## 验收材料与边界

- `tests/test_writer_eof_policy.py`：闭合与拒绝反例、原字节/单元范围、usage/finish/取消、持久化与重启、两次语义上限。
- `tests/test_writer_eof_profile.py`：显式profile、旧参数、重算hash后的身份漂移、脱敏四格式和真实trace消费者。
- `tests/test_native_responses.py`：只替换HTTP，实际执行Synthesis、Reviewer、唯一报告出口、Evaluation v3和自动判定；合法例可交付，科学不支持和缺记录例仍拒绝。
- 私有证明位于 `artifacts/acceptance/writer-eof-policy-20261026/`：实际重放、默认回归、相关测试、历史保护、桌面/390px中英文与键盘验收。离线UI明确标记 `offline_receipt_replay`，不能当作新真实Benchmark。
- 授权前Plan字节有独立备份，其余历史运行原件及487份冻结源码副本核对；费用与未知占用保持。

本步只证明工程接受规则及回归有效。新的三域真实验证必须在源码冻结后各执行一次；完整Evaluation和严格交付未达到前，Phase14.5仍为in_progress，不宣称普遍可靠。

## 本步最终离线结果

默认 `pytest -v`：4702 passed、25 deselected、8既有warnings，293.26秒。相关首轮320通过、最终导出/政策84通过；9个改动Python文件Black/isort及git diff --check通过。桌面1280、390px、中英文、Enter/Space键盘、无横向溢出与console零错误/警告完成。首轮沙箱回环监听error日志保留，最终在允许127.0.0.1监听的环境验证原PDF安全反例通过。上述结果不构成新的真实科学验收。

## 一次有界真实验证017

源码冻结于`a91b098`，固定三域及18父receipt；原问题、完整来源、配置和预算与016一致。few-shot完成94个正文及8个摘要/结论单元，分类章节实际触发一次容器闭合（14单元），语义修订0次。Lite Reviewer完整响应并报告5项问题，不能把“响应完成”称为质量通过。

Evaluation独立抽取出147条科学断言及5条组织陈述。全部96原文单元、102模型片段有处理记录，但147超过固定128容量，支持批次未发送，整体未完整核验。按技术停止规则，ViT/NeRF未运行，严格交付仍0/3；没有提高上限、截断清单或追加尝试。

本次8请求／146710完整已知tokens／392.665秒／零重试，2215思考tokens包含于输出，无新增未知占用。LangFuse读回8个generation及同用量，科学输入输出为空，并确认新政策的身份、SHA及数量元数据。原报告SHA`4a4130223daaebae488cf430f66d07c7ee2c16d85a32a2aaa01f2683901ab379`、评分、失败和账本保持。

[真实结果及完整边界](../benchmarks/survey/results/frozen-writer-three-domain-20261026-017/README.md)。当前待解决的是写作单元数量与独立原子断言容量不等价；下一步需联合审计事实密度、核心覆盖、完整分批及修复预算，不降低科学门槛。本次证明语法政策及Reviewer角色配置已在真实链路生效，没有证明科学支持率或普遍可靠性。
