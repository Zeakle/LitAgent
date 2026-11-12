# 044 固定三域真实验收：生成完成，精简映射协议停止

本轮已执行真实请求。API恢复可用，但验收目标未达：计划3题，运行1题，生成1题，Reviewer完成1题，完整Evaluation 0题，严格交付0/3，2题按技术失败停止规则未运行。Phase14.5保持`in_progress`。本轮不恢复043或044，不追加试跑、不自动运行15题。

## 实际输入、版本与边界

- 套件：`frozen-writer-three-domain-20261101-044`，manifest identity `63811c485ea0352952a40d40b300438f95efaed577f74ec9331bd82189c94540`。
- 最终冻结591文件，source fingerprint `704b22d3cf3aa54f2d38c3a7f609383fb8f4b9ec161d9477e8794eaf341bd3e9`。Git基准98abe219不能单独代表含既有未提交修改的本轮源码。
- few-shot→ViT→NeRF各一次；原公开快照20卡／9篇、20卡／12篇、18卡／11篇。不重新搜索、提取或计入历史提取费用。
- Pro `doubao-seed-2-1-pro-260915`写作／摘要／精简，Lite `doubao-seed-2-1-lite-260915` Reviewer／Evaluation；北京Ark Responses，thinking关闭，零重试。Judge／Jev关闭，无人工审批。
- 实际锁定444／448／448万tokens，共1340万；每题硬限450万／64请求，套件192请求，两次共享修复。科学支持1.0、完整覆盖与全局检查未调整。
- 启动前prepare64805、独立validate审查65515、纯数据费用65282、实际controller预检65637均退出0；6909默认、最大12＋4、50资源矩阵绑定同源。50中38实际／合法请求fit，12独立联合字段压力拒绝，不能写成50全部准入。

## 实际运行与停止原因

根唯一工具52116，wrapper65864／child65894，实际退出0，耗时480.979秒。进程退出0表示结束记录成功，科学结果未通过。

|题目|生成|Reviewer|完整Evaluation|交付|原因|
|---|---|---|---|---|---|
|few-shot|完成|完成|未完成|不可交付|writer_capacity_edit_position_invalid|
|ViT|未运行|未运行|未运行|不可交付|技术停止后未启动|
|NeRF|未运行|未运行|未运行|不可交付|技术停止后未启动|

few-shot共有45正文单元、4摘要／结论单元。初始完整清单201项，科学197项、其他4项，113/113原文单元均处理；核验支持批次尚未发送。全部科学项均未核验，0支持／0不支持不表示科学正确率为0。

容量规划从128自动扩到192，201项仍无法在现有额度内为全部后续步骤完整预留，进入计划内精简。第一次共享修复发出并结算，尚余一次不能覆盖技术失败继续调用；正文候选未提交，报告SHA仍为`525987d2e664b5643e00a54ecb843b7e1ebc68785a1c9ca70b2a131126189fbb`。

响应JSON和原生schema通过，但39个target各返回1个单元；45条旧单元映射中39条位置越界。首例r2只有本地[0]，响应却填写[1]。实际SDK system已有目标内索引说明和跨target各[0]示例，未发现producer／consumer约束矛盾。按global强行解释还有29条关联到其他target，不能安全自动改号。实际原raw→parse→guard独立重放仍精确拒绝，未改正文、引用或科学判定。

私有诊断001有旧模板写死47的元数据错误；实际脚本assert及新增纯数据clarification均为39越界记录／39越界整数。保留001原件，不把47视为另一统计分母。

## 用量与历史保护

新增9请求全部sent／settled：131587输入＋21267输出＝152854已知tokens，未知0。8个writer journal调用（含Reviewer和精简）＋1个Evaluation清单调用。不是实际货币账单；保守锁定额度不计为消费。

旧43费用父保持，新增后累计1120请求／9828197已知tokens／1283047历史未知占用，038的78205保留。没有恢复旧科学owner或重新发送未知请求。591当前／冻结源码和27738历史SHA保持，架构Plan8384未改。本轮终态后无新增模型请求。

## 展示及导出

8020静态验收页已实测1280×900、390×844中英文、document.lang、Space键展开／关闭诊断；最终重置1280×720、zh-CN。390px文档宽度390，控制台warn／error观察0。8000工作台返回新归档not found，因此该页原文／证据交互定位未完成，未重启服务或更改持久交付标记。

JSON／CSV／Markdown／HTML共同结果均保留三题分母、停止码、0/0/197及9／152854／0用量。Reviewer在JSON及详情存在，主矩阵仍缺独立列；部分诊断host_validated 0/0/0不能当作最终核验数量。

实际few-shot草稿已生成Markdown／PDF／Word和BibTeX／RIS／CSL JSON。元数据独立检查：119个显示编号／165个证据引用occurrences／9条书目，三种报告引用顺序和参考文献→附录顺序一致。54项缺失元数据警告保留，不补作者年份；六格式导出成功不代表科学通过。视觉验收单独以export044的最终QA凭证为准，不以文本解析代替页面检查。

## 下一步最小修复方向

在既有Worker和唯一报告出口内，采用版本化的selected精简关系契约：模型声明replacement对应的原source_unit_ids和显式删除理由，宿主依据实际target内顺序生成位置及旧→新映射。保留拆分／合并的语义责任、完整覆盖、跨target拒绝、删除保护、两共享、完整复评及失败不提交；旧v2／None身份保持，不追认044。

先用044真实响应建立离线反例并验证实际SDK消费者链、身份隔离、最大多对多关系及两次完整未来预留，再安排后续有界验收。不能只改提示、自动改号或增加预算后再次调用。

## 本地证据

- `artifacts/acceptance/evaluation-stability-live-20261101/paid-tool-exit-044.json`
- `terminal044-fee/proof.json`（同一验收目录下）
- `terminal044-diagnosis/position-review-001.json`及新增count clarification
- `ui044/proof.json`、`export044/`实际导出与QA
- `artifacts/acceptance/frozen-writer-three-domain-20261101-044/manifest.json`及`state.json`
- `benchmarks/survey/results/frozen-writer-three-domain-20261101-044/`四格式结果
