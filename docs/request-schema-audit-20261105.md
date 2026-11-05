# 模型请求契约审查（2026-11-05）

本次修复基于065真实失败，覆盖当前冻结写作主链及普通Survey/可选核验的实际发送入口。没有新增付费调用，不修改065或历史评分。原生schema约束、宿主跨字段检查和科学判断是不同层；不能把schema合法解释为科学正确。

## 发现与修复

065的逐条核验schema只枚举方面名称，宿主却要求每个必需方面恰好出现一次。重复`conditions`符合当时的原生schema，宿主正确拒绝。新内部传输`reason-only-decision-v3`改为固定必填方面键；模型只提供各方面的布尔值、来源选择和解释。宿主按冻结方面顺序还原原有内部列表与总来源集合，拒绝重复JSON键、缺项、额外项、旧数组及来源越权；不去重或追认旧响应。

新schema生成时还核对每个claim的宿主必需方面与最终native分支完全一致，冲突在发送前失败。完全相同的值schema共享定义，去除验证中性的注解，原始支持材料不截断。新旧运行身份隔离，历史v1/v2仅只读；科学结果契约及支持率1.0不变。

审计还发现：压缩注解会丢失旧description中“同一方面内来源ID不得重复”的提示。已把该规则放入实际system指令，并明确不同方面可以复用同一来源；宿主仍检查单方面去重和跨方面16个distinct来源总量。实际发送回归同时检查提示与拒绝反例，不能只证明schema接受集合未变就删去语义说明。

## 请求逐项核对

| 请求 | 实际入口与结构机制 | 与宿主要求的关系 |
|---|---|---|
| 查询规划 | `agents/planner.py::_decompose`；`json_object`＋提示词＋本地字段检查 | 没有原生字段schema。查询数、点名方法及字面覆盖约束在宿主检查；失败按既有规则降级并留记录，不能宣称已有strict schema保障。 |
| 相关性筛选 | `agents/relevance_gate.py`；CrossEncoder/词法后备 | 不是LLM JSON请求，无response_format；计数、阈值与来源字段由程序保留。 |
| Extractor普通单篇/旧批次 | `agents/extraction_strategy.py`；部分路径`json_object` | 提示词字段及本地解析保留历史语义。与新版原生提取分别标注，不全局升级旧接口。 |
| Extractor新字段/grounded批次 | `_field_span_response_format`、`_grounded_claim_response_format`；strict `json_schema` | paper成员、字段、chunk归属、类型绑定一致；token区间顺序/上界、连续原文、非空白及科学支持由宿主/后续核验负责。反例检查跨论文chunk及布尔值冒充置信度。 |
| 五章正文 | `writing_v2.py`请求构造、`writing_flat.py`投影和解析；strict native schema | `text/fact_card_refs/paragraph_start`与接收器对齐。冻结总池引用可跨章；章节名单仅推荐。首段边界、总体容量、全文覆盖及事实语义继续由宿主检查。 |
| 摘要/结论 | `summary_request_input`、`check_summary_request`、`check_summary_provider` | 同一有效约束生成payload、描述和本地schema；配置1/2/3均检查，默认1–2。服务端原生子集不表达的数量边界写入描述且宿主严格检查；旧“3–5/最多6”没有进入新路径。 |
| 当前冻结链Reviewer | `writing_v2.py::review`经`WriterSession.call`；strict schema/Responses `text.format` | 仅`issues`及`explanation`，issue的unit_id原生枚举绑定当前稿。这里不是普通`ReviewerWorker`的旧输出格式。 |
| 普通/旧Reviewer | `agents/reviewer.py`；`json_object`＋字段/引用/语言校验 | 保留已有score、issues、evidence_compliance等契约；不冒称与当前冻结Reviewer使用同一schema。 |
| Evaluation抽取/清单恢复 | `eval/scoped.py::build_evaluation_request`及flat inventory投影 | 最终片段键与宿主范围绑定；恢复不更改原文来源或免除全覆盖。分类、原子性与忠实性由逐条核验负责。 |
| Evaluation逐条核验 | 同一发送构造器＋`decision_output.py` | 本次修复固定方面键。claim/来源目录原生约束保持；跨方面来源总数、正项非空、数组中的claim/unit完整且唯一仍由宿主检查。 |
| Evaluation全局检查 | 同一发送构造器＋`global_permissions.py` | units/conflicts/coverage/limitations/analysis/issues的ID与冻结目录绑定。章节归属、完整覆盖、矛盾语义和最小充分引用集合由宿主/模型检查；不重复逐条原子性/分类/忠实性判定。 |
| 正文/容量/摘要修复 | `repair_request_input`、flat投影和原有接收器 | 来源、可编辑单元、替换字段及删除记录使用当前冻结映射；跨目标关系、所有旧单元的保留/改写/删除账目、最终整节数量及两次额度由宿主检查。失败候选不提交。 |
| 可选Judge | `benchmark/survey_judge.py`的inventory/report两类format | strict schema或显式json_object兼容模式分别保留；提示词示例和真实模型类由回归核对。清单、来源、位置及完整判断仍本地验证，本轮没有真实调用。 |
| 可选Jev | `benchmark/citation_audit.py::_request_payload/_validate_response` | TypeSafe专用questions/criteria协议，不使用OpenAI response_format。返回模型、问题集合、选项、概率和置信度按实际发送清单验证，本轮关闭、无真实调用。 |

## 明确保留的宿主责任

原生schema不能替代所有业务不变量。本轮没有把“数组中每个claim恰好一次”、跨字段来源总数、完整报告容量或科学支持等要求伪装成原生保证。它们在请求中明确说明，并由宿主严格拒绝错误结果。未支持的schema关键字不随意加入；没有通过放宽接收规则消除失败。

## 离线证据与边界

- `tests/test_keyed_assessment_aspects.py`：方面组合、两个来源命名空间、旧065重复、缺项/额外项、重复JSON键、旧owner隔离及结果身份不变。
- `tests/test_request_schema_alignment.py`：Extractor生产schema与解码器的跨论文/类型反例；摘要实际请求、原生描述与宿主数量边界一致。
- `tests/test_effective_summary_contract.py`、`test_native_writing_schema.py`、`test_native_summary_repair.py`、`test_native_candidate_repair.py`、`test_global_reference_closure.py`、`test_report_judge_prompt_contract.py`、`test_jev_citation_audit.py`保留相应生产者/消费者反例。
- `tests/test_evaluation_dimensions_pipeline.py`、`test_independent_evaluation_pipeline.py`、`test_report_repair_eligibility.py`检查真实生成、Reviewer、Evaluation、共享修复、完整复评和自动判定；Mock仅替代外部响应。

测试、完整资源和默认回归的实际终态统一存入本步验收目录；未结束的检查不得记为通过。离线成功不追认065，也不证明新版真实正文已经科学通过。

最终离线结果：默认`pytest -v`实际退出0，9549通过／25排除／8warnings；9个本步Python的Black/isort及`git diff --check`通过。705份最终源与3563份历史保护保持。最终完整请求矩阵和三域预算均通过，保留两次修复；原065第7批的科学输入不变，新native拒绝旧重复响应，32768输出预留不降。新增付费调用0，新v3真实运行尚未验证；不追认旧报告，也不把离线通过计入Survey质量成功。
