# Evaluation 身份契约收敛：步骤246

## 修复对象与保证边界

028 首题的全局响应把同一 a41/f12 同时指向 v9 和 v12，实际父单元只有 v9。原响应保持拒绝；本次离线实现不恢复该运行，不重新计算历史分数，也不声称正文科学通过。原028仍严格0/3、20请求304918已知tokens、两题未运行。

本次保持 Evaluation v3 科学支持1.0、完整覆盖、独立抽取、实际引用原文、全局检查、两次共享修复、450万tokens/64请求及旧策略边界。无新增付费请求。

## 统一字段审查

| 阶段与字段 | 处理 | 理由与消费者 |
|---|---|---|
| 抽取：fragment、原文范围、引用 | 宿主生成，保持 | 模型只提取语义；scoped负责原文忠实性与引用归属，不允许模型捏造位置 |
| 批次：claim_id | 保持 | 将独立判断绑定实际清单；漏项、重复、未知ID继续拒绝 |
| correct_kind / classification_valid | 保持 | 独立分类结果与对已有分类的检查；宿主校验一致，不能靠改分类移出分母 |
| atomic / source_faithful / status | 保持 | 原子性、报告→断言忠实性、论文→断言支持是不同命题，不能互相替代 |
| status / reason_code | 保持 | 汇总状态与具体诊断分别消费；既有一致性规则拒绝相矛盾的响应 |
| support_basis / aspect_checks | 保持 | 整体原文依据与每个必要方面的支持分别校验；均限当前断言实际引用 |
| global units / coverage / analyses / limitations | 保持 | 分别判断原文处理、核心问题、章节职责与具体局限；数量完整和章节身份仍校验 |
| global finding 的 claim_ids 与 source_unit_id | 新路径消除重复 | claim→fragment→unit已经由宿主确定，要求模型再输出父单元只增加身份冲突 |
| finding owner | 继续宿主派生 | inventory/report由code定义，禁止模型提交重复责任字段 |

没有增加Agent、模型、调用次数或新的科学免检类别。只新增显式全局响应版本，以隔离历史契约。

## 新契约与确定性定位

`anchored-global-findings-v2` 的每条问题仅包含 `code`、`anchor_ids`、`explanation`。线上使用输入中已有的 a/v 短ID，宿主将其可逆恢复为带类型的断言/单元锚点，再按冻结父映射生成既有内部定位。不存在按文字查找、借相邻引用或猜测父单元的操作。

已有断言的 non_atomic/source_fidelity/classification 必须选择断言锚点；missing_assertion必须选择原文单元锚点。报告问题可以跨多个位置，所有位置均保留，原finding索引及锚点存入阶段记录。空、重复、未知、类别不符或混合伪造身份拒绝；模型重新提交父ID或owner也被严格schema拒绝。

原 `located-global-findings-v1` 保持旧schema与父归属校验。旧无marker路径不隐式升级。新v7计划的marker、schema/指令/映射身份进入fingerprint；v5/v6算法保持，旧checkpoint不借新版本恢复。

原报告和抽取断言同时出现ICCNN不能仅凭拼写推断为抽取失真：source_fidelity只比较报告与清单。报告内部术语矛盾属于报告问题；论文是否支持正文仍由科学核验判断。本实现没有猜测ICCNN应改为什么，也没有自动修正来源文本。

## JSON解析信任边界

新v2的抽取、批次、全局响应均在原始响应及usage保存后、任何评分前拒绝重复对象键。Pydantic严格字段与原生schema不能替代这个检查；同值重复也不接受。复用既有唯一键解码器并保留阶段诊断，旧v1和无marker路径不隐式升级。响应非法仍计已知用量，不自动重发。

## 最终验收证据

- 默认 `pytest -v`：**5465通过、25排除、8项既有warnings，354.81秒，实际退出0**。534份受验源码/测试保持一致，新增29项回归；相关消费者组68项通过。
- 本次9个Python文件Black/isort与 `git diff --check` 均退出0。
- 54组最大实际构造器向量全部通过：144断言18组、176断言18组、生成/Reviewer/修复18组。最大全局分别116448/117851，均低于含10%上下文余量的117964限制；两次修复和完整核验保持。
- 176项最坏向量仅余113预留tokens。后续提示词/schema变化必须重新证明实际边界，不能认为还有宽裕空间。
- 六份v5/v6三域预算计划与027封存实现逐值一致。2423个原历史路径、1151个本步局部保护文件保持；028实际归档与保护副本逐字相同。
- 旧四格式文件SHA保持，未新生成PDF/DOCX。桌面1280、实际390中英文、键盘展开、page/block原文定位只读验收通过，console错误0，视口和语言恢复。

证明目录为 `artifacts/acceptance/evaluation-identity-20261030/`，核心文件是 `completion-review.json`、`pytest-release-proof.json`、`probe-release-proof.json`、`format-protection-proof.json`、`historical-proof.json` 和 `ui-proof.json`。中间通过、人工中断及测试字段错误日志均单独保留，不冒充最终源码证明。

## 真实质量边界及下一步

本次新增付费0，原028仍严格0/3。其局部诊断98支持/41证据不足/17未核验不是最终评分或人工金标准。17项中含实体未核验10、方面覆盖1、引用归属1、关系1及其他4；没有自动修正来源拼写、别名、数量或引用来提高分数。

下一步使用独立冻结版本及完整费用账本进行有界真实验收，不恢复028，不把这些离线测试算作新正文科学通过。重点检查完整Evaluation是否能以真实诊断进入最多两次修复，同时保持科学1.0、核心覆盖与全部原文。Phase14.5保持 `in_progress`。
