# 正文 EOF 恢复兼容修复与真实验收边界

021 三域真实验收在 few-shot 分类阶段停止：2 次 Pro 请求、21,843 已知 tokens、0 新增未知或思考 tokens。Reviewer 和 Evaluation 尚未运行，ViT 与 NeRF 未发请求，严格交付仍为 **0/3**。不是预算耗尽；预检与运行的完整预算计划一致。

分类响应的 HTTP 状态正常，但 642 字符的 JSON 在末尾缺少 `]}`。原始 HTTP 文本、SDK 输出和本地 journal 逐字一致。已有的 EOF 容器恢复策略允许只补扫描器确定的末尾容器闭合符，完整 schema 及原文保真校验仍必须通过；其 footprint 消费者只识别旧 `items` 路径，未识别新有界 `prefixItems`，因此提前报 `writer_reemission_schema_ineligible`。这是可复现的工程兼容缺陷。

本次只修复该消费者：

- 遍历所有章节、段落及句子槽位；支持旧 `items` 和新 `prefixItems + items:false`。所有科学单元 schema 必须一致，各段容量可以不同。
- 完整 schema 继续约束引用、非空字段及容量；原文、引用顺序、字符位置和重复 key 检查保留。
- 扫描器和后缀算法不变，不补正文、不重排单元、不修字符串或中间标点。恢复结果仍标记 `scientific_assessment=not_verified`。
- `bounded-eof-footprint-v1` 仅加入 v4 的运行、预算 builder 和 Session 身份。旧 v2/v3 保持，新代码拒绝复用旧 v4 checkpoint。

测试直接使用 021 原始 taxonomy 文本，SHA 为 `2bae3baad327c97e4a6a54f763bc14a9ff7dea41937194b2f290938f2a8731a4`。真实 SDK 仅替换 HTTP 传输，验证原 raw、usage、receipt 和恢复产物分别保存、不重发。另覆盖正文与摘要、多段非对称容量、坏槽位、未知引用、残缺科学单元及同路径恢复冲突。科学不支持反例经过编译和 Evaluation 后仍阻止严格交付；这是离线契约证据，不是人工校准或真实通过率证明。

021 原记录、正文状态、费用、88 个历史保护路径及 507 个冻结源码副本保持。离线修复不恢复 021，不追加付费请求；每题 100 万 tokens 上限、完整核验和支持率 1.0 均不变。

最终默认回归与源码封存证明见本地 `artifacts/acceptance/bounded-eof-compat-20261027/`；公开结果页分别展示原失败与离线后续，不把本地语法恢复算成科学通过。Phase 14.5 继续 `in_progress`。后续仍需独立冻结验证，证明正文生成、Reviewer 和完整 Evaluation 实际走通；较小正文容量及零修复预留仍是当前限制。

最终默认离线回归为 **5,087 passed、25 deselected、8 既有 warnings**，305.42 秒、退出 0。新增 39 项回归，3 个改动 Python 文件的 Black/isort 及 `git diff --check` 通过；509 个源码文件封存后保持。相关 209 项与新增 39 项包含重叠，不相加统计。
