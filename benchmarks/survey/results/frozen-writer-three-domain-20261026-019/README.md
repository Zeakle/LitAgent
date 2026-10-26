# focused-range-v1 三域真实验证：写作响应不完整

2026-10-26。源码 `de14d184a9eaa11b4b94b753d357110e0899083f`，冻结身份 `db64326d14e4cc8427ec2ef5b0e8327f8a5c4f9a39c9f0d303b739c58aebf91e`。固定few-shot→ViT→NeRF，原完整证据、模型及预算保持；新范围修复计划显式启用，Judge/Jev关闭。这是实际模型调用记录。

**严格交付0/3，完整生成0/3，Evaluation完成0/3。** few-shot发出2次请求；ViT与NeRF按协议故障停止而未运行，保留在计划分母。支持／不支持／未核验科学断言数量不可用，不记成零错误或满分。

引言完成4个单元；方法分类请求携带目标6、最大10及完整来源，Pro enabled/low返回HTTP200、incomplete/length。输入10306 tokens，20000输出全部是reasoning，没有message正文。该项触及既定20000输出上限；不是300秒HTTP超时或500000运行总预算耗尽，也不是Evaluation判定事实错误。思考effort=low并不保证每次能在输出额度内生成答案，本次不据此推断供应商内部原因或所有模型能力。

已知2请求／40843 tokens／226.056秒，0重试、0语义修订、0新增未知。两请求总reasoning 20222已含输出，不重复累计。新范围修复尚未被本次真实调用触发，不能声称其语义效果已验证。

累计790请求／4791977已知tokens／1142672旧未知保守占用；历史Jev67、Judge0保持。完整响应、原始材料和运行记录仅本地保存。LangFuse本机2个generation与用量读回一致、科学输入输出为空，未宣称完整trace树验证。

四格式同源统计见 [HTML](benchmark.html)、[JSON](benchmark.json)、[CSV](benchmark.csv)、[Markdown](benchmark.md)，[停止诊断](evaluation-diagnostic.json)。本次不恢复019、不追加付费至通过、不扩展15题、不改变门禁或输出额度。后续先离线审查Pro写作thinking策略与结构化输出契约；任何新的配置验证须独立绑定，不能回写本次失败。Phase14.5保持in_progress。
