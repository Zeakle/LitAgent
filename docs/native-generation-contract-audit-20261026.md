# 016之后的生成契约审查：已证实与尚未证实

当前真实问题仍在生成协议层，尚不能归因于Evaluation判断过严。[016](../benchmarks/survey/results/frozen-writer-three-domain-20261026-016/README.md)中，六个Pro请求用量完整，三份原章节响应在JSON末尾失败；分类、方法两次独立保真恢复后，综合比较章的第三个错误触发protocol停止。没有完整正文、摘要、Reviewer或Evaluation，严格交付0/3。

## 实际链路核对

| 项目 | 本地证据 | 结论边界 |
|---|---|---|
| 发送内容 | 六份私有HTTP请求及487份冻结源码 | 完整原文、既有供应商投影schema及20k上限均发送；宿主保留完整本地约束，二者分别绑定hash |
| 宿主/供应商schema | 六份host schema独立重建，按原provider_response_format投影后与wire逐值一致 | 数量/长度约束由宿主继续强制，不把两个不同hash误判成丢字段 |
| SDK与正常化 | 原包全部message/output_text拼接与journal逐字一致；安装SDK→Synthesis六份原包逐字重放 | 未发现客户端丢末尾字符、挑选部分答案或静默裁剪；不排除供应商内部原因 |
| 模型与模式 | 实际Pro模型符合冻结配置，thinking=enabled、effort=low、temperature=0.1均回显一致 | 本轮没有Lite请求，不能宣称Reviewer修复已获真实验证 |
| 输出与费用 | 每次实际输出均低于20000，83749总用量低于500000，2409思考已包含 | 不能用增加输出、上下文或超时作为本次已验证修复 |
| 服务层级 | 六个响应的service_tier均为default | 未发现本次误用不支持结构化输出的保障包模式 |
| strict标记 | 六份请求/响应的完整text.format相同；三份内容仍非法 | HTTP完成、schema回显和strict标记不能替代宿主校验 |
| 修复次数 | 两次保真恢复；第三个错误保留原protocol根因，未发第三次恢复 | 次数耗尽分类修复获真实证据，科学质量仍未核验 |

这些结论针对已保存的六份原包，不能扩大成对全部运行或供应商内部实现的证明。原包、完整正文片段和请求标识保留本地，不写入公开包。

## 官方文档核对

2026-10-26重新读取[创建Response参考](https://console.volcengine.com/ark/region:cn-beijing/docs/create-model-responses-api)及[结构化输出说明](https://console.volcengine.com/ark/region:cn-beijing/docs/responses-api-structured-output)。前者分别列出schema严格标记、思考开关及包含思考的总输出上限；后者仍标记为beta，并指出保障包模式的限制。实际六个响应均为default，因此该限制不能解释本次失败。

结构化输出的Curl示例关闭thinking，但这不等于文档禁止开启；同页Python示例没有显式指定该字段。因此，**尚无足够依据将“thinking与schema不兼容”认定为根因，也不据此切换配置并再次付费**。文档中支持某种能力，也不能反过来否定实际收到的非法响应。

## 剩余条件与下一步边界

本轮没有发现足以解释三个EOF错误的新客户端字段或解析缺陷，不能为了产生一个提交再增加输出编码、降低校验或补括号。保留现有合法响应正例、非法响应反例及原始身份，后续若调整实现，须先有可复现的本地缺陷证据或明确的供应商契约依据。

目标仍缺三份完整正文、独立科学核验、严格交付及对应导出。旧015的未核验中间稿、离线测试和当前四格式Benchmark均不能替代这些条件。016首次390px、语言切换、键盘及截图动作因自动审批额度耗尽未执行；203在状态恢复后补齐，并修复了[错误位置绑定](terminal-diagnostic-review-20261026.md)，不把已恢复的早期错误当作终态。

审查新增付费调用0，016运行源码为fa760bf；203只修改展示读取及测试，生成和核验接受规则未改。64项历史保护、487份旧源码副本、累计777请求/4567451已知tokens/1142672旧未知占用保持。Phase14.5继续`in_progress`，不恢复016、不追加到通过。
