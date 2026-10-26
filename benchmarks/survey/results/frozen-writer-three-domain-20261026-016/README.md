# 三域真实验收016：两次恢复后第三个章节JSON仍非法

**严格交付0/3，质量目标未完成。** few-shot的引言、方法分类、代表性方法三个章节通过本地契约检查；综合比较章节在2538字符末尾发生JSON错误。两次共享恢复额度已经使用，系统保留`writer_json_invalid / protocol`与次数耗尽诊断，不发送第三次恢复。完整正文、摘要、Reviewer和Evaluation未完成；ViT/NeRF按技术停止规则未请求，仍保留在三题分母。

- 冻结源码`fa760bf6c1e162759482f979285a82fd94e3e37c`，487源码，identity`2f9c151744144af47ba3b49775d234b63d0d5ee18d8b37801d9d323ff619624d`；原`grounded-writer-v1`、提示v3及恢复v2不变。
- 原三域20/20/18卡、9/12/11篇完整材料保持，17完整父账本、60旧保护。每题500000 tokens/64请求、总1500000/192、两次共享修改、零重试、科学支持率1.0及全部门禁保持；不重搜提取，Judge/Jev关闭，无人工审批。
- 配置为Pro enabled/low、Lite Reviewer disabled/无effort、Evaluation disabled；**本轮实际只有Pro请求，未能真实验证Reviewer配置修复的效果**。
- 共6次请求、83749完整已知tokens、0新增未知占用、223.977秒。73649输入＋10100输出；2409思考tokens已包含在输出中，不重复计费。没有达到20000单次输出或500000总预算上限。
- 6次HTTP200/completed，原生strict schema回显与请求相同，仍有三个原始章节响应非法JSON。分类、方法各一次独立新响应通过原schema及全部内容/引用/段落顺序保真校验，消耗2/2共享额度；第三个非法响应保持失败。协议耗尽不再被改记为普通科学失败，套件确实停止后续付费。
- 六份真实原包经安装SDK→Synthesis逐字离线重放，所有实际请求字节一致，复现同一protocol终态及2/2次数，没有第三次恢复，0新增外部调用。只读扫描能找到完整单元不代表科学支持，也不授权宿主补括号接受原响应。
- 最终报告SHA与完整中间稿SHA均为null，科学分数不可用。原015未核验中间稿仍保留，不能将它当作本轮正文。
- LangFuse读回6个generation、83749tokens、2409思考明细及空科学IO；不声称完整业务链或科学内容正确性已验证。
- 累计777请求、4567451已知tokens、1142672旧未知占用；Jev67、Judge0保持。全部历史失败及费用保留，没有退款、清零或追加到通过。

[HTML](benchmark.html) · [JSON](benchmark.json) · [CSV](benchmark.csv) · [Markdown](benchmark.md) · [脱敏诊断与分阶段用量](protocol-diagnostic.json) · [Reviewer离线修复](../../../../docs/reviewer-role-review-20261026.md)

本轮没有在运行中改源码或恢复终止任务。旧的两次恢复后协议分类缺陷已由真实停止行为验证，但生成协议可靠性问题仍存在；这不是科学支持率零分。下一步只离线审查现有生成契约与供应商响应边界，不提高预算、不补JSON、不追加付费尝试或新增另一套输出编码。Phase14.5保持`in_progress`。

## 后续只读展示修复

终态定位升级为view-v10：原先错误指向已恢复taxonomy/2330，现在JSON、CSV、Markdown和HTML准确指向experiments/2538及原SHA b93374a5...。旧导出副本保留本地；原始报告、评分、state、journal及费用不变，运行源码仍为fa760bf。桌面/390px、中英文和键盘验收已在自动审批服务恢复后补齐，页面console零错误/警告。详见[定位修复说明](../../../../docs/terminal-diagnostic-review-20261026.md)。未核验科学内容不能因展示修复变为可交付。
