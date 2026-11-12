# 终态诊断绑定：不再显示已恢复的旧JSON错误

016的Benchmark原先显示taxonomy响应的2330位置，但该调用已由repair_0成功恢复；真正停止运行的是experiments响应的2538位置。原因是展示代码选择journal中第一个JSON错误，忽略了终态引用和resolved_by。

现在按journal.failure.call优先、failed_call其次定位，失败的修复响应可以与其原始调用区分。已恢复、引用冲突或缺失不能借其他调用兜底；旧记录仅在唯一未恢复JSON错误存在时提供位置，多个候选不按字典顺序猜测。

展示还核对原始文本SHA、诊断SHA和实际JSON错误偏移。缺字段、类型错误、越界、坏sidecar或非法路径只使定位不可用，主失败原因和统计保留。不输出科学正文或任意异常字符串，不调用模型。

JSON/HTML及新增CSV标量列、Markdown定位表使用同一结果；展示版本为view-v10，运行源码仍标为原manifest中的fa760bf。旧四格式副本及SHA已保留，原报告、评分、state、journal及费用没有改变。016现在准确显示body:experiments、offset2538、原SHA以b93374a5开头；科学核验仍未运行，交付0/3。

使用冻结fa760bf的实际旧展示函数和相同016输入，独立复现错误的已恢复taxonomy定位，再验证新函数得到终态experiments定位。这个基线证明针对原缺陷，不依赖新增view版本字段断言。[实际export测试](../tests/test_writer_v2_campaign.py)覆盖终态绑定、修复调用优先、历史歧义、过期诊断清除、损坏定位及四格式一致性，均检查原state和sidecar不变。

最终默认离线**4653 passed、25 deselected、8既有warnings**；81相关测试、两个Python文件Black/isort及差异检查通过。桌面1280与390px、中英文、Enter切换、Space展开、终态位置及console零错误/警告已验证；viewport已恢复。此前自动审批额度失败保留记录，本次在只读额度状态确认恢复后通过原审批路径完成，没有切换工具绕过。

64项历史保护和487份旧源码副本保持，原始运行的6请求/83749已知及累计777/4567451已知/1142672旧未知保持；本修复新增模型调用0。完整报告和PDF/Word导出仍不存在，不能宣称该部分验收通过。[EOF语法边界草案](writer-syntax-boundary-proposal-20261026.md)仍待确认，未启用；Phase14.5仍in_progress。

私有证据：artifacts/acceptance/frozen-writer-three-domain-20261026-016/presentation-diagnostic-fix/。
