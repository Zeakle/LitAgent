# 040 三域真实验收结果

2026-11-01，用户确认账户已恢复后，按已有有界授权单次启动040。套件计划三题、实际运行一题，技术停止后其余两题未发送。Phase14.5保持in_progress，未达到三域严格通过目标。

|项目|实际结果|
|---|---|
|生成契约|few-shot五章48单元，摘要1、结论2；Reviewer完成|
|首次核验|167断言、11批和全局请求完成；11条非原子，清单无效|
|共享修复|1/2，用于同正文清单恢复v2；没有正文改写或候选提交|
|恢复后核验|168断言，首批响应一致性无效，0项有效处理；150科学断言未核验|
|套件终态|stopped / evaluation_decision_consistency_invalid|
|计划分母|3题；生成1、有效Evaluation0、严格交付0；ViT/NeRF未运行|
|新增用量|22请求、344347已知tokens、0新增未知；零重试|
|历史累计|1032请求、8427025已知、1283047旧未知；038的78205未知保持|

原会话6912实际exit0，controller99495、child99525，运行1083.876秒。退出0表示执行器按规则停止，不能当科学通过。每题444/448/448万、套件1340万是启动时上限，不是实际费用；本轮没有预算耗尽或账户错误，也没有走完三题完整预算工程验收。

## 原因与下一修复方向

恢复后raw的claim_2/3/15均给出unsupported/attribution_mismatch，同时所有必需aspect为true，违反现有一致性契约。宿主原样拒收，未修改模型判断。原文段落的共享引用集合与每条断言的支持basis没有被充分区分：模型解释因集合中还有其他论文而拒绝，尽管选中basis的主体与原句对应。必须离线澄清这两个概念，保留错误归属、条件和别名守卫；本次不能据此把三条改成支持，也不能推断整体正文科学正确或错误。

[完整replay与逐字段审查](../artifacts/acceptance/run-budget-display-live-20261101/launch040/observer-review/040-decision-consistency-review.md)保留原文、全部实际来源、raw、basis和真实拒绝结果。不恢复040，不重发失败响应，不追加付费样本。

## 可浏览结果与导出

- [Benchmark](http://127.0.0.1:8020/frozen-writer-three-domain-20261101-040/benchmark.html)：JSON/CSV/Markdown/HTML同源，失败与未运行保留分母。
- [few-shot质量与草稿](http://127.0.0.1:8000/flow-demo?run=archive%3Afrozen-writer-three-domain-20261101-040-few-shot-metric&view=quality)：只读presentation，已知344347、未知0、22请求；科学分数显示跳过/未核验。

8000仅精确两份038与一份040缓存，原owner82754已校验后替换为2532，lifespan off，无生成/核验/恢复服务。ASGI三条view及四类写拒实际通过；8020和Docker不变。13张固定尺寸真实截图覆盖1280×720/390×840、中英文、样式切换、同SHA证据定位、Esc及折叠附录；另外三张过渡/尺寸错帧和默认恢复帧保留，不冒称通过。仅r1，未宣称多版本竞态验收。

应用原生MD/PDF/DOCX与BibTeX/RIS/CSL JSON均实际导出，继续标草稿。171引用occurrences、9篇书目，九项作者和年份均缺失，保留54元数据提示，不补造。PDF7页与修正本地字体配置后的Word13页全部视觉检查；首轮Word7页缺字保留，修正只影响bundled渲染环境，原文件SHA不变。全文、引用、书目和附录来源一致不等于科学通过。

## 证据与保护

- 源570，digest 91f52e8c682922c9b4feedbf2116e17be7f58ab8b3a7db3d9e1eb221deda3b93。manifest 0f3ca5718d011475f4cd0886a2aa20b45b7307506a63afcd86d290c191b8caad。
- 报告r1 SHA b03443c33948f5faa6d14711b7ecd6ee3aea2a2bfc0b5c8d722f68fdc2331566。
- [一次独立终态审计](../artifacts/acceptance/run-budget-display-live-20261101/launch040/independent-terminal/review-001.md)：PID1911/tool1793实际exit0，570源码和16210保护不变，39父费用链保持，全部22请求usage与实际wire闭合。
- [真实UI证据](../artifacts/acceptance/run-budget-display-live-20261101/launch040/ui040/ui-proof-001.json)、[导出全页QA](../artifacts/acceptance/run-budget-display-live-20261101/launch040/exports/visual-review/QAproof.md)。

本步没有业务代码变更，沿用同源默认pytest6451通过/实际exit0和最大002十二项验证，不虚增新测试；没有commit/push或Plan.md架构修改。源码、历史报告、评分、费用、原批准及失败保持。本轮证明摘要契约和真实扩批已前进，尚未证明完整科学验收闭环。
