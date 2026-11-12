# Phase 14.5 / 步骤311：语义冲突恢复实施与验收

## 当前结论

用户已明确允许原草案中的新策略。实现、兼容和默认离线回归通过；**最大诊断资源验收未全部通过，不启动新付费套件**。Phase 14.5 保持 `in_progress`，既有真实三域严格交付仍为0/3，本轮没有新的科学通过率结论。

最终源码637文件，摘要为 `341f14501e20326414d23bc7a42abd31cfb8b58946c71683cb441cfafa9a96e4`，详见 [source-final-002.json](../artifacts/acceptance/semantic-conflict-recovery-20261103/source-final-002.json)。所有首红、中断、原始响应和两版源码均保留；未提交或推送。

## 实现行为

仅内部冻结选择 `semantic_conflict_policy=host-unverified-continue-v1` 生效。普通API、工作台默认、旧None/V1/V2口径不变；不恢复052，不追认旧失败。

在原文、引用归属、完整ID、精确aspect集合和实际已结算请求通过验证后，仅两种指定语义矛盾产生独立 `assessment_conflict_effect`。模型raw及canonical原判保持，宿主将该条记为未核验，在科学分母中保留且不贡献支持数量。其余批次和global继续；未核验项仍阻止交付。

只有完整批次、global、清单及当前报告／checkpoint／真实账本重放均通过，才能授予原有共享两次修复。修复指导要求独立复查，不把模型解释当科学金标。候选需更新摘要、结论并完整复评后才可提交；partial、未知usage、atomic/fidelity/分类失败不借此新增正文修复资格。旧11项支持问题白名单未扩大。

预算和实际请求身份包含新选择；实际writer请求因新增双字段每章增加165保守预留，五章需求增加825，完整原文、权限和原None计划不变。三份只读资源计划仍为444／448／448万，单题450万／64、两次修复、输出预留、上下文余量与零重试不变。以上是预留，不是新增费用。

## 实际验收

| 检查 | 实际结果 |
|---|---|
| 默认离线 `pytest -v` | PID38357，exit0，8288通过／25排除／8既有warnings，865.39秒；前后637源一致 |
| 端到端及界面专项 | PID38920，exit0，34通过，外部传输Mock，实际生成／Reviewer／Evaluation／修复／完整复评消费者执行 |
| 两项独立发现复验 | PID37291，exit0，11通过；合法record_mismatch及既有长原文五字段窗口已修，篡改负例仍拒 |
| Black／isort／diff检查 | PID37186，exit0，仅本轮15个Python文件；未格式化已执行观察器 |
| 完整核验请求资源 | PID37237，exit0，26组，全部原文及最大合法请求保留 |
| 修复几何与预算 | PID38415，exit0，36组布局及9条两次最大消耗修复／完整复评路径 |
| 52项完整typed诊断压力 | PID38586，exit1，writer_feedback_capacity_exceeded；135051预留超过117964上下文边界，发送前拒绝，未裁剪、未构造owner、零模型调用 |
| 历史保护 | PID36396，45322原件及630份原源码、既有309／310证据精确映射保持 |

资源结论引用同一最终源码的真实进程，不将attempt003声称为重跑26组。新元数据计量差额、观察器错误、沙箱端口限制和主动中断分别留证，不把它们当科学失败。

52项是完整合成effect表示压力，不认证原047费用或修复资格。它证明现有边界会安全拒绝超大请求，**不证明所有合法问题组合都可发送或修复**。因此不能把默认测试通过写成全部资源门禁通过；剩余问题是修复请求传输规模，不能靠提高总预算、减少科学核验或遗漏问题解决。

## 展示与范围

界面及JSON／CSV／Markdown／HTML分别保留原模型判定和宿主未核验结果。双语、实际390px、键盘切换、合成样例内原文定位已检查，支持数保持0、未核验计数一次，临时视口已恢复。

可浏览的 [离线契约样例](http://127.0.0.1:8020/semantic-conflict-offline-20261103/) 明确标注合成输入。它不代表052重评、真实归档定位、附录或新正文PDF／Word导出验收。本轮没有新的真实报告，旧报告／评分／批准／SHA／已知费用及未知占用不变；新付费调用0，Judge/Jev关闭。

## 证据入口

- [默认测试实际进程](../artifacts/acceptance/semantic-conflict-recovery-20261103/default-pytest-003/process.json)
- [最终资源分项审阅](../artifacts/acceptance/semantic-conflict-recovery-20261103/resource/review-002.md)
- [独立代码与契约审阅](../artifacts/acceptance/semantic-conflict-recovery-20261103/independent-review/final-review-seal002.md)
- [界面范围记录](../artifacts/acceptance/semantic-conflict-recovery-20261103/ui-observations-001.json)
- [用户确认原件](../artifacts/acceptance/semantic-conflict-recovery-20261103/user-confirmation.json)

下一步应先收敛完整诊断请求的无损表示和发送前规模检查，再重新核最大压力；本记录不批准提高边界或启动新的真实验收。
