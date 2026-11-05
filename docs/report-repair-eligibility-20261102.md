# 完整合法评估后的正文修复资格（2026-11-02）

本次完成步骤304的离线工程验收，Phase14.5保持in_progress；没有运行新的真实Survey、付费请求、历史owner恢复、服务变更或提交推送。049原报告、严格0/3、评分、SHA和费用不变。

问题是runner允许完整、精确定位的未核验报告尝试修复，writer却因coverage.complete=false整体拒绝。新例外仅作用于当前完整V2服务端策略，允许既有11类host语义支持未建立的问题进入原两次共享HTTP修复。它不修改原coverage、科学分数或通过状态；未知来源、非法aspect/basis、分类/忠实性错、不完整评估、未知用量仍阻断整报告。实际缺引用的科学负判继续允许定向修复。

认证复用真实来源账本、已结算allocation、全部batch raw→短ID恢复→canonical→host重放、精确native定位、当前run既有checkpoint和未提交候选父链。两种input SHA使用不同序列化，分别与真实checkpoint记录核对；这不是完整messages逐wire重建，也不对敌对整体替换可信本机文件作安全保证。跨批平衡仅在当前路径允许唯一完整ID集合，旧helper默认、None/v1语义保持；新V2身份为16ec9f6bcf1a85a5647b52c79183fbe247e47cfbcbac65b96121790bf5420d0c。

## 最终同源验收

- source：624文件，9d7af5bc9af268ac885c7198f793b1d00c63ece01e3e5e1aa10f58b297cc2c26；4生产文件和1新测试文件变化，18个已有sender/science函数AST保持。
- 默认pytest -v：7932通过、25排除、8既有warnings，真实exit0；actual PID 91202。
- 相关回归：960通过，真实exit0；新增72例覆盖11语义原因族、两次共享修复及完整复评、raw/来源/身份/费用/定位负控、真实batch平衡、checkpoint和prior链。
- 独立审查：8条实际消费者正反链exit0，包含一次mixed清单恢复加一次正文修复；只Mock外部传输，不是科学gold。
- 26组最大请求和9条完整两修复预算路径实际exit0；max global 117634≤117964，batch 88848，recovery 106626。三计划444/448/448万，合1340万，仍两共享/64；这些是保守预留/合成路径，不是实际费用。
- 5个本次Python的Black/isort及git diff --check实际exit0。41466旧保护文件逐SHA保持，Plan.md保持原SHA。

完整证据在artifacts/acceptance/report-repair-basis-20261102/。原始首红全部保留：两项input SHA合法值篡改是实际资格漏洞，已由checkpoint认证修复；首默认7931通过/1个loopback绑定PermissionError的exit1保持，未改测试，解除本地测试沙箱后同源完整重跑；并行ImportError、fixture状态/枚举、balance及独立观察器过强断言是开发或观察器错误，不能混称科学失败。历史84科学矩阵和36修复wire几何未在本步重跑，不能借26/9或18 AST相等冒称重跑。

本轮没有新报告的UI或导出验收，也没有真实科学通过率结论。后续须另行固定新输入身份做真实质量验收，不能恢复049或把当前离线绿改写为旧成功。
