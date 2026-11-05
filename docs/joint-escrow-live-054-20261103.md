# 054 三域真实验收记录

本轮完成固定三域运行，三份正文、摘要与 Reviewer 均完成，严格交付 **0/3**。三题的核验请求均有结果，但整体 Evaluation 无效，不能用局部支持数量计算有效科学分数。Phase 14.5 保持 `in_progress`。

## 冻结范围与执行证据

- 套件：`frozen-writer-three-domain-20261103-054`；manifest identity `72f8af3bc9d63bef34d10fbda8cb9afe928d217fe80f3742e6c6c3245657fb86`。
- 源码：650 文件，digest `ec1cded273ed68becc60ef411c70bc768d8e94c34c2e080e34f50e28eaaec113`；运行前后逐文件相同。架构 `plan/Plan.md` 保持 `b710b028f044270d1f3ad37d6582f09fedd9da470d6566c827c839145020fffc`。
- 唯一 controller wrapper 10233／child 10234，实际退出 0，用时 4079.39 秒；该退出码表示套件正常收束，不表示质量通过。日志 SHA `d0dfdefc4a0ca8e825b36e9bdbd2876be87403a36dd76df4aba0519edbfd4bc4`。
- 原 few-shot、ViT、NeRF 冻结材料各一次，未重新搜索或提取。Pro 写作，Lite 审阅与 Evaluation，思考关闭、传输重试 0；Judge／Jev 关闭，无人工审批。
- 同一源码继承步骤320的默认离线验收：actual7157 退出0，8711通过／25排除／8 warnings；最终专项68项及9个Python文件格式检查通过。本步没有修改产品或测试源码，未重复运行默认测试。

## 三域结果

下表支持数量仅为诊断分布；三题都没有有效的整份科学分数。

| 题目 | 清单／核验批次 | 科学支持／不支持／未核验 | 共享修复 | 严格交付 | 实际请求／已知tokens |
| --- | --- | --- | --- | --- | --- |
| few-shot | 173条／11批，1轮 | 121／33／10 | 0 | 阻止 | 20／308020 |
| ViT | 每轮185条／12批，3轮 | 最后112／22／44 | 2次清单恢复，正文修订0 | 阻止 | 49／790790 |
| NeRF | 105条／7批，1轮 | 72／13／15 | 0 | 阻止 | 16／255858 |

三份正文SHA分别为：

- few-shot：`cf6cc1575c18e4dffa94e0f0f58342c45b48b694180d1964e5eafb8263ff8e8b`。
- ViT：`e82fb30ce25bc32ee9c8ea060e271a0b35b7479018ca1f1520f40cfda5d588f2`。
- NeRF：`135448248b3360456119c9917015e155d99bc59a3b838817a37e2aea9a62dee9`。

few-shot、NeRF的inventory通过、各批与global完成，但宿主仍有未核验项。ViT两次恢复后inventory仍失败；没有正文候选或提交，不能把预算stage=repair解释为正文修订。

## 已定位的阻断

1. **方面契约未完全投影到native schema。** few-shot一条、NeRF八条方面覆盖错误足以阻止修复授权。来源权限已绑定，但每条claim要求的方面集合未同样收窄；宿主继续拒绝多项、缺项或重复，未删字段追认旧结果。
2. **清单恢复和评估维度混用。** ViT首轮七条被判非原子，第一次恢复未实际拆分；后两轮分别由claim55、112的原文忠实性判断触发失败。理由使用“论文没有陈述”否定“报告是否陈述”，存在可定位的维度混用疑点，原判保留。
3. **提取候选超出自身绑定原文。** ViT c17的finding_candidate含若干训练条件，绑定原文没有这些条件。实际writer同时收到未核验标记和完整原文，结构化conditions为null；不能将词语重合当成已证明的因果，也不能因此将正文判为有原文支持。
4. **科学解释与宿主词义边界仍需处理。** few-shot存在机制、成本或比较推论未获其实际引用支持；NeRF还有参数化等词语触发parameter_count、5D及实体匹配问题。部分负判理由未解释另一允许来源为何不足，已保存真实权限和完整来源，但不直接翻判，也不当人工金标。

core单项满足、global文字评价肯定或局部supported，都不能覆盖其他不支持／未核验项。没有人工校准，不报告准确率、误放行率或误拒绝率。

## 预算工程与费用

三题锁定额度444／448／448万，合计1340万tokens，硬上限192请求；每轮联合资源域分别保留N=15、C=100000、E=800894／802594／802380。预留不是实际费用。

本轮实际 **85请求／1354668已知tokens／新增unknown 0**，没有预算耗尽、规划违约或请求身份碰撞。ViT完整执行三轮与两次共享恢复，说明这些实际输入下后续额度没有被前置阶段挪用；这不是任意模型输出均可容纳或科学必过的证明。

52父记录保持1488请求／15417163已知tokens／1283047历史unknown。累计1573请求／16771831已知tokens／1283047历史unknown；旧占用不释放、不重发。Jev／Judge新增请求均0。费用审child16705实际退出0，3292审计输入与650源码前后相同；四种展示文件与逐请求账本一致。

## 展示与导出范围

[终态Benchmark](http://127.0.0.1:8020/frozen-writer-three-domain-20261103-054/benchmark.html)显示计划3、运行3、生成3、有效完整核验0、交付0；失败全部保留分母。JSON、CSV、Markdown、HTML同源统计已核对。

三份草稿均已通过真实产品消费者导出Markdown、PDF、Word、BibTeX、RIS和CSL JSON，共18文件；未重导。PDF20页、Word35页，共55页，全部按原尺寸逐张检查，未见裁切、重叠或缺字，书目、附录及草稿标记可读。结论是**核对完成且有限制**，不是全部检查全绿：

- 首次导出观察器在末尾守卫断言失败；具体被拒事件未即时记录，类别保留未知。原脚本、首错和已生成文件不覆盖；后续只读观察器另存，不重marker、模型或导出。
- 原书目观察器将BibTeX合法TeX转义与Unicode标题直接比较，首红保留；后续按实际契约逆解核对。
- NeRF Word渲染PDF的两处希腊字符／标点在raw文本提取中次序不同。XML与图像完整，但该raw顺序检查仍标失败，不用宽泛normalize消除限制。
- 长SHA折行及少量跨页分段仍有精细排版空间；不宣称投稿级排版或公式能力已全面认证。
- 原归档的产品书目catalog为空，作者、年份和期刊等缺失警告保留；导出没有借其他运行或外部来源补齐。样式应用不等于书目信息完整。
- 8020桌面中英文及键盘展开／切换已验。两次请求390px后实际仍1280，移动端本次未验；8000未挂载新归档，新报告的原文定位和折叠附录UI未验，没有重启服务。

## 后续优先级

先做纯离线修复：逐claim方面约束与宿主一致；分清报告忠实性、原子性和论文支持；核对卡候选与绑定原文的边界；为参数词义、数字表示和材料缺失判断建立成对反例。不得仅扩大修复白名单、翻旧判或增加预算换通过。

保留本轮三份正文、全部失败、SHA及费用；不恢复053／054、不追加运行直到通过、不自动进入15题、不推广默认策略或提交代码。新的质量改动应先绑定明确契约和真实失败回归，再安排独立的有界验证。

## 本地证据入口

- `artifacts/acceptance/joint-escrow-live-054-20261103/controller-terminal-001.json`
- `artifacts/acceptance/joint-escrow-live-054-20261103/cost-review-001/review.md`
- `artifacts/acceptance/joint-escrow-live-054-20261103/result-review-001/proof.json`（完整私有诊断，体积较大）
- `artifacts/acceptance/joint-budget-implementation-20261103/live054/few-shot-review-001/review.md`
- `artifacts/acceptance/joint-escrow-live-054-20261103/vit-review-001/review-002-terminal.md`
- `artifacts/acceptance/joint-escrow-live-054-20261103/nerf-review-001/review.md`
- `artifacts/acceptance/joint-escrow-live-054-20261103/export-qa-001/content-audit-003.json`
- `artifacts/acceptance/joint-escrow-live-054-20261103/visual-nerf-001.json`、`visual-vit-001.json`及`export-qa-001/visual-few-shot-001.json`
