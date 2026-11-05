# 报告展示与导出收尾：2026-10-31

本记录对应 Phase 14.5 / 步骤260，只报告离线工程与实际视觉检查范围。Phase 14.5 仍为 `in_progress`，没有新增项目付费请求，没有恢复033或修改其评分、正文、SHA与费用。

## 修复内容与复现依据

- 目录：033样例的旧平滑滚动终态能够准确到达摘要和参考文献，未复现持久定位失败；不能把动画中间帧当根因。新导航限定当前报告内唯一标题，展开附录、移交焦点，再即时滚动实际容器，避免全局目标与动画中间态不确定性。
- 移动历史菜单：390px打开后切到1280px，旧遮罩仍命中页面，缺陷已真实复现。统一关闭函数和响应式断点清理状态，桌面CSS额外隐藏移动遮罩；同步ARIA，保留对话框、Inspector和菜单的Escape优先级。
- PDF：实测当前Story忽略常见的标题保持同页CSS。改用实际位置测量，对孤立标题追加引擎支持的前分页；每轮只增加有限标题集合，保持完整HTML和链接。
- Word：标题保持后续段落，调整列表行距和段后间距；不删除正文、引文或附录。

生产改动限于 `report_export.py` 与工作台 `app.js`、`report-view.js`、`styles.css`、`index.html`。相关新旧测试消费者同步真实依赖，不增加生成协议、公开API参数、科学豁免或模型调用。

## 最终离线结果

| 检查 | 结果 |
|---|---|
| 默认 `pytest -v` | 5633通过、25排除、8既有warnings，370.78秒，实际退出0 |
| 源码绑定 | 544个文件与最终封存一致 |
| Black / isort / diff | 本步6个Python文件及`git diff --check`通过 |
| 最大请求 | 90组向量通过；最大全局请求117943，小于保留10%余量后的117964 |
| 完整资源路径 | 三域两次共享修复及最终完整复评、三域清单恢复路径均通过 |
| 三域预算准入 | few-shot443万、ViT447万、NeRF447万；合计1337万，每题41计划请求、两次共享修复 |
| 兼容 | 六份旧v5/v6计划相同；031原始结论3/2仍拒绝，不删项追认 |
| 历史保护 | 7476个受保护文件逐hash保持 |

首轮全量实际退出1的记录保留在 `interim-consumer-fixtures/`：7项旧测试消费者/缓存假设失败，另1项恶意图片测试在本机socket监听前被沙箱拒绝。前者补真实helper和平台DOM，保留选择、身份与竞态断言；后者不改Mock、不跳过，在允许loopback监听的环境重跑。最终全量与预算证明均重新绑定最终源码。

## 实际草稿与导出

只读使用033的完整未核验草稿，SHA为：

`465e65179a04a51ad56dc35e1f695b7a368ddfce1614aeec3d6c4e69e1d8f353`

新导出与旧稿保持122条证据出现记录、9篇实际文献、54项元数据警告。122条记录不等于可见编号字符串数量；核对使用原可见序列及宿主映射。

- PDF由7页排为6页，Word由12页排为11页，全部17页逐页看图；孤立章节标题、短尾页、缺字和遮挡已检查。
- Markdown、BibTeX、RIS、CSL JSON逐字保持；Word正文文本相同，PDF及Word渲染文本规范化后相同。
- 六类旧导出均未覆盖；新输出另存。草稿不可交付标记保持，缺失作者、年份等未自动补写。
- 桌面与390px、中英文`document.lang`、键盘目录/附录、引用Inspector完整原文、IEEE/APA切换、033→031→033历史切换均实际核对；科学支持及一致性仍显示未核验，预算预留与实际费用分开。

视觉范围仍有明确边界：原生外侧鼠标点击未完成。390px下全屏scrim的中心在侧栏后方，中心locator点击不能证明外侧点击；原生Codex控制不可用，备用Chrome启动中断。键盘与断点行为已实际通过，但不将这一边界冒充完整视觉退出。033只有一个实际revision，多revision隔离由现有离线消费者测试覆盖。

临时52820只读服务已停止，原8000/8020服务未重启。稳定草稿入口为：

[033报告页](http://127.0.0.1:8000/flow-demo?run=archive%3Afrozen-writer-three-domain-20261031-033-few-shot-metric&view=report)

## 科学结论与后续

033真实首题完成五章、摘要/结论及Reviewer，摘要契约修复确实经过真实请求。但Evaluation在宿主分类/异常诊断处终止，后两题未运行，严格交付仍为0/3。该次8请求、117462已知tokens、0新增未知占用保持；本步不改写结果。

259的新独立分类修复只有离线证据，本步排版和测试通过不证明其真实模型语义效果。后续仍需补齐视觉范围，并在独立冻结输入与预算下验证完整Reviewer→Evaluation→必要修复→完整复评链路；不恢复033、不追加直到通过、不推广默认、不运行15题。本步未commit或push。

## 证据位置

本机完整证据位于 `artifacts/acceptance/report-presentation-20261031/`：

- `completion-review.json`、`pytest-release-proof.json`、`source-seal.json`
- `maximum-resource-proof.json`、`offline/probe-release-proof.json`
- `format-proof.json`、`preservation-proof.json`、`031-replay-proof.json`
- `ui-proof.json`、`ui-before/`、`ui-after/`
- `export-agent/visual-and-content-proof.json`及`draft-exports/`
- `review-agent/step259-read-only-review.json`、`step260-export-read-only-review.json`

独立审查未发现新增实质工程缺陷，但不背书真实模型正确率或三域科学质量。
