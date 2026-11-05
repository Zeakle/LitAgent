# 修复任务视图：048 真实验收

048 已执行并封存，未恢复047。实际 controller wrapper55834/child55874 退出0，437.79秒；生产终态是 `stopped / writer_text_invalid`，退出0只表示失败归档正常完成。

## 结果与直接原因

固定三题中运行1题、生成1份、Reviewer完成1题；完整Evaluation与严格交付均0/3。ViT、NeRF因协议失败未启动，仍保留在计划分母。Judge/Jev关闭。

few-shot原稿包含46个正文单元及摘要、结论各1个单元。完整清单抽取210项，其中197项科学断言、13项报告组织陈述；科学支持批次尚未执行，197项均为未核验，不能把0支持/0不支持解释成科学零分或通过。

完成阶段结算后，宿主计算当前完整流程可容纳208项/13批；实际210项/14批，因此进入共享容量精简。第1次Pro修复确实使用了新的 `host-repair-task-view-v1`，完整请求预留88,188，返回完整JSON、24个patch。但14个patch的单元 `text` 含真实换行，违反既有“模型只写单元正文、宿主管理布局”检查，原响应被拒绝、候选未提交。

只读重放PID57649从实际冻结envelope、完整authority和真实请求构造器重建system/payload/schema，SHA与原记录一致；原始响应仍抛出 `writer_text_invalid`。没有删除换行、改写响应或追认修复成功。当前错误只记录代码、缺少字段位置，发送指令也没有明确说明单元内换行禁令，是下一步需要对齐的工程契约。

## 用量与保护

本轮9请求、144,411已知tokens、无新增未知占用；其中1次修复HTTP也正常结算。独立终态审计PID56937退出0，一对一核对raw、SDK receipt、Evaluation进度和账本。累计1,264请求、12,071,787已知tokens、1,283,047历史未知占用保持，038的78,205占用不释放。

冻结源码618文件，digest `8bf7f14ecc318fb4d85aa27739a28bcb935d66cac20022c027091df0eda832ac`；当前源码、冻结副本、39,035个既有保护文件及架构Plan均保持。未重新检索、提取、恢复、重发或追加运行。预算444/448/448万只是锁定上限，不是实际费用。

## 展示与导出

JSON、CSV、Markdown、HTML实际PID57443核对同一分母和统计。8020结果页完成桌面1280×900、390×844、中英文和键盘展开检查；表格使用局部横向滚动，页面本身无横向溢出，临时视口已恢复。

8000现有服务没有挂载此归档，实际返回 `Flow run not found`；未重启或更改服务，因此新原稿的引用点击、折叠附录工作台验收仍未完成。

仅对确实存在且SHA绑定的few-shot原稿导出Markdown/PDF/Word/BibTeX/RIS/CSL JSON，未使用候选或journal回退。PDF7页、Word12页共19页逐页视觉通过，完整正文、中文、草稿标记与链接保留；9篇书目、54项原元数据警告保持。PDF annotation数组与DOCX relationship数组的原始顺序不同不作为阅读顺序证明，实际几何链接顺序另行核对。导出成功不代表科学通过。

实际证明位于 `artifacts/acceptance/repair-task-view-live-20261102/`：controller-result、failure-replay-proof、terminal-accounting-review、export-qa-048与ui-048。Phase14.5继续 `in_progress`。
