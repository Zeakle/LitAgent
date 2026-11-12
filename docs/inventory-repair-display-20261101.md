# 清单恢复展示修复验收（2026-11-01）

步骤274已完成指定标签修复、默认离线回归和实际界面验收；Phase 14.5仍为`in_progress`。本次没有模型调用，最新真实038仍为严格0/3，未完成Evaluation的草稿仍不可交付。

## 修复范围与证据

038 few-shot的`bounded-inventory-recovery-v2`实际只重新抽取断言清单，却被质量页底部误写为“正文定向修复”。现显式识别v1和v2，显示“断言清单重抽取·正文未修改”；共享计数仍为1/2，正文SHA仍为`10e02cba211ccab033350e8c6cb068bdb8cf3dd9a3e098af23988abb5e2ecde7`。真正的正文修复、容量修复和未知purpose保持原分支，不据名称前缀猜类型。

首次实际reload仍加载旧模块，因此按照现有缓存契约同步app、quality和styles引用的版本查询。样式内容及其他模块版本未改。总共变更三个静态文件和一个测试文件，没有修改模型、预算、账本或科学门禁。

最终567文件源码digest为`042a6bda608835f1cfa2031c668c972b7c3c992e89d79932319ea267d85717dd`。新增24项中英、状态、历史兼容及真实SDK消费者测试；相关四组151项通过。默认`pytest -v`实际6322 passed、25外部测试deselected、8既有warnings，实际退出0。12项最大资源验证、汇总及独立检查全部实际退出0；90向量、54 Evaluation预检、6最大路径和既有计划保持，69项聚合修复资源拒绝仍被保留，并非全部可发送。Black、isort和`git diff --check`通过。

## 实际浏览器验收

仅访问已授权的本机8000和8020。实际核对桌面1280、移动390，中英文、键盘关闭与焦点、稳定APA切换后的证据定位、可展开检索附录及038失败Benchmark。12份观察中包含一份仍为IEEE的异步过渡帧，该帧不计APA通过；15张JPEG包括修复前、缓存旧帧和一次后台视口reset未生效帧，均保留原事实。最终当前工作台恢复为默认1280×720、中文、IEEE。

稳定APA后的引用弹窗关闭与焦点恢复已核对；切换未稳定时旧触发器的焦点边界仍保留，不宣称所有异步竞态通过。现有归档仅r1，没有实测同run多revision切换。本次没有生成新报告，未将旧导出或截图计为新的Markdown、PDF、Word导出验收。

14625个规范化保护文件SHA全部不变，包括旧费用、正文、评分、批准及039原始冻结。039仍是已准备未运行，但其全源码绑定已因这次静态修复过期；真实运行保护器已证明会拒绝使用新工作树启动旧039。没有回填manifest、创建040、写启动门禁或重发038未知请求。

## 尚未完成

实际UI另发现运行预算摘要缺口：038 ViT账本已保存已知218506 tokens、未知保守占用78205、14请求及usage不完整，但public view及前端未接入该快照，主摘要只显示已知用量。原始诊断可查到未知请求，仍不能代替专用预算摘要；容量分配时差额也不能充当终态费用或退款。下一步仅补只读白名单投影及展示，不改账本、旧数据、科学结果或预算上限。

账户恢复状态仍待明确；已有有界付费授权保持。本次没有外部服务探测，不能用UI和离线通过替代三域真实科学验收。

证据根为`artifacts/acceptance/support-receipts-ui-20261101`。主要证明：`completion-proof.json`、`default-001-proof.json`、`maximum-001/independent-final-check.json`、`proof/independent-label-002/review.json`、`proof/039-binding-stale-independent.json`、`browser-observations-final.json`及`ui-files-proof.json`。
