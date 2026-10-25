# 同一 schema 的结构提示与提示词身份

014的引言响应省略了`sentences`段落层。HTTP200、合法JSON、原生strict schema回显均不能替代宿主字段检查。本文先记录195的离线修复与历史重放；随后015真实验收结果单列在文末，不能把离线数字算作新增真实用量。

## 改动边界

`native_layout_guidance(schema)`从正在使用的同一份宿主schema生成简短类型速记，明确object、array、property和primitive层级。它不是示例JSON，不包含假正文、假引用、枚举值、schema description或论文内容。出现`sentences`时明确段落对象与句子数组边界；Reviewer仅获得实际的issues结构，摘要使用body_unit_refs，修订保留原outcome/patches结构。

正文、摘要、Reviewer、修订和恢复请求继续共用`request_messages`；预算预检也使用这份消息，不预检短提示后发送长提示。完整科学payload与原生schema不变，新增文字计入原输入与上下文预留。旧坏响应仍拒绝，不补括号、不猜分段、不合并段落、不改科学判断。

业务协议仍为`grounded-writer-v1`，生成策略仍为`body_summary_v2`，恢复契约仍为`intact-unit-reemission-v2`，服务端profile不变。没有增加Agent、配置开关、模型调用、来源或运行预算。

## 版本与历史

提示词版本升级为`grounded-writer-native-intent-v3`。既有`RunOptions.fingerprint`只在`body_summary_v2`分支绑定该版本：旧API默认`single_v1`与显式`body_summary_v1`身份不变；历史`body_summary_v2`的旧prompt-v2 checkpoint不能恢复到新提示下。归档和旧报告继续只读，不能把这项有意的版本失效当作历史内容损坏。

Synthesis保存`generation_prompt_contract`，Reviewer和终稿修订在调用模型前校验，缺失或过期时明确失败。新冻结清单保存`writer_prompt_contract`，重hash也不能掩盖旧值或缺失。JSON/CSV/Markdown/HTML只展示清单中的实际值；历史缺少字段时显示不可用，不能从当前源码反推旧运行版本。展示契约为view-v9，旧公开包不覆盖。

## 离线证据

| 检查 | 结果与边界 |
|---|---|
| 完整默认回归 | 4611通过、25默认排除、8条既有警告，231.68秒 |
| 相关回归 | 268通过；实际SDK→Synthesis→Reviewer→唯一出口→Evaluation正反例仍执行，只替换HTTP |
| 格式与diff | 本批6个Python文件Black/isort与`git diff --check`通过 |
| 科学与权限反例 | 缺段落层、未知引文、错位单元、科学不支持及缺可信方法记录继续阻止交付；没有改变科学预期 |
| 005合法历史响应 | 六份原响应经新提示重放，完整科学payload和原生schema相同，正文SHA仍为`34e8fca38d54b469c098281a4e320798ac9ae6ccf75801991659adaab6002e4f`；不等于科学通过 |
| 新请求预留 | few-shot首请求56795、Reviewer107756；实际SDK请求分别35865/102665 bytes，保留原20000/4096输出，均在当前上下文上限内 |
| 其他两域 | 只预检首请求：ViT63052、NeRF63178；不保证未来整轮均能完成 |
| 014坏响应 | 仅system增加有标识的结构提示，科学payload和native schema逐值相同；原raw SHA相同，仍在恢复追加请求前拒绝，0共享修复 |
| 版本与保护 | 56项历史保护、486份旧源码副本不变；真实014旧身份可重建，新提示明确拒绝恢复且不改旧journal；旧API两种策略fingerprint保持 |
| 展示 | 同源四格式、新prompt版本、历史不可用；桌面/390px、中英、Enter语言切换、Space诊断展开，无页面水平溢出，console无警告或错误；明确标记离线夹具 |

重放合计10次模拟HTTP，新增外部模型请求0。014的11184 tokens是重放的历史用量，不计为新增费用。历史累计保持762请求、4334574已知tokens、1142672保守未知占用；旧欠费记录不退款、不清零，Jev67、Judge0不变。

私有证据位于本机`artifacts/acceptance/native-layout-guidance-20261025/`，不提交原包或论文全文。没有新的真实正文，因此不虚构新增的正文定位或报告导出验收。测试证明接线、兼容、预算与拒绝契约，**不证明模型一定遵守新提示，更不证明科学内容正确**。后续仍需同源码、原三域各一次、原预算的独立真实验收，技术失败停止，不追加到通过；Phase14.5保持`in_progress`。

## 015真实验收补记

[015结果](../benchmarks/survey/results/frozen-writer-three-domain-20261025-015/README.md)使用5212192冻结源码及相同三域完整材料。few-shot五章、摘要和结论完成编译，包含93个正文句子单元和8个派生单元；这不是已核验科学断言数。方法、实验两章各一次非法JSON经独立新响应逐字保真恢复，共用2/2额度。随后Lite Reviewer返回incomplete/length，4096输出tokens全部为reasoning，没有答案；Evaluation未运行，另外两题技术停止，严格交付0/3。

真实新增9请求、149128已知tokens、0新增未知，累计771请求/4483702已知/1142672旧未知。9份原包经实际SDK→Synthesis→Reviewer离线逐字重放，编译SHA一致，仍拒绝不完整Reviewer，未新增外部调用。草稿仅本地保存为未核验预览，不覆盖最终报告出口。LangFuse读回9个generation及同额usage，科学IO为空。下一步离线审查角色间thinking耦合，保持4096上限及科学门禁；本次不能证明提示导致科学质量提升，Phase14.5仍in_progress。
