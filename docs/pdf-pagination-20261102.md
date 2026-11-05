# Phase 14.5 / 292：PDF 正文分页修复

本次只修本地导出，原045三域报告、评分、正文SHA、失败和费用记录不变。没有新模型请求、真实Survey、套件恢复、服务重启或提交。真实完整Evaluation仍0/3，严格交付仍0/3；Phase14.5继续in_progress。291的Evaluation定义/指导输入离线闭合记录见[evaluation-definition-alignment-20261102.md](evaluation-definition-alignment-20261102.md)。

## 问题和最小修复

原ViT PDF第3页有70个可见正文字符落在正文clip外；它们存在于PDF内容流中，页面却看不到。原helper只移动孤立标题，不能保证正文没有裁剪。原HTML自然分页8页完整，第一次移动taxonomy标题变7页且丢2227字符，进一步移动comparison标题回8页后仍少70字符。故不能用坏PDF的文字层当完整性基线，也不能只把标题修好就返回。

修复限于 `src/litagent/report_export.py`：

- 只在HTML布局投影插入唯一host ID，保全部原文字、引用和href。后续重复heading slug局部改为唯一布局ID，保第一个原href目标；不改网页renderer或canonical报告。
- 移除会触发错误的CSS强制分页路径；按平衡root块拆多个自然分页Story。真实绘制越界时精确定位到最深宿主body块，连同leading标题链重排；长段仍自然跨页。每轮增加有限合法分界，无法安全收敛明确拒绝，不返回裁字PDF。
- 全局页号按实际分页次数累计，最后一次建立全部链接，保跨段内部锚点和外链顺序。
- 返回前核全部正文paint边界。字体ascender框、空格advance不等于真实墨迹；可疑fill-text仅在原native font/gid、有效原轮廓及TRM可证明时核真实glyph，字体替代、Type3、伪样式或能力未知保守拒绝。空轮廓用原path证明，不靠字符串、面积或已被clip的像素忽略；非空轮廓变换塌空/非有限不放行。
- 保原A4、48pt边距、字号、行高、独立草稿页脚及安全转义。正文、参考文献、附录与来源绑定不改。

单调保留的分页可能留下较大余白，ViT第4页为可见例子。本轮验收内容完整、无裁剪/重叠和可读性，没有宣称页利用率或版式已经优化。

## 最终源码与离线证据

最终612文件，digest `f6f8336f95b54c6d30c7da4cdb2b8476979470b91d15736d1474b8fe3ebbf81e`；生产文件SHA `7665533282e1424b29e0f51c3e47918d05669b054a9da3379e852730c18e1086`。相对291仅一个生产文件及新增的 `tests/test_pdf_pagination.py`、中性文本fixture `tests/fixtures/pdf-pagination-overflow.json` 变化。

- 默认 `.venv/bin/python -m pytest -v`：actual22829退出0，7543 passed / 25 deselected / 8既有warnings，455.73秒；wrapper22814同为0、源前后不变。
- 两个相关测试文件：actual22808退出0，50 passed；其中新增测试39项，既有导出11项。首轮空格红及mixed开发红保留，未改科学反例或忽略失败。
- 两个本次Python Black/isort及git diff --check：actual23308三个实际退出均0。
- 最终独立review实际22718退出0，同最终生产SHA；真实旧裁剪负例仍红，正常长段、字形四边、空格、ID、跨段链接及异常资源关闭通过。数学极小字号导致MuPDF整paint塌空是库精度边界，不是对任意微小轮廓的正确性保证；本报告实际10pt正文不在该范围。

详细证据均在 `artifacts/acceptance/pdf-pagination-20261102/`。首候选、私有命名/超时失败及首导出流计数错误均留在原attempt，最终结果使用新目录，不覆写旧证据。

## 三域实际导出

通过真实 `LocalReportExporter`，使用原三域归档及已选证据构建同一ReaderReport，先核Markdown与原导出逐字一致，再仅生成新的PDF草稿：

| 题目 | 页数 | 按顺序完整保留的原HTML叶文本块 | 正文越框 |
|---|---:|---:|---:|
| few-shot | 7 | 123 | 0 |
| ViT | 9 | 131 | 0 |
| NeRF | 7 | 124 | 0 |

导出actual23014退出0、4.43秒，三份均原版本/SHA及publishable=false，原六格式文件保持。378个文本块按顺序匹配；只展开明确的PDF字体ligature及布局空白，不做NFKC或科学数值归一化。保存的PDF原body内容流逐SHA保留，额外流仅MuPDF q/Q包装及精确位置/字体的host草稿页脚。没有重新创作Word或文献数据，也不把此前Word验收冒充本次新导出。

最终PDF位于 `exports-final-003/{case_id}/{case_id}-draft-reflow.pdf`；首 `exports-final-002` 仅因私有审计器错猜页脚增加流数量而失败，其文件保留，不作为最终输出。

## 视觉与保护核对

bundled Poppler实际渲染7+9+7共23页，actual23339/23343/23347均退出0；23/23页均已实际view_image(original)，独立审计actual24557退出0且visual_pass=true，两项分列。root另实际查看ViT3/4原尺寸PNG：缺尾段完整移至新页，页脚与正文不重叠；第4页大余白作为非阻断保守分页限制单列。逐页SHA/尺寸/观察见qa-visual-001/visual-review.json，首渲染进程0没有冒充已查看。

root最终实际24997退出0，重新逐SHA核全部36630保护文件、最终612源码及架构Plan SHA `f0b300c90faf60a7cabaa1e633b7c4f4a88e3e375341c28ef36ad9ebd45ff625`，全部保持。独立数据审计actual24995退出0，三新PDF、378来源叶文本、外链顺序及原六格式绑定保持；其历史保护证明明确复用24635在后续本地Node观察器失败前已完成的扫描，原整体exit1不改，不能冒充完整0。root24997此次新鲜全量重核单独退出0。

最终关闭凭证：`artifacts/acceptance/pdf-pagination-20261102/final-closure-002.json`，绑定默认/相关测试、格式、实际导出、23页视觉、独立审核及保护证明；五份plan副本另存。离线工程与视觉已完成，新增付费调用为0。真实完整Evaluation仍0/3、严格交付仍0/3；本次未重做语义验收，Phase14.5保持in_progress，下一步另行冻结真实语义验证。
