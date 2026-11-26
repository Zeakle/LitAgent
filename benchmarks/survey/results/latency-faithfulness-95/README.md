# Survey Benchmark

基准包含 3 次工作台实测和 18 次受控回放。表中另列真实复跑、摘要续跑及诊断复验。
点击每行的「查看过程」可打开只读执行工作台，查看阶段总览、任务图、调用时间轴和阶段耗时。

执行 21/21 · 原始交付 11/21 · 同批归档离线复核 12/21 · 计入真实复跑交付 12/21 · 修复复跑 1 次 · 计入摘要阶段续跑交付 16/21 · 摘要阶段续跑 6 次 · 诊断复验 1 次 · Faithfulness ≥ 0.95

| 运行 | 执行过程 | 严格交付 | Faithfulness | 引用 ID 有效性 | 内部一致性 | 耗时（秒） | Tokens |
| --- | --- | --- | --- | --- | --- | --- | --- |
| few-shot-metric · 工作台 · external-abstract · r1 | [查看过程](https://zeakle.github.io/LitAgent/?run=c4b4fdb92580) | 离线复核通过（原交付因账本绑定被拒） | 96.39% (587/609) | 100.00% | 100.00% | 966.9 | 427,627 |
| vit-foundations · 工作台 · external-abstract · r1 | [查看过程](https://zeakle.github.io/LitAgent/?run=9ef7e1b9c2a5) | 通过 | 98.71% (612/620) | 100.00% | 100.00% | 1,153.8 | 439,234 |
| nerf-foundations · 工作台 · external-abstract · r1 | [查看过程](https://zeakle.github.io/LitAgent/?run=89dc828fd629) | 通过 | 97.02% (553/570) | 100.00% | 100.00% | 1,048.7 | 409,978 |
| few-shot-metric · 受控回放 · external-only · r1 | [查看过程](https://zeakle.github.io/LitAgent/?run=replay-few-shot-metric-external-only-r1) | 未通过（摘要或结论超长） | — | — | — | 553.9 | 215,864 |
| few-shot-metric · 修复复跑 · external-only · r1 | [查看过程](https://zeakle.github.io/LitAgent/?run=replay-few-shot-metric-external-only-r1-retry-6248c89379c1) | 通过 | 99.12% (904/912) | 100.00% | 100.00% | 1,555.6 | 713,346 |
| few-shot-metric · 受控回放 · external-abstract · r1 | [查看过程](https://zeakle.github.io/LitAgent/?run=replay-few-shot-metric-external-abstract-r1) | 未通过（摘要或结论超长） | — | — | — | 510.9 | 219,749 |
| few-shot-metric · 摘要阶段续跑 · external-abstract · r1 | [查看过程](https://zeakle.github.io/LitAgent/?run=replay-few-shot-metric-external-abstract-r1-retry-784e96a6da72) | 通过 | 99.80% (510/511) | 100.00% | 100.00% | 875.1 | 432,083 |
| few-shot-metric · 受控回放 · external-fulltext · r1 | [查看过程](https://zeakle.github.io/LitAgent/?run=replay-few-shot-metric-external-fulltext-r1) | 未通过（摘要或结论超长） | — | — | — | 557.4 | 236,290 |
| few-shot-metric · 摘要阶段续跑 · external-fulltext · r1 | [查看过程](https://zeakle.github.io/LitAgent/?run=replay-few-shot-metric-external-fulltext-r1-retry-0945d57d18ce) | 通过 | 99.62% (531/533) | 100.00% | 100.00% | 831.2 | 420,254 |
| vit-foundations · 受控回放 · external-abstract · r1 | [查看过程](https://zeakle.github.io/LitAgent/?run=replay-vit-foundations-external-abstract-r1) | 未通过（摘要或结论超长） | — | — | — | 557.1 | 238,614 |
| vit-foundations · 摘要阶段续跑 · external-abstract · r1 | [查看过程](https://zeakle.github.io/LitAgent/?run=replay-vit-foundations-external-abstract-r1-retry-d29bfeea69d9) | 通过 | 100.00% (587/587) | 100.00% | 100.00% | 414.0 | 241,229 |
| vit-foundations · 受控回放 · external-fulltext · r1 | [查看过程](https://zeakle.github.io/LitAgent/?run=replay-vit-foundations-external-fulltext-r1) | 通过 | 98.43% (815/828) | 100.00% | 100.00% | 1,123.4 | 511,015 |
| vit-foundations · 受控回放 · external-only · r1 | [查看过程](https://zeakle.github.io/LitAgent/?run=replay-vit-foundations-external-only-r1) | 未通过（方舟账户拒绝请求） | — | — | — | 78.2 | 42,270（部分） |
| nerf-foundations · 受控回放 · external-fulltext · r1 | [查看过程](https://zeakle.github.io/LitAgent/?run=replay-nerf-foundations-external-fulltext-r1) | 通过 | 95.26% (523/549) | 100.00% | 100.00% | 988.4 | 438,143 |
| nerf-foundations · 受控回放 · external-only · r1 | [查看过程](https://zeakle.github.io/LitAgent/?run=replay-nerf-foundations-external-only-r1) | 通过 | 98.95% (566/572) | 100.00% | 100.00% | 1,276.4 | 559,606 |
| nerf-foundations · 受控回放 · external-abstract · r1 | [查看过程](https://zeakle.github.io/LitAgent/?run=replay-nerf-foundations-external-abstract-r1) | 通过 | 99.59% (485/487) | 100.00% | 100.00% | 881.2 | 409,455 |
| few-shot-metric · 受控回放 · external-fulltext · r2 | [查看过程](https://zeakle.github.io/LitAgent/?run=replay-few-shot-metric-external-fulltext-r2) | 通过 | 99.15% (583/588) | 100.00% | 100.00% | 951.3 | 445,497 |
| few-shot-metric · 受控回放 · external-abstract · r2 | [查看过程](https://zeakle.github.io/LitAgent/?run=replay-few-shot-metric-external-abstract-r2) | 未通过（正文压缩来源归属错误） | — | — | — | 782.7 | 303,409 |
| few-shot-metric · 受控回放 · external-only · r2 | [查看过程](https://zeakle.github.io/LitAgent/?run=replay-few-shot-metric-external-only-r2) | 未通过（摘要或结论超长） | — | — | — | 543.3 | 228,736 |
| few-shot-metric · 摘要阶段续跑 · external-only · r2 | [查看过程](https://zeakle.github.io/LitAgent/?run=replay-few-shot-metric-external-only-r2-retry-11c205c4ab91) | 执行失败（摘要或结论超长） | — | — | — | — | 78,619 |
| few-shot-metric · 摘要阶段续跑 · external-only · r2 | [查看过程](https://zeakle.github.io/LitAgent/?run=replay-few-shot-metric-external-only-r2-retry-70a2fbdbbf80) | 未通过（评估未通过） | 94.14% (626/665) | 100.00% | 100.00% | 2,745.6 | 1,346,630 |
| vit-foundations · 受控回放 · external-only · r2 | [查看过程](https://zeakle.github.io/LitAgent/?run=replay-vit-foundations-external-only-r2) | 通过 | 98.90% (538/544) | 100.00% | 100.00% | 1,485.5 | 628,681 |
| vit-foundations · 受控回放 · external-fulltext · r2 | [查看过程](https://zeakle.github.io/LitAgent/?run=replay-vit-foundations-external-fulltext-r2) | 未通过（摘要或结论超长） | — | — | — | 594.6 | 259,002 |
| vit-foundations · 摘要阶段续跑 · external-fulltext · r2 | [查看过程](https://zeakle.github.io/LitAgent/?run=replay-vit-foundations-external-fulltext-r2-retry-6c840c5a854e) | 通过 | 98.79% (652/660) | 100.00% | 100.00% | 506.9 | 252,939 |
| vit-foundations · 受控回放 · external-abstract · r2 | [查看过程](https://zeakle.github.io/LitAgent/?run=replay-vit-foundations-external-abstract-r2) | 通过 | 100.00% (715/715) | 100.00% | 100.00% | 1,692.2 | 675,239 |
| nerf-foundations · 受控回放 · external-abstract · r2 | [查看过程](https://zeakle.github.io/LitAgent/?run=replay-nerf-foundations-external-abstract-r2) | 通过 | 97.14% (475/489) | 100.00% | 100.00% | 855.7 | 389,892 |
| nerf-foundations · 受控回放 · external-only · r2 | [查看过程](https://zeakle.github.io/LitAgent/?run=replay-nerf-foundations-external-only-r2) | 未通过（方舟账户拒绝请求） | — | — | — | 483.4 | 199,758（部分） |
| nerf-foundations · 受控回放 · external-fulltext · r2 | [查看过程](https://zeakle.github.io/LitAgent/?run=replay-nerf-foundations-external-fulltext-r2) | 通过 | 100.00% (493/493) | 100.00% | 100.00% | 1,436.4 | 686,989 |
| few-shot-metric · 诊断复验 · external-abstract · r1 | [查看过程](https://zeakle.github.io/LitAgent/?run=decdecd2e3ec) | 未通过（文献抽取超时） | — | — | — | 340.4 | 97,858（部分） |

工作台耗时取提交至终稿可读，终稿缺失时取提交至终态。受控回放耗时取外部输入就绪至 Survey 完成。Tokens 为供应商已报告用量，部分回执缺失时单独标记。— 表示缺测，Judge 未运行。摘要阶段续跑耗时取摘要修订开始至 Survey 完成，单独展示。

<details>
<summary>延迟汇总与同题配对差值（秒）</summary>

| 配置 | 原始交付 | 有效样本 | p50 | 最大值 | p95 | 同题差值 |
| --- | --- | --- | --- | --- | --- | --- |
| 受控回放 · external-abstract | 未通过 | 3 | 557.1 | 782.7 | — | — |
| 受控回放 · external-abstract | 通过 | 3 | 881.2 | 1,692.2 | — | — |
| 受控回放 · external-fulltext | 未通过 | 2 | 576.0 | 594.6 | — | — |
| 受控回放 · external-fulltext | 通过 | 4 | 1,055.9 | 1,436.4 | — | — |
| 受控回放 · external-only | 未通过 | 4 | 513.3 | 553.9 | — | — |
| 受控回放 · external-only | 通过 | 2 | 1,380.9 | 1,485.5 | — | — |
| 工作台 · external-abstract | 未通过 | 1 | 966.9 | 966.9 | — | — |
| 工作台 · external-abstract | 通过 | 2 | 1,101.2 | 1,153.8 | — | — |
| 同题配对 · external-fulltext − external-abstract | 未通过 → 未通过 | 1 | 46.6 | 46.6 | — | few-shot-metric/r1 46.6 |
| 同题配对 · external-fulltext − external-abstract | 未通过 → 通过 | 2 | 367.4 | 566.2 | — | vit-foundations/r1 566.2、few-shot-metric/r2 168.6 |
| 同题配对 · external-fulltext − external-abstract | 通过 → 未通过 | 1 | -1,097.6 | -1,097.6 | — | vit-foundations/r2 -1,097.6 |
| 同题配对 · external-fulltext − external-abstract | 通过 → 通过 | 2 | 344.0 | 580.8 | — | nerf-foundations/r1 107.2、nerf-foundations/r2 580.8 |
| 同题配对 · external-abstract − external-only | 未通过 → 未通过 | 3 | 239.5 | 479.0 | — | few-shot-metric/r1 -43.0、vit-foundations/r1 479.0、few-shot-metric/r2 239.5 |
| 同题配对 · external-abstract − external-only | 未通过 → 通过 | 1 | 372.3 | 372.3 | — | nerf-foundations/r2 372.3 |
| 同题配对 · external-abstract − external-only | 通过 → 通过 | 2 | -94.2 | 206.7 | — | nerf-foundations/r1 -395.2、vit-foundations/r2 206.7 |
| 同题配对 · external-fulltext − external-only | 未通过 → 未通过 | 1 | 3.6 | 3.6 | — | few-shot-metric/r1 3.6 |
| 同题配对 · external-fulltext − external-only | 未通过 → 通过 | 3 | 953.1 | 1,045.2 | — | vit-foundations/r1 1,045.2、few-shot-metric/r2 408.1、nerf-foundations/r2 953.1 |
| 同题配对 · external-fulltext − external-only | 通过 → 未通过 | 1 | -890.9 | -890.9 | — | vit-foundations/r2 -890.9 |
| 同题配对 · external-fulltext − external-only | 通过 → 通过 | 1 | -288.0 | -288.0 | — | nerf-foundations/r1 -288.0 |

</details>
