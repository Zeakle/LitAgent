# Survey Benchmark

Phase 14.5 evaluates the complete Survey product path rather than retrieval alone. The versioned files in
this directory are safe to review in Git; private reports, RunArtifacts, and traces are written under
`artifacts/benchmarks/survey/` and `artifacts/runs/`.

`dataset.yaml` is `owner_approved_ai_assisted`: Codex performed the source audit and the project owner
accepted its labels on 2026-08-13. This status permits the project baseline while remaining distinct from
`human_reviewed`; every artifact preserves that provenance. The unchecked items in
[`HUMAN_REVIEW_CHECKLIST.md`](HUMAN_REVIEW_CHECKLIST.md) remain available if an independent human
paper-by-paper audit is performed later.

The ablation command performs 30 runs (18 first pass plus 12 repeats for the top two profiles):

```powershell
litagent benchmark survey --stage ablation --dataset benchmarks/survey/dataset.yaml --profiles benchmarks/survey/profiles.yaml --judge-config benchmarks/survey/judge.example.yaml --max-observations 9 --max-product-tokens 2000000 --max-judge-tokens 120000 --live
```

After a comparable, accepted ablation, reuse that exact artifact for the remaining
nine runs:

```powershell
litagent benchmark survey --stage baseline --dataset benchmarks/survey/dataset.yaml --profiles benchmarks/survey/profiles.yaml --judge-config benchmarks/survey/judge.example.yaml --reuse-ablation artifacts/benchmarks/survey/<run-id>.json --live
```

For a cost-controlled, non-formal comparison, the pilot stage accepts an explicit
case/profile matrix. This example compares external search against selected-fulltext
RAG on one case from each domain, for six observations total:

```powershell
litagent benchmark survey --stage pilot --dataset benchmarks/survey/dataset.yaml --profiles benchmarks/survey/profiles.yaml --judge-config artifacts/benchmarks/survey/judge.live.yaml --case-id few-shot-metric --case-id vit-foundations --case-id nerf-foundations --profile-id external-only --profile-id external-fulltext --max-observations 6 --max-product-tokens 2000000 --max-judge-tokens 120000 --live
```

Pilot results never select a formal winner and cannot be reused as a baseline input.
Explicit selectors are rejected by the formal ablation and baseline stages.

## Cost-bounded live pilot

On 2026-08-25, the project ran a reduced matrix with one case per domain and all
three profiles, for nine real Survey observations. This pilot verified checkpoint
recovery, RunArtifact integrity, Judge persistence, separate cost accounting, and
18 readable Survey/Judge LangFuse traces. It did not pass the product-quality
acceptance gate: external inputs differed across profiles, citation recall was zero,
runtime faithfulness evaluation remained unavailable, and one full-text run was
partial.

The versioned, sanitized summary is available in
[`results/pilot-2026-08-25.md`](results/pilot-2026-08-25.md) and
[`results/pilot-2026-08-25.json`](results/pilot-2026-08-25.json). Private reports,
evidence ledgers, trace identifiers, and local artifact paths remain ignored under
`artifacts/`.

The pilot findings have since been repaired in code and covered by the default
offline suite (`896 passed, 25 deselected`). The runner now freezes and replays one
Planner/Search snapshot per case and repetition, reports paper coverage at each
product stage, fails closed on incomplete observations, and enforces all three
explicit cost limits before starting another paid observation. Runtime citation
and faithfulness evaluation and long-context synthesis/review also have bounded,
provider-aware paths.

One `nerf-foundations / external-fulltext` post-repair live pilot then reached
runtime faithfulness 0.8958 and delivery `ready`; see the
[`sanitized result`](results/post-repair-pilot-2026-09-08.md). Because it is one
self-judged dirty-worktree observation, it proves only that the repaired path can
complete. The earlier result remains historical failure evidence, and no formal
profile winner or full-text gain has been established.
