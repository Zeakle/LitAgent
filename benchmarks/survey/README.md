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
litagent benchmark survey --stage ablation --dataset benchmarks/survey/dataset.yaml --profiles benchmarks/survey/profiles.yaml --judge-config benchmarks/survey/judge.example.yaml --live
```

After human review, reuse that exact ablation artifact for the remaining nine runs:

```powershell
litagent benchmark survey --stage baseline --dataset benchmarks/survey/dataset.yaml --profiles benchmarks/survey/profiles.yaml --judge-config benchmarks/survey/judge.example.yaml --reuse-ablation artifacts/benchmarks/survey/<run-id>.json --live
```
