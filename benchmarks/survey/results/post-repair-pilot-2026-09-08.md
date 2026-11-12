# Phase 14.5.7 Post-repair Live Pilot

On 2026-09-08, LitAgent ran one real `nerf-foundations / external-fulltext`
Survey after the Phase 14.5.7 repairs. This sanitized summary excludes report
text, prompts, paper excerpts, credentials, trace identifiers, and local paths.

| Metric | Result |
| --- | ---: |
| DAG | 14 tasks completed; `partial=false` |
| Required-paper flow | 1.0 at recall, dedup, relevance, evidence, context, and final citation |
| Initial runtime faithfulness | 0.7636 |
| Evidence rewrite | committed after validation and re-evaluation |
| Final runtime faithfulness | 0.8958; 48 claims, 5 unsupported |
| Citation accuracy / internal consistency | 1.0 / 1.0 |
| Quality / delivery | `passed` / `ready`; publishable |
| Judge | 0.875; 24 claims, 3 unsupported; topic coverage 1.0 |
| Product / Judge tokens | 127,511 / 14,816 |
| Tool retries / final tool or worker failures | 5 / 0 |
| Trace completeness | 64/64 generations had input, output, and usage |

The generation and Judge model were both `doubao-seed-2-0-mini-260428`, so this
run is `self_judged=true` and `formal_eligible=false`. It used a dirty worktree
and only one case/profile. It validates the repaired execution, scoring,
transactional rewrite, delivery, artifact, and trace path; it does not establish
a formal benchmark, a selected-fulltext gain, or model generalization.
