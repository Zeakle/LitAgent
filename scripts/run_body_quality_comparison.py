"""Run six frozen writing comparisons and conditionally a separate formal suite."""

from __future__ import annotations

import argparse
import asyncio
import copy
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path

from litagent.benchmark.datasets import load_survey_dataset, load_survey_profiles
from litagent.benchmark.quality_suite import (
    QualitySuiteResult,
    QualitySuiteRunner,
    export_bundle,
    suite_lock,
    summarize,
)
from litagent.config import load_config
from litagent.context.evidence_selector import EvidenceSelection
from litagent.contracts import build_config_summary
from litagent.observability.recorder import ArchiveRepository
from litagent.quality import build_fact_cards, validate_fact_snapshot
from litagent.report_audit import atomic_json, digest
from litagent.workbench import RunOptions, VerificationOptions
from scripts.run_frozen_generation import (
    MODEL,
    ORIGIN,
    file_sha,
    matrix_stop_reason,
    run_trial,
)

BENCHMARK_ROOT = Path("artifacts/benchmarks/survey")
VERSION = "body-quality-comparison-v1"
CASE_IDS = {
    "vit": "vit-data-efficient",
    "few-shot": "few-shot-metric",
    "nerf": "nerf-foundations",
}
ZERO = {
    s: {"charged": 0, "requests": 0, "usage_complete": True}
    for s in ("survey", "judge", "jev")
}


def prepare(
    root: Path,
    config,
    source_root: Path,
    prior_comparison: Path | None = None,
    *,
    formal_enabled: bool = True,
):
    """Freeze source artifacts, complete evidence and both writer identities before calls."""
    if type(formal_enabled) is not bool:
        raise ValueError("body_comparison_formal_configuration_invalid")
    root.mkdir(parents=True, exist_ok=True)
    checkpoint = root / "comparison.json"
    if checkpoint.exists() and json.loads(checkpoint.read_text()).get("observations"):
        raise ValueError("body_comparison_already_started")
    if config.llm.base_url.rstrip("/") != ORIGIN or config.llm.model != MODEL:
        raise ValueError("body_comparison_model_invalid")
    history_path = source_root / "vit" / "checkpoint.json"
    history = json.loads(history_path.read_text())
    historical = {
        "rounds": history["total_rounds"],
        "services": history["cumulative_after"],
        "usage_interpretation": "includes_unknown_request_reservations",
    }
    prior_hashes = {}
    if prior_comparison is not None:
        prior_manifest_path = prior_comparison / "manifest.json"
        prior_state_path = prior_comparison / "comparison.json"
        previous = json.loads(prior_manifest_path.read_text())
        state = json.loads(prior_state_path.read_text())
        if (
            previous.get("version") != VERSION
            or digest({k: v for k, v in previous.items() if k != "binding"})
            != previous.get("binding")
            or state.get("binding") != previous["binding"]
            or state.get("status") not in {"stopped", "completed"}
            or state.get("pending")
            or state.get("pending_formal")
            or state.get("formal_services")
            or previous.get("hashes", {}).get(str(history_path.resolve()))
            != file_sha(history_path)
        ):
            raise ValueError("body_comparison_prior_not_terminal_or_compatible")
        observations = state.get("observations", [])
        allowed = {c["run_id"] for c in previous["cases"]}
        ids = [o["run_id"] for o in observations]
        if len(ids) != len(set(ids)) or not set(ids).issubset(allowed):
            raise ValueError("body_comparison_prior_observations_invalid")
        services = {}
        for service in ZERO:
            before = previous["historical"]["services"][service]
            current = state["budgets"][service]
            if any(
                type(row.get(key)) is not int or row[key] < 0
                for row in (before, current)
                for key in ("charged", "requests")
            ) or any(
                type(row.get("usage_complete")) is not bool for row in (before, current)
            ):
                raise ValueError("body_comparison_prior_budget_invalid")
            known_tokens = sum(
                (o.get("costs", {}).get(service, {}).get("tokens") or 0)
                for o in observations
            )
            known_requests = sum(
                (o.get("costs", {}).get(service, {}).get("requests") or 0)
                for o in observations
            )
            if (
                current["charged"] < known_tokens
                or current["requests"] < known_requests
            ):
                raise ValueError("body_comparison_prior_budget_invalid")
            services[service] = {
                "charged": before["charged"] + current["charged"],
                "requests": before["requests"] + current["requests"],
                "usage_complete": before["usage_complete"]
                and current["usage_complete"],
            }
        historical = {
            "rounds": previous["historical"]["rounds"] + len(observations),
            "services": services,
            "usage_interpretation": "includes_unknown_request_reservations",
            "prior_comparison": {
                "suite_id": previous["suite_id"],
                "binding": previous["binding"],
                "checkpoint_sha256": file_sha(prior_state_path),
            },
        }
        prior_hashes = {
            str(p.resolve()): file_sha(p)
            for p in (prior_manifest_path, prior_state_path)
        }
    hashes = {
        str(p.resolve()): file_sha(p)
        for p in sorted(Path("src/litagent").rglob("*.py"))
    }
    hashes.update(prior_hashes)
    for p in (
        Path(__file__),
        Path("scripts/run_frozen_generation.py"),
        Path("benchmarks/survey/dataset.yaml"),
        Path("benchmarks/survey/profiles.yaml"),
        history_path,
    ):
        hashes[str(p.resolve())] = file_sha(p)
    cases = []
    for domain, cid in CASE_IDS.items():
        input_path = source_root / domain / "input-snapshot.json"
        inputs = json.loads(input_path.read_text())
        selected = EvidenceSelection.from_dict(inputs["evidence_selection"])
        validate_fact_snapshot(
            inputs["fact_cards"], selected.selected_items, inputs["fact_cards_sha256"]
        )
        if build_fact_cards(selected.selected_items) != inputs["fact_cards"]:
            raise ValueError("body_comparison_source_changed")
        source_path = Path(inputs["source_artifact"])
        if file_sha(source_path) != inputs["source_artifact_sha256"]:
            raise ValueError("body_comparison_archive_changed")
        hashes[str(input_path.resolve())] = file_sha(input_path)
        hashes[str(source_path.resolve())] = file_sha(source_path)
        for variant in ("baseline", "candidate"):
            slot = root / f"{domain}-{variant}"
            slot.mkdir(exist_ok=True)
            target = slot / "input-snapshot.json"
            target.write_bytes(input_path.read_bytes())
            hashes[str(target.resolve())] = file_sha(target)
            cases.append(
                {
                    "domain": domain,
                    "case_id": cid,
                    "variant": variant,
                    "root": str(slot.resolve()),
                    "run_id": f"{root.name}-{domain}-{variant}",
                }
            )
    manifest = {
        "version": VERSION,
        "suite_id": root.name,
        "cases": cases,
        "historical": historical,
        "hashes": hashes,
        "config": build_config_summary(config, "strict_auto_v2"),
        "runtime": {
            "python": sys.version,
            "packages": {
                name: version(name)
                for name in ("openai", "httpx", "pydantic", "langgraph")
            },
        },
        "limits": {"survey_tokens": 3000000, "survey_requests": 384, "rounds": 6},
        "verification": VerificationOptions().model_dump(mode="json"),
        "evaluation_body_quality": "analysis_v1",
        "formal_enabled": formal_enabled,
    }
    manifest["binding"] = digest(manifest)
    atomic_json(root / "manifest.json", manifest)
    return manifest


def validate_manifest(root, manifest, config):
    """Reject changed code, inputs or accounting without sending a request."""
    if manifest.get("version") != VERSION or digest(
        {k: v for k, v in manifest.items() if k != "binding"}
    ) != manifest.get("binding"):
        raise ValueError("body_comparison_manifest_invalid")
    if (
        len(manifest["cases"]) != 6
        or len({r["run_id"] for r in manifest["cases"]}) != 6
    ):
        raise ValueError("body_comparison_matrix_invalid")
    if manifest["config"] != build_config_summary(config, "strict_auto_v2"):
        raise ValueError("body_comparison_config_changed")
    for path, sha in manifest["hashes"].items():
        if file_sha(Path(path)) != sha:
            raise ValueError("body_comparison_identity_changed")
    if manifest["verification"] != VerificationOptions().model_dump(mode="json"):
        raise ValueError("body_comparison_review_configuration_invalid")
    if type(manifest.get("formal_enabled", True)) is not bool:
        raise ValueError("body_comparison_formal_configuration_invalid")


def observation(case, state):
    """Keep failed science, missing evaluation and disabled services explicit."""
    report = state.get("report") or {}
    metadata = report.get("metadata") or {}
    acceptance = metadata.get("automatic_acceptance") or {}
    details = report.get("evaluation", {}).get("faithfulness", {}).get("details", {})
    return {
        "phase": "pilot",
        "case_id": f"{case['case_id']}-{case['variant']}",
        "domain": "few_shot" if case["domain"] == "few-shot" else case["domain"],
        "run_id": case["run_id"],
        "status": "delivered" if acceptance.get("publishable") else "blocked",
        "acceptance": acceptance,
        "gates": acceptance.get(
            "gates",
            {
                "evaluation": "not_run",
                "judge": "disabled",
                "extraction_jev": "disabled",
            },
        ),
        "complete_verification": details.get("coverage", {}).get("complete") is True
        and details.get("stages", {}).get("global_check") == "completed",
        "counts": {
            k: details.get(k)
            for k in ("supported_count", "unsupported_count", "unverified_count")
        },
        "costs": state.get("costs", {}),
        "failure_codes": (
            state.get("reason", [])
            if isinstance(state.get("reason"), list)
            else [state.get("reason")]
        ),
        "report_sha256": state.get("report_sha256"),
        "artifact_path": str(Path("artifacts/runs") / f"{case['run_id']}.json"),
        "generation_variant": case["variant"],
        "body_diagnostics": metadata.get("body_diagnostics", {}),
        "analysis_check": details.get("analysis_check", {}),
    }


def presentation(root, manifest, state, formal=None):
    """Use the existing archive and four exporters with the actual planned denominator."""
    dataset = load_survey_dataset(Path("benchmarks/survey/dataset.yaml"))
    planned = [
        {
            "phase": "pilot",
            "case_id": f"{c['case_id']}-{c['variant']}",
            "domain": "few_shot" if c["domain"] == "few-shot" else c["domain"],
        }
        for c in manifest["cases"]
    ]
    planned += [
        {"phase": "formal", "case_id": c.case_id, "domain": c.domain}
        for c in dataset.cases
    ]
    observations = list(state["observations"])
    if state.get("pending"):
        pending = state["pending"]
        case = next(c for c in manifest["cases"] if c["run_id"] == pending["run_id"])
        observations.append(
            {
                "phase": "pilot",
                "case_id": f"{case['case_id']}-{case['variant']}",
                "domain": (
                    "few_shot" if case["domain"] == "few-shot" else case["domain"]
                ),
                "run_id": case["run_id"],
                "status": "started_no_replay",
                "costs": {
                    "survey": {"tokens": None, "requests": 0, "usage_complete": False},
                    "judge": {"tokens": 0, "requests": 0, "usage_complete": True},
                    "jev": {"tokens": 0, "requests": 0, "usage_complete": True},
                },
                "gates": {
                    "evaluation": "not_run",
                    "judge": "disabled",
                    "extraction_jev": "disabled",
                },
            }
        )
    result = QualitySuiteResult(
        run_id=root.name,
        status=state["status"],
        dataset_id=dataset.dataset_id,
        dataset_version=dataset.dataset_version,
        dataset_fingerprint=dataset.fingerprint,
        judgment_status=dataset.judgment_status,
        annotation_source=dataset.judgment_status,
        binding=manifest["binding"],
        planned=planned,
        observations=observations + (formal or {}).get("observations", []),
        budgets={
            **state["budgets"],
            "historical_before": manifest["historical"],
            "formal_services": (formal or {}).get(
                "budgets", state.get("formal_services", {})
            ),
        },
        stop_reasons=state.get("stop_reasons", []),
        formal_eligible=state.get("formal_eligible", False),
    ).model_dump(mode="json")
    result["counts"] = summarize(result)
    path = BENCHMARK_ROOT / f"{root.name}.json"
    atomic_json(path, result)
    export_bundle(path.parent / "presentation" / root.name, result)
    return result


class FormalBodyRunner(QualitySuiteRunner):
    """Run exactly fifteen formal cases after a validated three-domain candidate pilot."""

    def __init__(self, *args, pilot_manifest, pilot_state, **kwargs):
        """Require three passed candidate observations before formal execution."""
        candidate = [
            r
            for r in pilot_state["observations"]
            if r.get("generation_variant") == "candidate"
        ]
        if len(candidate) != 3 or not all(
            r.get("acceptance", {}).get("publishable") is True for r in candidate
        ):
            raise ValueError("body_formal_pilot_not_passed")
        super().__init__(*args, **kwargs)
        self.round_limit = 15
        self.binding = digest(
            [
                self.binding,
                pilot_manifest["binding"],
                digest(pilot_state),
                "formal-only-15",
            ]
        )

    def initial(self):
        """Retain only the fifteen formal slots after external pilot validation."""
        state = super().initial()
        state["planned"] = [p for p in state["planned"] if p["phase"] == "formal"]
        state["formal_eligible"] = True
        return state


async def run(root, config, *, live=False, trial=run_trial):
    """Persist starts before transport; an unknown outcome is never automatically replayed."""
    manifest = json.loads((root / "manifest.json").read_text())
    validate_manifest(root, manifest, config)
    if not live:
        return {
            "status": "planned",
            "paid_calls": 0,
            "binding": manifest["binding"],
            "rounds": 6,
            "formal_conditional": 15 if manifest.get("formal_enabled", True) else 0,
        }
    with suite_lock(root / ".comparison.lock"):
        cp = root / "comparison.json"
        state = (
            json.loads(cp.read_text())
            if cp.exists()
            else {
                "binding": manifest["binding"],
                "status": "running",
                "observations": [],
                "budgets": copy.deepcopy(ZERO),
                "stop_reasons": [],
            }
        )
        if state["binding"] != manifest["binding"]:
            raise ValueError("body_comparison_identity_conflict")
        if state["status"] in {"completed", "stopped"}:
            return state
        if state.get("pending") or state.get("pending_formal"):
            state.update(
                status="stopped", stop_reasons=["body_comparison_interrupted_no_replay"]
            )
            atomic_json(cp, state)
            presentation(root, manifest, state)
            return state
        for case in manifest["cases"][len(state["observations"]) :]:
            validate_manifest(root, manifest, config)
            if (
                state["budgets"]["survey"]["charged"] + 500000 > 3000000
                or state["budgets"]["survey"]["requests"] + 64 > 384
            ):
                state.update(
                    status="stopped", stop_reasons=["body_comparison_budget_exceeded"]
                )
                break
            slot = Path(case["root"])
            plan = {
                "run_id": case["run_id"],
                "quality_policy": "strict_auto_v2",
                "config": manifest["config"],
                "runtime": manifest["runtime"],
                "hashes": {
                    **manifest["hashes"],
                    str((root / "manifest.json").resolve()): file_sha(
                        root / "manifest.json"
                    ),
                },
                "verification": manifest["verification"],
                "body_quality": (
                    "analysis_v1" if case["variant"] == "candidate" else "legacy"
                ),
                "evaluation_body_quality": "analysis_v1",
                "limits": {
                    "survey": {"tokens": 500000, "requests": 64},
                    "judge": {"tokens": 0, "requests": 0},
                    "jev": {"tokens": 0, "requests": 0},
                },
                "cumulative_before": copy.deepcopy(state["budgets"]),
                "rounds_before": len(state["observations"]),
                "scope": "frozen public originals; paired writers; common body-analysis evaluation; no gold labels",
            }
            plan["binding"] = digest(plan)
            atomic_json(slot / "plan.json", plan)
            state["pending"] = {
                "run_id": case["run_id"],
                "tokens_reserved": 500000,
                "requests_reserved": 64,
            }
            state["budgets"]["survey"]["charged"] += 500000
            state["budgets"]["survey"]["requests"] += 64
            state["budgets"]["survey"]["usage_complete"] = False
            atomic_json(cp, state)
            presentation(root, manifest, state)
            outcome = await trial(slot, config, plan, live=True)
            state["observations"].append(observation(case, outcome))
            state["budgets"] = outcome["cumulative_after"]
            state.pop("pending", None)
            stop = matrix_stop_reason(outcome)
            semantic = {
                "strict_writing_uncited_science",
                "strict_writing_uncited_scientific_gap",
                "strict_writing_scientific_section_missing",
                "strict_writing_summary_content_duplicate",
            }
            if (
                stop == "strict_worker_failed"
                and str(outcome.get("reason", "")).split(":", 1)[0] in semantic
            ):
                stop = None
            if stop:
                state.update(status="stopped", stop_reasons=[stop])
            atomic_json(cp, state)
            presentation(root, manifest, state)
            if stop:
                break
        else:
            candidate = [
                r
                for r in state["observations"]
                if r["generation_variant"] == "candidate"
            ]
            state["formal_eligible"] = len(candidate) == 3 and all(
                r["acceptance"].get("publishable") is True for r in candidate
            )
            state["status"] = "completed" if state["formal_eligible"] else "stopped"
            state["stop_reasons"] = (
                [] if state["formal_eligible"] else ["candidate_quality_goal_not_met"]
            )
        atomic_json(cp, state)
        presentation(root, manifest, state)
        if state.get("formal_eligible") and not manifest.get("formal_enabled", True):
            state["stop_reasons"] = ["formal_disabled_for_bounded_pilot"]
            atomic_json(cp, state)
            presentation(root, manifest, state)
        if state.get("formal_eligible") and manifest.get("formal_enabled", True):
            dataset = load_survey_dataset(Path("benchmarks/survey/dataset.yaml"))
            profile = next(
                p
                for p in load_survey_profiles(Path("benchmarks/survey/profiles.yaml"))
                if p.profile_id == "external-fulltext"
            )
            runner = FormalBodyRunner(
                config,
                dataset,
                profile,
                root=BENCHMARK_ROOT,
                run_archive=ArchiveRepository(),
                suite_id=f"{root.name}-formal",
                quality_policy="strict_auto_v2",
                run_options=RunOptions(
                    quality_policy="strict_auto_v2",
                    verification=VerificationOptions(),
                    body_quality="analysis_v1",
                ),
                pilot_manifest=manifest,
                pilot_state=state,
            )
            state.update(status="formal_running", pending_formal=True)
            atomic_json(cp, state)
            formal = await runner.run(max_new_observations=15)
            state.pop("pending_formal", None)
            state["formal_services"] = formal.get("budgets", {})
            state["status"] = formal["status"]
            state["stop_reasons"] = formal.get("stop_reasons", [])
            presentation(root, manifest, state, formal)
            atomic_json(cp, state)
        return state


def main():
    """Prepare without calls by default; live requires the complete persisted identity."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--source-root",
        type=Path,
        default=Path("artifacts/acceptance/three-domain-v21-20261015"),
    )
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--prior-comparison", type=Path)
    parser.add_argument("--pilot-only", action="store_true")
    parser.add_argument("--live", action="store_true")
    args = parser.parse_args()
    config = load_config()
    if args.prepare:
        prepare(
            args.root,
            config,
            args.source_root,
            args.prior_comparison,
            formal_enabled=not args.pilot_only,
        )
    result = asyncio.run(run(args.root, config, live=args.live))
    print(
        json.dumps(
            {
                k: result.get(k)
                for k in ("status", "binding", "stop_reasons", "formal_eligible")
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
