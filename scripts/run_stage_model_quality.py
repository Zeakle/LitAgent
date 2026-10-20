"""Run a frozen Lite/Pro quality campaign without changing historical Mini trials."""

from __future__ import annotations

import argparse
import asyncio
import hashlib
import json
import sys
from importlib.metadata import version
from pathlib import Path

from litagent.benchmark.datasets import load_survey_dataset, load_survey_profiles
from litagent.benchmark.quality_suite import (
    PILOT_EVALUATION_GATE_CONTRACT,
    PILOT_IDS,
    QualitySuiteRunner,
)
from litagent.config import load_config
from litagent.contracts import build_config_summary
from litagent.observability.recorder import ArchiveRepository
from litagent.report_audit import atomic_json, digest
from litagent.workbench import RunOptions, VerificationOptions

VERSION = "stage-model-quality-v2"
MODELS = {
    "extraction": "doubao-seed-2-1-lite-260915",
    "generation": "doubao-seed-2-1-pro-260915",
    "evaluation": "doubao-seed-2-1-lite-260915",
}
ORIGIN = "https://ark.cn-beijing.volces.com/api/v3"
DATASET = Path("benchmarks/survey/dataset.yaml")
PROFILES = Path("benchmarks/survey/profiles.yaml")


def sha(path: Path) -> str:
    """Hash immutable experiment inputs without publishing their contents."""
    return hashlib.sha256(path.read_bytes()).hexdigest()


def options(*, generation_strategy="single_v1") -> RunOptions:
    """Keep optional paid verification disabled and complete Evaluation mandatory."""
    return RunOptions(
        quality_policy="strict_auto_v2",
        report_language="zh-CN",
        review_structure="academic_v2",
        body_quality="analysis_v1",
        generation_strategy=generation_strategy,
        verification=VerificationOptions(judge=False, jev=False),
    )


def check_models(config):
    """Reject drift in provider, model roles, or the fixed output reservation."""
    if (
        config.llm.base_url.rstrip("/") != ORIGIN
        or {role: config.llm.model_for(role) for role in MODELS} != MODELS
        or config.llm.model != MODELS["extraction"]
        or config.llm.max_tokens != 20000
        or config.safety.max_cost_tokens != 500000
        or config.planner.thinking_mode != "disabled"
        or any(
            value != "disabled"
            for value in (
                config.extractor.thinking_mode,
                config.eval.thinking_mode,
                config.adversarial.thinking_mode,
            )
        )
    ):
        raise ValueError("stage_quality_model_configuration_invalid")


def history_summary(path):
    """Carry known usage and unknown holds separately from the new campaign cap."""
    summary = json.loads(path.read_text())["cumulative"]
    integer_fields = (
        "initial_generation_rounds",
        "model_requests",
        "model_known_tokens",
        "model_unknown_conservative_tokens",
        "jev_requests",
        "jev_conservative_input_tokens",
        "judge_requests",
    )
    if any(type(summary.get(k)) is not int or summary[k] < 0 for k in integer_fields):
        raise ValueError("stage_quality_history_invalid")
    if type(summary.get("jev_usage_complete")) is not bool:
        raise ValueError("stage_quality_history_invalid")
    return {k: summary[k] for k in (*integer_fields, "jev_usage_complete")}


def prepare(root: Path, config, history_path: Path) -> dict:
    """Freeze code, public questions, existing corpus bytes, and prior accounting."""
    check_models(config)
    historical = history_summary(history_path)
    dataset = load_survey_dataset(DATASET)
    profile = next(
        p for p in load_survey_profiles(PROFILES) if p.profile_id == "external-fulltext"
    )
    paths = set(Path("src/litagent").rglob("*.py"))
    paths.update(
        [Path(__file__), DATASET, PROFILES, history_path, Path("config/default.yaml")]
    )
    paths.update(Path(p) for p in dataset.manifest_paths)
    # Existing corpus bytes are fixed; newly fetched public PDFs are recorded by ingestion.
    paths.update(p for p in Path(config.rag.raw_root).rglob("*") if p.is_file())
    manifest = {
        "version": VERSION,
        "suite_id": root.name,
        "config": build_config_summary(config, "strict_auto_v2"),
        "options": options(generation_strategy="body_summary_v1").model_dump(
            mode="json"
        ),
        "models": MODELS,
        "historical_accounting": historical,
        "hashes": {str(p.resolve()): sha(p) for p in sorted(paths)},
        "runtime": {
            "python": sys.version,
            "packages": {
                p: version(p) for p in ("openai", "httpx", "pydantic", "langgraph")
            },
        },
        "dataset_fingerprint": dataset.fingerprint,
        "profile_fingerprint": profile.fingerprint,
        "pilots": list(PILOT_IDS),
        "formal_cases": [c.case_id for c in dataset.cases],
        "formal_gate": PILOT_EVALUATION_GATE_CONTRACT,
        "limits": {
            "rounds": 18,
            "survey_tokens": 9000000,
            "survey_requests": 1152,
            "per_round_tokens": 500000,
            "per_round_requests": 64,
            "judge_requests": 0,
            "jev_requests": 0,
            "retries": 0,
        },
    }
    # Never replace a prepared or paid identity, including a partially started directory.
    root.mkdir(parents=True, exist_ok=False)
    snapshots = {}
    for path in sorted(paths):
        if path.suffix not in {".py", ".yaml", ".toml"}:
            continue
        try:
            relative = path.resolve().relative_to(Path.cwd())
        except ValueError:
            continue
        target = root / "frozen-files" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(path.read_bytes())
        if sha(target) != manifest["hashes"][str(path.resolve())]:
            raise ValueError("stage_quality_prepare_input_changed")
        snapshots[str(target.resolve())] = sha(target)
    manifest["snapshots"] = snapshots
    manifest["binding"] = digest(manifest)
    atomic_json(root / "manifest.json", manifest)
    return manifest


def validate_manifest(root: Path, config) -> dict:
    """Reject changed experiment inputs before any transport is constructed."""
    check_models(config)
    m = json.loads((root / "manifest.json").read_text())
    if (
        m.get("limits")
        != {
            "rounds": 18,
            "survey_tokens": 9000000,
            "survey_requests": 1152,
            "per_round_tokens": 500000,
            "per_round_requests": 64,
            "judge_requests": 0,
            "jev_requests": 0,
            "retries": 0,
        }
        or m.get("version") != VERSION
        or m.get("suite_id") != root.name
        or digest({k: v for k, v in m.items() if k != "binding"}) != m.get("binding")
        or m.get("config") != build_config_summary(config, "strict_auto_v2")
        or m.get("options")
        != options(
            generation_strategy=m.get("options", {}).get(
                "generation_strategy", "single_v1"
            )
        ).model_dump(mode="json")
        or m.get("models") != MODELS
        or m.get("formal_gate") != PILOT_EVALUATION_GATE_CONTRACT
        or m.get("pilots") != list(PILOT_IDS)
    ):
        raise ValueError("stage_quality_manifest_invalid")
    for path, expected in {**m["hashes"], **m["snapshots"]}.items():
        if not Path(path).is_file() or sha(Path(path)) != expected:
            raise ValueError("stage_quality_frozen_input_changed")
    actual_runtime = {
        "python": sys.version,
        "packages": {p: version(p) for p in m["runtime"]["packages"]},
    }
    if actual_runtime != m["runtime"]:
        raise ValueError("stage_quality_runtime_changed")
    return m


class StageModelQualityRunner(QualitySuiteRunner):
    """Reuse the real pipeline with an explicit frozen identity and stricter pilot gate."""

    def __init__(self, campaign: Path, config, *, executor=None, output_root=None):
        """Bind all new observations to one immutable 18-slot campaign."""
        self.campaign = campaign
        self.manifest = validate_manifest(campaign, config)
        dataset = load_survey_dataset(DATASET)
        profile = next(
            p
            for p in load_survey_profiles(PROFILES)
            if p.profile_id == "external-fulltext"
        )
        if (
            dataset.fingerprint != self.manifest["dataset_fingerprint"]
            or profile.fingerprint != self.manifest["profile_fingerprint"]
        ):
            raise ValueError("stage_quality_dataset_changed")
        super().__init__(
            config,
            dataset,
            profile,
            root=output_root or Path("artifacts/benchmarks/survey"),
            run_archive=ArchiveRepository(),
            suite_id=self.manifest["suite_id"],
            run_options=RunOptions.model_validate(self.manifest["options"]),
            executor=executor,
        )
        self.binding = digest([self.binding, self.manifest["binding"]])

    def initial(self):
        """Publish only model identities and accounting, never private frozen paths."""
        state = super().initial()
        state["self_judged"] = False
        state["execution_context"] = {
            k: self.manifest[k]
            for k in (
                "version",
                "binding",
                "models",
                "historical_accounting",
                "formal_gate",
            )
        }
        state["execution_context"]["verification"] = options().verification.model_dump(
            mode="json"
        )
        return state

    def _formal_gate_failure(self, state):
        """Admit a fixed matrix after explainable pilots, retaining science failures."""
        pilots = [o for o in state["observations"] if o["phase"] == "pilot"]
        if len(pilots) != 3:
            return "stage_quality_pilot_contract_incomplete"
        for item in pilots:
            acceptance = item.get("acceptance")
            if (
                item.get("status") not in {"delivered", "blocked"}
                or item.get("technical_failure")
                or type(item.get("complete_verification")) is not bool
                or not isinstance(acceptance, dict)
                or type(acceptance.get("publishable")) is not bool
            ):
                return "stage_quality_pilot_contract_incomplete"
            if (
                (item["status"] == "delivered") != acceptance["publishable"]
                or acceptance["publishable"]
                and not item["complete_verification"]
            ):
                return "stage_quality_pilot_contract_inconsistent"
            if item["status"] == "blocked":
                reasons = item.get("failure_codes") or item.get("diagnostic_failures")
                if not isinstance(reasons, list) or not any(
                    isinstance(r, str) and r.strip() for r in reasons
                ):
                    return "stage_quality_pilot_contract_incomplete"
        if not any(item["complete_verification"] for item in pilots):
            return "stage_quality_no_complete_evaluation"
        # Suite dispatch is independent of the stricter report delivery decision.
        state["formal_eligible"] = True
        return None

    async def _live(self, case, config, run_id):
        """Check immutable sources again between paid rounds, using the real runner."""
        validate_manifest(self.campaign, self.config)
        return await super()._live(case, config, run_id)


async def execute(args):
    """Prepare offline, then explicitly run or inspect the bounded campaign."""
    config = load_config()
    if args.prepare:
        prepare(args.root, config, args.history)
    runner = StageModelQualityRunner(args.root, config)
    result = await runner.run(
        dry_run=not args.live,
        max_new_observations=args.max_new_observations,
    )
    print(
        json.dumps(
            {
                k: result.get(k)
                for k in (
                    "run_id",
                    "binding",
                    "status",
                    "counts",
                    "budgets",
                    "stop_reasons",
                )
            },
            ensure_ascii=False,
        )
    )


def main():
    """Keep live execution distinct from immutable preparation."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, required=True)
    parser.add_argument(
        "--history",
        type=Path,
        default=Path(
            "artifacts/acceptance/stage-model-routing-20261017/acceptance-summary.json"
        ),
    )
    parser.add_argument("--prepare", action="store_true")
    parser.add_argument("--live", action="store_true")
    parser.add_argument("--max-new-observations", type=int, choices=range(1, 19))
    asyncio.run(execute(parser.parse_args()))


if __name__ == "__main__":
    main()
