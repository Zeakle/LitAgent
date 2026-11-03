"""Run exactly three frozen-evidence writing cases, with durable no-replay accounting."""

from __future__ import annotations

import argparse
import asyncio
import base64
import copy
import csv
import hashlib
import html
import io
import json
import os
import re
import subprocess
import sys
import time
import types
from importlib.metadata import version
from pathlib import Path

from litagent.benchmark.quality_suite import suite_lock
from litagent.config import load_config
from litagent.contracts import build_config_summary
from litagent.eval.batched import located_findings_enabled
from litagent.frozen_writing import CASES, frozen_options, read_bundle, run_frozen
from litagent.observability.recorder import (
    ArchiveRepository,
    CompositeTraceHook,
    RedactingTraceHook,
    RunRecorder,
)
from litagent.observability.tracing import LangFuseTracer
from litagent.report_audit import atomic_json, digest
from litagent.runner import LitAgent, RunPolicy
from litagent.writing_v2 import (
    CONTRACT,
    FOCUSED_WRITING_PLAN,
    PROMPT_CONTRACT,
    RANGE_WRITING_PLAN,
    classify_failure,
    writing_plan_spec,
)
from scripts.run_stage_model_quality import check_models

VERSION = "frozen-writing-three-domain-v1"
LIMITS = {
    "cases": 3,
    "per_case_tokens": 500000,
    "per_case_requests": 64,
    "total_tokens": 1500000,
    "total_requests": 192,
    "repairs": 2,
    "retries": 0,
    "formal_cases": 0,
}
DISABLED_PROFILE = "disabled_v1"
THINKING_PROFILE = "writer_reviewer_thinking_v1"
LOW_THINKING_PROFILE = "writer_reviewer_low_thinking_v1"
LEGACY_RESPONSES_PROFILE = "writer_reviewer_low_responses_v1"
RESPONSES_PROFILE = "writer_reviewer_low_responses_v2"
SHAPE_PROFILE = "writer_reviewer_low_responses_shape_v1"
LEGACY_REEMISSION_PROFILE = "writer_reviewer_low_responses_reemit_v1"
REEMISSION_PROFILE = "writer_reviewer_low_responses_reemit_v2"
ROLE_PROFILE = "writer_low_reviewer_disabled_responses_reemit_v2"
EOF_PROFILE = "writer_low_reviewer_disabled_responses_eof_v1"
PLANNED_PROFILE = "writer_disabled_responses_planned_budget_v2"
ADAPTIVE_PROFILE = "writer_disabled_responses_planned_budget_v3"
BUDGETED_PROFILE = "writer_disabled_responses_planned_budget_v4"
FLAT_PROFILE = "writer_disabled_responses_planned_budget_v5"
CAPACITY_PROFILE = "writer_disabled_responses_planned_budget_v6"
ROUND_PROFILE = "writer_disabled_responses_planned_budget_v7"
PLANNED_PROFILES = (
    PLANNED_PROFILE,
    ADAPTIVE_PROFILE,
    BUDGETED_PROFILE,
    FLAT_PROFILE,
    CAPACITY_PROFILE,
    ROUND_PROFILE,
)
CONSERVATIVE_ACCOUNTING = "carry_terminal_rejection_v1"
PLANNED_UNKNOWN_ACCOUNTING = "carry_planned_writer_unknown_v1"
PLANNED_MIXED_UNKNOWN_ACCOUNTING = "carry_planned_mixed_unknown_v2"
HISTORICAL_BUDGET_HASHES = {
    "3758d71f3b5fd8004bc1bc51f8e8edd87ded28668cd739458e1c9901f835992f"
}
_HISTORICAL_COST_PLAN_PIN = {
    "root": "artifacts/acceptance/frozen-writer-three-domain-20261027-020",
    "run_id": "frozen-writer-three-domain-20261027-020-few-shot-metric",
    "case_id": "few-shot-metric",
    "profile": "writer_disabled_responses_planned_budget_v4",
    "anchor_manifest_sha256": (
        "d1e0c551be2164d5213b70dd17e5b472e231c19a7f8655936c4f527d61bd4efd"
    ),
    "anchor_receipt_sha256": (
        "8cadcad90d35e027eda039e8cd4efadbd29f8fe44a56a7fb03731b1050098dd0"
    ),
    "manifest_sha256": (
        "41de55d76648254a21edb3085c868f6f5d77e625e59179a0492b6d64ae452ef9"
    ),
    "state_sha256": (
        "74a3021aa5a8255032a412e2f2ad1beb6d698f0be248bdf8cc89a5d0ba26b7fe"
    ),
    "ledger_sha256": (
        "438293aab08744e8cc69c755b5003f83d8e27dd2d7e76730fa1e2a49292d499f"
    ),
    "archive_sha256": (
        "95f0d72fd88148b67d7070ac6fe7bd055f7f10f056f03291a5eb2299ff07bdaa"
    ),
    "caller_manifest_matches": True,
    "caller_state_matches": True,
    "caller_row_matches": True,
    "ledger_equals_snapshot": True,
    "receipt_history_before_matches": True,
    "anchor_receipt_matches": True,
    "requests": 3,
    "known_tokens": 32720,
    "unknown_hold": 0,
    "usage_complete": True,
}


def closeout_options(manifest):
    """Select a complete server-frozen release policy without upgrading old tasks."""
    from litagent.closeout import FIELDS, PROFILE, profile_identity

    if not set(FIELDS).intersection(manifest):
        return {}
    if (
        manifest.get("closeout_profile") != PROFILE
        or manifest.get("closeout_profile_identity") != profile_identity()
        or manifest.get("writing_profile") != ROUND_PROFILE
        or manifest.get("writer_source_view") != "source-only-writer-v1"
        or manifest.get("evaluation_dimensions_contract")
        != "separated-evaluation-dimensions-v1"
    ):
        raise ValueError("frozen_closeout_profile_invalid")
    return {"closeout_profile": PROFILE}


def profile_options(profile):
    """Bind new execution options while keeping historical manifests identical."""
    if profile == ROUND_PROFILE:
        return profile_options(CAPACITY_PROFILE).model_copy(
            update={"budget_policy": "planned-budget-v7"}
        )
    profile_mode(profile)
    options = frozen_options()
    if profile in (
        ADAPTIVE_PROFILE,
        BUDGETED_PROFILE,
        FLAT_PROFILE,
        CAPACITY_PROFILE,
        ROUND_PROFILE,
    ):
        options = options.model_copy(
            update={
                "budget_policy": (
                    (
                        "planned-budget-v6"
                        if profile in (CAPACITY_PROFILE, ROUND_PROFILE)
                        else "planned-budget-v5"
                    )
                    if profile in (FLAT_PROFILE, CAPACITY_PROFILE, ROUND_PROFILE)
                    else (
                        "planned-budget-v4"
                        if profile == BUDGETED_PROFILE
                        else "planned-budget-v3"
                    )
                )
            }
        )
    if profile in (FLAT_PROFILE, CAPACITY_PROFILE, ROUND_PROFILE):
        options = options.model_copy(update={"generation_strategy": "body_summary_v3"})
    if profile in (CAPACITY_PROFILE, ROUND_PROFILE):
        options = options.model_copy(update={"capacity_policy": "adaptive-capacity-v1"})
    return options


def inventory_recovery_options(manifest):
    """Allow one server-frozen recovery option without changing old manifests."""
    if "inventory_recovery" not in manifest:
        return {}
    if (
        manifest["inventory_recovery"] is not True
        or manifest.get("writing_profile") != ROUND_PROFILE
    ):
        raise ValueError("frozen_inventory_recovery_configuration_invalid")
    return {"inventory_recovery": True}


def mixed_inventory_recovery_options(manifest):
    """Enable the mixed contract only for an explicitly frozen v7 recovery."""
    if "mixed_inventory_recovery" not in manifest:
        return {}
    if (
        manifest["mixed_inventory_recovery"] is not True
        or manifest.get("inventory_recovery") is not True
        or manifest.get("writing_profile") != ROUND_PROFILE
    ):
        raise ValueError("frozen_mixed_inventory_recovery_configuration_invalid")
    return {"mixed_inventory_recovery": True}


def classification_review_options(manifest):
    """Keep new semantic responsibility explicit in server-frozen campaigns."""
    if "classification_review" not in manifest:
        return {}
    if (
        manifest["classification_review"] is not True
        or manifest.get("writing_profile") != ROUND_PROFILE
    ):
        raise ValueError("frozen_classification_review_configuration_invalid")
    return {"classification_review": True}


def repair_feedback_options(manifest):
    """Select the new feedback contract only in an explicitly frozen v7 suite."""
    if "repair_feedback" not in manifest:
        return {}
    if (
        manifest["repair_feedback"] is not True
        or manifest.get("writing_profile") != ROUND_PROFILE
    ):
        raise ValueError("frozen_repair_feedback_configuration_invalid")
    return {"repair_feedback": True}


def reference_contract_options(manifest):
    """Resolve one frozen reference policy without upgrading old manifests."""
    if "reference_contract" not in manifest:
        return {}
    from litagent.writing_flat import REFERENCE_SET_CONTRACT

    if (
        manifest["reference_contract"] != REFERENCE_SET_CONTRACT
        or manifest.get("writing_profile") != ROUND_PROFILE
    ):
        raise ValueError("frozen_reference_contract_configuration_invalid")
    return {"reference_contract": manifest["reference_contract"]}


def transport_revision_options(manifest):
    """Bind the lossless transport only to its explicitly frozen parent contracts."""
    if "transport_revision" not in manifest:
        return {}
    if (
        manifest["transport_revision"] != "host-lossless-transport-v1"
        or manifest.get("writing_profile") != ROUND_PROFILE
        or manifest.get("inventory_recovery") is not True
        or manifest.get("mixed_inventory_recovery") is not True
        or manifest.get("repair_feedback") is not True
        or manifest.get("reference_contract") != "ordered-reference-set-v1"
    ):
        raise ValueError("frozen_transport_revision_configuration_invalid")
    return {"transport_revision": manifest["transport_revision"]}


def support_validation_options(manifest):
    """Freeze support semantics only with every explicitly selected v7 parent."""
    if "support_validation_revision" not in manifest:
        return {}
    from litagent.eval.scoped import SUPPORT_VALIDATION_REVISIONS

    if (
        type(manifest["support_validation_revision"]) is not str
        or manifest["support_validation_revision"] not in SUPPORT_VALIDATION_REVISIONS
        or manifest.get("writing_profile") != ROUND_PROFILE
        or manifest.get("inventory_recovery") is not True
        or manifest.get("mixed_inventory_recovery") is not True
        or manifest.get("classification_review") is not True
        or manifest.get("repair_feedback") is not True
        or manifest.get("reference_contract") != "ordered-reference-set-v1"
        or manifest.get("transport_revision") != "host-lossless-transport-v1"
    ):
        raise ValueError("frozen_support_validation_configuration_invalid")
    return {"support_validation_revision": manifest["support_validation_revision"]}


def repair_mapping_options(manifest):
    """Select source lineage only from an explicit trusted frozen manifest."""
    if "repair_mapping_contract" not in manifest:
        return {}
    if (
        manifest["repair_mapping_contract"]
        not in (
            "capacity-edit-map-v2",
            "capacity-source-map-v3",
            "capacity-source-map-v4",
        )
        or manifest.get("writing_profile") != ROUND_PROFILE
        or manifest.get("repair_mapping_contract")
        in ("capacity-source-map-v3", "capacity-source-map-v4")
        and not support_validation_options(manifest)
    ):
        raise ValueError("frozen_repair_mapping_configuration_invalid")
    if manifest["repair_mapping_contract"] == "capacity-source-map-v4" and (
        manifest.get("support_validation_revision") != "host-support-receipts-v2"
        or manifest.get("repair_guidance_contract") != "host-structured-repair-v1"
        or manifest.get("repair_guidance_wire_revision") != "host-repair-task-view-v1"
    ):
        raise ValueError("frozen_repair_mapping_configuration_invalid")
    return {"repair_mapping_contract": manifest["repair_mapping_contract"]}


def repair_guidance_options(manifest):
    """Select structured repair inputs only through the complete frozen v3 path."""
    if "repair_guidance_contract" not in manifest:
        return {}
    if (
        manifest["repair_guidance_contract"] != "host-structured-repair-v1"
        or manifest.get("writing_profile") != ROUND_PROFILE
        or manifest.get("repair_mapping_contract")
        not in ("capacity-source-map-v3", "capacity-source-map-v4")
        or not support_validation_options(manifest)
    ):
        raise ValueError("frozen_repair_guidance_configuration_invalid")
    return {"repair_guidance_contract": manifest["repair_guidance_contract"]}


def repair_guidance_wire_options(manifest):
    """Select one pinned input representation without adding public run options."""
    fields = {"repair_guidance_wire_revision", "repair_guidance_wire_identity"}
    if not (fields | {"repair_guidance_capacity_rule"}).intersection(manifest):
        return {}
    from litagent.writing_flat import (
        REPAIR_GUIDANCE_WIRE_REVISION,
        REPAIR_GUIDANCE_WIRE_REVISIONS,
        REPAIR_TASK_VIEW_ADMISSION_RULE,
        REPAIR_TASK_VIEW_REVISION,
        repair_guidance_wire_identity,
    )

    revision = manifest.get("repair_guidance_wire_revision")
    task_view = revision == REPAIR_TASK_VIEW_REVISION
    if (
        not fields.issubset(manifest)
        or type(revision) is not str
        or revision not in REPAIR_GUIDANCE_WIRE_REVISIONS
        or manifest["repair_guidance_wire_identity"]
        != (
            repair_guidance_wire_identity()
            if revision == REPAIR_GUIDANCE_WIRE_REVISION
            else repair_guidance_wire_identity(revision)
        )
        or ("repair_guidance_capacity_rule" in manifest) is not task_view
        or task_view
        and manifest["repair_guidance_capacity_rule"] != REPAIR_TASK_VIEW_ADMISSION_RULE
        or not repair_guidance_options(manifest)
    ):
        raise ValueError("frozen_repair_guidance_wire_configuration_invalid")
    return {"repair_guidance_wire_revision": manifest["repair_guidance_wire_revision"]}


def semantic_conflict_options(manifest):
    """Select the private conservative policy only from a complete frozen stack."""
    fields = {"semantic_conflict_policy", "semantic_conflict_identity"}
    if not fields.intersection(manifest):
        return {}
    from litagent.eval.semantic_conflict import (
        SEMANTIC_CONFLICT_POLICY,
        semantic_conflict_identity,
    )

    if (
        not fields.issubset(manifest)
        or manifest["semantic_conflict_policy"] != SEMANTIC_CONFLICT_POLICY
        or manifest["semantic_conflict_identity"] != semantic_conflict_identity()
        or manifest.get("writing_profile") != ROUND_PROFILE
        or manifest.get("support_validation_revision") != "host-support-receipts-v2"
        or manifest.get("repair_mapping_contract") != "capacity-source-map-v4"
        or manifest.get("repair_guidance_contract") != "host-structured-repair-v1"
        or manifest.get("repair_guidance_wire_revision") != "host-repair-task-view-v1"
        or not repair_guidance_wire_options(manifest)
    ):
        raise ValueError("frozen_semantic_conflict_configuration_invalid")
    return {"semantic_conflict_policy": manifest["semantic_conflict_policy"]}


def evaluation_response_options(manifest):
    """Select response constraints only from a complete private frozen manifest."""
    fields = {"evaluation_response_contract", "evaluation_response_identity"}
    if not fields.intersection(manifest):
        return {}
    from litagent.eval.batched import (
        EVALUATION_RESPONSE_CONTRACTS,
        response_contract_identity,
    )

    if (
        not fields.issubset(manifest)
        or type(manifest["evaluation_response_contract"]) is not str
        or manifest["evaluation_response_contract"] not in EVALUATION_RESPONSE_CONTRACTS
        or manifest["evaluation_response_identity"]
        != response_contract_identity(contract=manifest["evaluation_response_contract"])
        or manifest.get("writing_profile") != ROUND_PROFILE
        or manifest.get("support_validation_revision") != "host-support-receipts-v2"
        or not support_validation_options(manifest)
    ):
        raise ValueError("frozen_evaluation_response_configuration_invalid")
    return {"evaluation_response_contract": manifest["evaluation_response_contract"]}


def evaluation_dimensions_options(manifest):
    """Select dimensions only from a complete private frozen response stack."""
    fields = {"evaluation_dimensions_contract", "evaluation_dimensions_identity"}
    if not fields.intersection(manifest):
        return {}
    from litagent.eval.dimensions import dimensions_identity

    if (
        not fields.issubset(manifest)
        or type(manifest["evaluation_dimensions_contract"]) is not str
        or manifest["evaluation_dimensions_contract"]
        != "separated-evaluation-dimensions-v1"
        or manifest["evaluation_dimensions_identity"] != dimensions_identity()
        or manifest.get("writing_profile") != ROUND_PROFILE
        or manifest.get("support_validation_revision") != "host-support-receipts-v2"
        or manifest.get("evaluation_response_contract") != "host-bound-evaluation-v2"
        or not evaluation_response_options(manifest)
    ):
        raise ValueError("frozen_evaluation_dimensions_configuration_invalid")
    return {
        "evaluation_dimensions_contract": manifest["evaluation_dimensions_contract"]
    }


def writer_source_view_options(manifest):
    """Select the private source view only with its complete frozen parent stack."""
    fields = {"writer_source_view", "writer_source_view_identity"}
    if not fields.intersection(manifest):
        return {}
    from litagent.writer_source_view import source_view_identity

    if (
        not fields.issubset(manifest)
        or type(manifest["writer_source_view"]) is not str
        or manifest["writer_source_view"] != "source-only-writer-v1"
        or manifest["writer_source_view_identity"] != source_view_identity()
        or manifest.get("writing_profile") != ROUND_PROFILE
        or manifest.get("inventory_recovery") is not True
        or manifest.get("mixed_inventory_recovery") is not True
        or manifest.get("classification_review") is not True
        or manifest.get("repair_feedback") is not True
        or manifest.get("repair_mapping_contract") != "capacity-source-map-v4"
        or manifest.get("evaluation_response_contract") != "host-bound-evaluation-v2"
        or not evaluation_response_options(manifest)
        or not repair_prompt_options(manifest)
        or not repair_guidance_wire_options(manifest)
        or not repair_mapping_options(manifest)
    ):
        raise ValueError("frozen_writer_source_view_configuration_invalid")
    return {"writer_source_view": manifest["writer_source_view"]}


def repair_prompt_options(manifest):
    """Select repair instructions only through a complete private frozen pair."""
    fields = {"repair_prompt_revision", "repair_prompt_identity"}
    if not fields.intersection(manifest):
        return {}
    from litagent.writing_flat import REPAIR_PROMPT_REVISION, repair_prompt_identity

    if (
        not fields.issubset(manifest)
        or type(manifest["repair_prompt_revision"]) is not str
        or manifest["repair_prompt_revision"] != REPAIR_PROMPT_REVISION
        or manifest["repair_prompt_identity"] != repair_prompt_identity()
        or manifest.get("writing_profile") != ROUND_PROFILE
        or manifest.get("repair_mapping_contract") != "capacity-source-map-v4"
        or manifest.get("repair_guidance_contract") != "host-structured-repair-v1"
        or manifest.get("repair_guidance_wire_revision") != "host-repair-task-view-v1"
        or manifest.get("support_validation_revision") != "host-support-receipts-v2"
        or not repair_guidance_wire_options(manifest)
        or not repair_mapping_options(manifest)
    ):
        raise ValueError("frozen_repair_prompt_configuration_invalid")
    return {"repair_prompt_revision": manifest["repair_prompt_revision"]}


def round_resource_options(manifest):
    """Bind the opt-in round resource rule outside all model-facing envelopes."""
    fields = {"round_resource_contract", "round_resource_identity"}
    if not fields.intersection(manifest):
        return {}
    from litagent.safety.planned_budget import (
        JOINT_ROUND_ESCROW_CONTRACT,
        joint_round_escrow_identity,
    )

    if (
        not fields.issubset(manifest)
        or type(manifest["round_resource_contract"]) is not str
        or manifest["round_resource_contract"] != JOINT_ROUND_ESCROW_CONTRACT
        or manifest["round_resource_identity"] != joint_round_escrow_identity()
        or manifest.get("writing_profile") != ROUND_PROFILE
    ):
        raise ValueError("frozen_round_resource_configuration_invalid")
    return {"round_resource_contract": manifest["round_resource_contract"]}


def manifest_options(manifest):
    """Preserve historical run identities unless the new option is explicit."""
    repair_guidance_options(manifest)
    repair_guidance_wire_options(manifest)
    semantic_conflict_options(manifest)
    evaluation_response_options(manifest)
    repair_prompt_options(manifest)
    round_resource_options(manifest)
    writer_source_view_options(manifest)
    evaluation_dimensions_options(manifest)
    options = profile_options(manifest.get("writing_profile", DISABLED_PROFILE))
    changes = {
        **classification_review_options(manifest),
        **repair_feedback_options(manifest),
        **reference_contract_options(manifest),
        **transport_revision_options(manifest),
        **support_validation_options(manifest),
        **mixed_inventory_recovery_options(manifest),
    }
    if changes:
        from litagent.workbench import RunOptions

        options = RunOptions(**{**options.model_dump(mode="json"), **changes})
    return options.model_dump(mode="json")


def evaluation_attempt_label(attempt, index):
    """Keep inventory recovery distinct from a changed-prose candidate."""
    prefix = (
        "inventory_recovery"
        if attempt.get("kind") == "inventory_recovery"
        else "repair"
    )
    return f"{prefix}_{index + 1}"


def profile_mode(profile):
    """Resolve only server-defined campaign modes, never arbitrary provider kwargs."""
    if profile in (DISABLED_PROFILE, *PLANNED_PROFILES):
        return "disabled"
    if profile in (
        THINKING_PROFILE,
        LOW_THINKING_PROFILE,
        RESPONSES_PROFILE,
        LEGACY_RESPONSES_PROFILE,
        SHAPE_PROFILE,
        REEMISSION_PROFILE,
        ROLE_PROFILE,
        EOF_PROFILE,
        LEGACY_REEMISSION_PROFILE,
    ):
        return "enabled"
    raise ValueError("frozen_writing_profile_invalid")


def profile_parameters(profile, *, writing_plan=None):
    """Bind the single supported native effort without adding legacy config fields."""
    if profile == ROUND_PROFILE:
        return {
            **profile_parameters(CAPACITY_PROFILE, writing_plan=writing_plan),
            "budget_policy": "planned-budget-v7",
        }
    profile_mode(profile)
    writing_plan_spec(writing_plan)
    if writing_plan is not None and profile not in (EOF_PROFILE, *PLANNED_PROFILES):
        raise ValueError("frozen_writing_plan_configuration_invalid")
    if profile in PLANNED_PROFILES:
        from litagent.writing_v2 import EOF_CONTAINER_POLICY

        if writing_plan != RANGE_WRITING_PLAN:
            raise ValueError("planned_budget_requires_range_plan")
        return {
            "native_writing_plan": writing_plan,
            "native_reviewer_thinking_mode": "disabled",
            "native_provider_api": "responses",
            "native_syntax_policy": EOF_CONTAINER_POLICY,
            "budget_policy": (
                (
                    "planned-budget-v6"
                    if profile in (CAPACITY_PROFILE, ROUND_PROFILE)
                    else "planned-budget-v5"
                )
                if profile in (FLAT_PROFILE, CAPACITY_PROFILE, ROUND_PROFILE)
                else (
                    "planned-budget-v4"
                    if profile == BUDGETED_PROFILE
                    else (
                        "planned-budget-v3"
                        if profile == ADAPTIVE_PROFILE
                        else "planned-budget-v2"
                    )
                )
            ),
        }
    if profile in (
        RESPONSES_PROFILE,
        LEGACY_RESPONSES_PROFILE,
        SHAPE_PROFILE,
        REEMISSION_PROFILE,
        ROLE_PROFILE,
        EOF_PROFILE,
        LEGACY_REEMISSION_PROFILE,
    ):
        from litagent.writing_v2 import (
            EOF_CONTAINER_POLICY,
            INTACT_UNIT_REEMISSION_CONTRACT,
            LEGACY_INTACT_UNIT_REEMISSION_CONTRACT,
        )

        return {
            **(
                {"native_writing_plan": writing_plan}
                if writing_plan is not None
                else {}
            ),
            "native_reasoning_effort": "low",
            **(
                {"native_reviewer_thinking_mode": "disabled"}
                if profile in (ROLE_PROFILE, EOF_PROFILE)
                else {}
            ),
            **(
                {"native_syntax_policy": EOF_CONTAINER_POLICY}
                if profile == EOF_PROFILE
                else {}
            ),
            "native_provider_api": "responses",
            **({"native_shape_recovery": True} if profile == SHAPE_PROFILE else {}),
            **(
                {
                    "native_reemission_contract": (
                        INTACT_UNIT_REEMISSION_CONTRACT
                        if profile in (REEMISSION_PROFILE, ROLE_PROFILE)
                        else LEGACY_INTACT_UNIT_REEMISSION_CONTRACT
                    )
                }
                if profile
                in (REEMISSION_PROFILE, ROLE_PROFILE, LEGACY_REEMISSION_PROFILE)
                else {}
            ),
        }
    return {"native_reasoning_effort": "low"} if profile == LOW_THINKING_PROFILE else {}


def profile_config(config, profile):
    """Create an explicit experiment override without mutating deployment defaults."""
    mode = profile_mode(profile)
    if profile == DISABLED_PROFILE:
        return config
    result = config.model_copy(deep=True)
    result.adversarial.thinking_mode = mode
    return result


def check_profile(config, profile, *, writing_plan=None):
    """Reuse the unchanged baseline gate while checking the explicit mode separately."""
    profile_parameters(profile, writing_plan=writing_plan)
    if profile in (LEGACY_RESPONSES_PROFILE, LEGACY_REEMISSION_PROFILE):
        raise ValueError("frozen_writing_profile_read_only")
    if config.adversarial.thinking_mode != profile_mode(profile):
        raise ValueError("frozen_writing_profile_mode_conflict")
    if profile == DISABLED_PROFILE:
        check_models(config)
        return
    if config.adversarial.review_max_tokens != 4096 or config.eval.max_tokens != 16384:
        raise ValueError("frozen_writing_profile_output_conflict")
    baseline = config.model_copy(deep=True)
    baseline.adversarial.thinking_mode = "disabled"
    check_models(baseline)


def profile_description(profile, summary, *, writing_plan=None):
    """Publish only mode and output metadata that belongs to the frozen config."""
    if profile == ROUND_PROFILE:
        return {
            **profile_description(CAPACITY_PROFILE, summary, writing_plan=writing_plan),
            "id": profile,
            "budget_policy": "planned-budget-v7",
        }
    profile_parameters(profile, writing_plan=writing_plan)
    mode = profile_mode(profile)
    summary = summary.get("effective", {})
    llm = summary.get("llm", {})
    routing = llm.get("model_routing", {})
    result = {
        "id": profile,
        "generation_thinking_mode": mode,
        "reviewer_thinking_mode": (
            "disabled" if profile in (ROLE_PROFILE, EOF_PROFILE) else mode
        ),
        "evaluation_thinking_mode": "disabled",
        "output_limit_parameter": (
            "max_tokens" if mode == "disabled" else "max_completion_tokens"
        ),
        "generation_output_tokens": llm.get("max_tokens"),
        "reviewer_output_tokens": summary.get("adversarial", {}).get(
            "review_max_tokens"
        ),
        "evaluation_output_tokens": summary.get("eval", {}).get("max_tokens"),
        "configured_models": {
            role: routing.get(role, llm.get("model"))
            for role in ("extraction", "generation", "evaluation")
        },
    }
    if profile in PLANNED_PROFILES:
        from litagent.writing_v2 import EOF_CONTAINER_POLICY

        result.update(
            provider_api="responses",
            output_limit_parameter="max_output_tokens",
            reviewer_thinking_mode="disabled",
            reviewer_reasoning_effort=None,
            syntax_policy=EOF_CONTAINER_POLICY,
            budget_policy=(
                (
                    "planned-budget-v6"
                    if profile in (CAPACITY_PROFILE, ROUND_PROFILE)
                    else "planned-budget-v5"
                )
                if profile in (FLAT_PROFILE, CAPACITY_PROFILE, ROUND_PROFILE)
                else (
                    "planned-budget-v4"
                    if profile == BUDGETED_PROFILE
                    else (
                        "planned-budget-v3"
                        if profile == ADAPTIVE_PROFILE
                        else "planned-budget-v2"
                    )
                )
            ),
        )
    if profile in (ROLE_PROFILE, EOF_PROFILE):
        result["reviewer_reasoning_effort"] = None
    if profile in (
        LOW_THINKING_PROFILE,
        RESPONSES_PROFILE,
        LEGACY_RESPONSES_PROFILE,
        SHAPE_PROFILE,
        REEMISSION_PROFILE,
        ROLE_PROFILE,
        EOF_PROFILE,
        LEGACY_REEMISSION_PROFILE,
    ):
        result["reasoning_effort"] = "low"
    if profile in (
        RESPONSES_PROFILE,
        LEGACY_RESPONSES_PROFILE,
        SHAPE_PROFILE,
        REEMISSION_PROFILE,
        ROLE_PROFILE,
        EOF_PROFILE,
        LEGACY_REEMISSION_PROFILE,
    ):
        result["provider_api"] = "responses"
        result["output_limit_parameter"] = "max_output_tokens"
    if profile in (
        RESPONSES_PROFILE,
        SHAPE_PROFILE,
        REEMISSION_PROFILE,
        ROLE_PROFILE,
        EOF_PROFILE,
        LEGACY_REEMISSION_PROFILE,
    ):
        from litagent.llm.responses import RESPONSES_CONTRACT

        result.update(
            mapping_contract=RESPONSES_CONTRACT,
            storage=False,
            stream=False,
            host_input="complete",
            server_truncation_control="unavailable",
        )
    if profile == SHAPE_PROFILE:
        from litagent.writing_v2 import SHAPE_RECOVERY_CONTRACT

        result["shape_recovery_contract"] = SHAPE_RECOVERY_CONTRACT
    if profile in (REEMISSION_PROFILE, ROLE_PROFILE, LEGACY_REEMISSION_PROFILE):
        from litagent.writing_v2 import (
            INTACT_UNIT_REEMISSION_CONTRACT,
            LEGACY_INTACT_UNIT_REEMISSION_CONTRACT,
        )

        result["reemission_contract"] = (
            INTACT_UNIT_REEMISSION_CONTRACT
            if profile in (REEMISSION_PROFILE, ROLE_PROFILE)
            else LEGACY_INTACT_UNIT_REEMISSION_CONTRACT
        )
    if profile == EOF_PROFILE:
        from litagent.writing_v2 import EOF_CONTAINER_POLICY

        result["syntax_policy"] = EOF_CONTAINER_POLICY
    if writing_plan is not None:
        result["writing_plan"] = writing_plan_spec(writing_plan)
    return result


def runtime_identity():
    """Bind core transport, schema and workflow dependencies before execution."""
    return {
        "python": sys.version,
        "packages": {
            name: version(name)
            for name in ("openai", "httpx", "pydantic", "langgraph", "PyMuPDF")
        },
    }


def source_paths():
    """Freeze the complete implementation and test inventory, including new files."""
    tracked = subprocess.check_output(["git", "ls-files"], text=True).splitlines()
    paths = [
        Path(p)
        for p in tracked
        if p.startswith(("src/", "tests/", "scripts/", "config/"))
        or p in {"pyproject.toml", "requirements-demo.txt"}
    ]
    paths += (
        list(Path("src/litagent").rglob("*.py"))
        + list(Path("scripts").glob("*.py"))
        + list(Path("tests").rglob("*.py"))
        + list(Path("tests/fixtures").rglob("*.json"))
    )
    return sorted({p for p in paths if p.is_file() and "__pycache__" not in str(p)})


def bound_native_receipt(start, end, journal, *, reservation_counter=None):
    """Bind exactly one private HTTP receipt to the frozen host request identity."""
    from litagent.llm.responses import ARK_NATIVE_FIELDS, decode_private_receipt
    from litagent.quality import digest as writer_digest
    from litagent.safety.budget import strict_request_reservation

    reservation_counter = reservation_counter or strict_request_reservation

    matches = []
    for call in journal.get("calls", {}).values():
        receipt = call.get("provider_receipt")
        if not isinstance(receipt, dict):
            continue
        request, response = decode_private_receipt(receipt)
        rf = start.get("response_format")
        cap = start.get("max_tokens")
        mode, effort = call.get("thinking_mode"), call.get("reasoning_effort")
        extra = None if mode == "provider_default" else {"thinking": {"type": mode}}
        identity = [start.get("messages"), rf, {"max_output_tokens": cap}, extra]
        if effort is not None:
            identity.append({"reasoning_effort": effort})
        identity.append(["responses", receipt["contract"]])
        if call.get("input_sha256") != writer_digest(identity):
            continue
        if (
            not isinstance(rf, dict)
            or rf.get("type") != "json_schema"
            or type(cap) is not int
            or not 0 < cap <= 20000
            or set(request)
            - (
                ARK_NATIVE_FIELDS
                | (
                    {"background", "truncation"}
                    if receipt["contract"].endswith("-v1")
                    else set()
                )
            )
            or (
                receipt["contract"].endswith("-v1")
                and (
                    request.get("background") is not False
                    or request.get("truncation") != "disabled"
                )
            )
            or request.get("input") != start.get("messages")
            or request.get("text")
            != {"format": {"type": "json_schema", **rf["json_schema"]}}
            or request.get("model") != start.get("model")
            or request.get("max_output_tokens") != cap
            or request.get("thinking") != (extra or {}).get("thinking")
            or request.get("reasoning") != ({"effort": effort} if effort else None)
            or request.get("store") is not False
            or request.get("stream") is not False
            or call.get("provider_api") != "responses"
            or start.get("provider_api") != "responses"
            or start.get("output_limit_parameter") != "max_output_tokens"
            or end.get("model") != start.get("model")
            or end.get("messages") != start.get("messages")
            or call.get("provider_schema_sha256")
            != writer_digest(rf["json_schema"]["schema"])
            or call.get("reservation")
            != reservation_counter(start["messages"], cap, rf)
            or len(base64.b64decode(receipt["request_body_base64"], validate=True))
            + cap
            > call.get("reservation", 0)
            or start.get("reasoning_effort") != effort
            or call.get("max_tokens") != cap
            or call.get("output_limit_parameter") != "max_output_tokens"
        ):
            raise ValueError("frozen_accounting_receipt_binding_invalid")
        matches.append((call, receipt, request, response))
    if len(matches) != 1:
        raise ValueError("frozen_accounting_receipt_binding_missing_or_ambiguous")
    return matches[0]


def historical_accounting_budget(root, manifest):
    """Load only reviewed immutable legacy accounting, never a runtime owner."""
    from litagent.safety import budget as current

    name = "src/litagent/safety/budget.py"
    saved = Path(root) / "source" / name
    if not saved.is_file():
        raise ValueError("frozen_accounting_unknown_proof_missing")
    source = saved.read_bytes()
    sha = hashlib.sha256(source).hexdigest()
    if manifest.get("source_hashes", {}).get(name) != sha:
        raise ValueError("frozen_accounting_unknown_proof_missing")
    if sha == hashlib.sha256(Path(current.__file__).read_bytes()).hexdigest():
        return current.CostBudget, current.strict_request_reservation, sha
    if sha not in HISTORICAL_BUDGET_HASHES:
        raise ValueError("frozen_accounting_historical_source_unreviewed")
    # The allowlist pins reviewed repository bytes before execution. Do not
    # import arbitrary manifest-selected Python or replace shared runtime modules.
    module = types.ModuleType("_litagent_historical_accounting_" + sha)
    exec(compile(source, str(saved), "exec"), module.__dict__)
    return module.CostBudget, module.strict_request_reservation, sha


def unknown_token_hold(snapshot, known_tokens):
    """Count sent unsettled escrows separately from known planned usage."""
    if (
        not isinstance(snapshot, dict)
        or type(known_tokens) is not int
        or known_tokens < 0
    ):
        raise ValueError("frozen_accounting_budget_snapshot_invalid")
    if snapshot.get("schema_version") != 2:
        return max(0, snapshot.get("used_tokens", 0) - known_tokens)
    from litagent.safety.budget import CostBudget

    # Restoration validates the complete plan and accounting without a path or
    # transport. Unsent escrows stay blocked but do not invent a service charge.
    validated = CostBudget.from_snapshot(snapshot).snapshot()
    if validated["used_tokens"] != known_tokens:
        raise ValueError("frozen_accounting_planned_known_usage_conflict")
    return sum(
        entry["reserved_tokens"]
        for entry in validated["entries"].values()
        if entry["sent"] and entry["status"] != "settled"
    )


def historical_unknown_token_hold(snapshot, known_tokens, *, expected_plan):
    """Inspect pinned historical costs without restoring current runtime authority."""
    if (
        not isinstance(snapshot, dict)
        or type(known_tokens) is not int
        or known_tokens < 0
    ):
        raise ValueError("frozen_accounting_budget_snapshot_invalid")
    if snapshot.get("schema_version") != 2 and not isinstance(expected_plan, dict):
        return unknown_token_hold(snapshot, known_tokens)
    from litagent.safety.planned_accounting import validate_accounting_snapshot

    validated = validate_accounting_snapshot(snapshot, expected_plan=expected_plan)
    if validated["used_tokens"] != known_tokens:
        raise ValueError("frozen_accounting_planned_known_usage_conflict")
    return sum(
        entry["reserved_tokens"]
        for entry in validated["entries"].values()
        if entry["sent"] and entry["status"] != "settled"
    )


def _historical_cost_plan_pin(snapshot, prepared_plan, binding):
    """Read one reviewed old cost plan; this never grants runtime authority."""
    from litagent.report_audit import digest as data_digest
    from litagent.safety.planned_accounting import validate_accounting_snapshot

    _mixed_require(
        isinstance(binding, dict)
        and set(binding) == set(_HISTORICAL_COST_PLAN_PIN)
        and all(
            type(binding[key]) is type(value) and binding[key] == value
            for key, value in _HISTORICAL_COST_PLAN_PIN.items()
        )
        and isinstance(snapshot, dict)
        and data_digest(prepared_plan)
        == "db4af224e1a074ea3b80932e4cf310fd9b6a2829515c75e9ab9e0bdbbbab58c5"
        and data_digest(snapshot)
        == "6e394cc678a81297d1dee4e9a6c309085f4fecbd60360915dada0e853a0b8acd"
        and data_digest(snapshot.get("plan"))
        == "b7fe6807299f4abadd1efaf923c1cc71f7166d829642e163c79dc2ea7add63b6",
        "historical_plan_pin_invalid",
    )
    _mixed_require(
        snapshot["plan"]["contract"] == prepared_plan["contract"] == "planned-budget-v4"
        and snapshot["usage_uncertain"] is False
        and (snapshot["used_tokens"], snapshot["call_count"], snapshot["requests_sent"])
        == (32720, 3, 3)
        and len(snapshot["entries"]) == 3
        and all(
            entry["sent"] is True and entry["status"] == "settled"
            for entry in snapshot["entries"].values()
        ),
        "historical_plan_pin_invalid",
    )
    actual = snapshot["plan"]
    validate_accounting_snapshot(snapshot, expected_plan=actual)
    return copy.deepcopy(actual)


def _historical_accounting_plan(root, manifest, state, row):
    """Bind the sole reviewed prepare/runtime drift to original immutable files."""
    from litagent.report_audit import digest as data_digest

    pin = _HISTORICAL_COST_PLAN_PIN
    _mixed_require(
        Path(root).resolve() == Path(pin["root"]).resolve()
        and row.get("run_id") == pin["run_id"]
        and row.get("case_id") == pin["case_id"],
        "historical_plan_pin_invalid",
    )
    try:
        mp, sp = Path(root) / "manifest.json", Path(root) / "state.json"
        ledger_path = Path(
            "artifacts/targeted-repairs", pin["run_id"] + ".json.budget-ledger.json"
        )
        archive_path = Path("artifacts/runs", pin["run_id"] + ".json")
        anchor_path = Path(
            "artifacts/acceptance/frozen-writer-three-domain-20261101-040/manifest.json"
        )
        raw = {
            "manifest": mp.read_bytes(),
            "state": sp.read_bytes(),
            "ledger": ledger_path.read_bytes(),
            "archive": archive_path.read_bytes(),
            "anchor_manifest": anchor_path.read_bytes(),
        }
        saved = {key: json.loads(value) for key, value in raw.items()}
        receipts = [
            value
            for value in saved["anchor_manifest"]["accounting_receipts"]
            if value["root"] == pin["root"]
        ]
        _mixed_require(len(receipts) == 1, "historical_plan_pin_invalid")
        receipt = receipts[0]
        original_rows = [
            value
            for value in saved["state"]["cases"]
            if value.get("case_id") == pin["case_id"]
        ]
        binding = {
            key + "_sha256": hashlib.sha256(value).hexdigest()
            for key, value in raw.items()
        }
        binding.update(
            root=pin["root"],
            run_id=row.get("run_id"),
            case_id=row.get("case_id"),
            profile=manifest.get("writing_profile"),
            anchor_receipt_sha256=data_digest(receipt),
            caller_manifest_matches=data_digest(manifest)
            == data_digest(saved["manifest"]),
            caller_state_matches=data_digest(state) == data_digest(saved["state"]),
            caller_row_matches=len(original_rows) == 1
            and data_digest(row) == data_digest(original_rows[0]),
            ledger_equals_snapshot=saved["ledger"] == row.get("budget_snapshot"),
            receipt_history_before_matches=receipt["history_before"]
            == manifest.get("history"),
            anchor_receipt_matches=receipt["manifest_sha256"]
            == binding["manifest_sha256"]
            and receipt["state_sha256"] == binding["state_sha256"]
            and receipt["archives"] == {str(archive_path): binding["archive_sha256"]}
            and receipt["requests"] == row.get("requests")
            and receipt["known_tokens"] == row.get("known_tokens")
            and receipt.get("unknown_hold", 0) == 0,
            requests=row.get("requests"),
            known_tokens=row.get("known_tokens"),
            unknown_hold=row.get("unknown_hold"),
            usage_complete=row.get("usage_complete"),
        )
        return _historical_cost_plan_pin(
            row["budget_snapshot"], manifest["budget_plans"][pin["case_id"]], binding
        )
    except (KeyError, TypeError, AttributeError, OSError, UnicodeError) as exc:
        raise ValueError("frozen_accounting_mixed_historical_plan_pin_invalid") from exc


def _historical_source_view_binding(value, envelope, *, call=False):
    """Authenticate saved view authority without running the current projector."""
    fields = ("writer_source_view", "writer_source_view_identity")
    selected = any(key in envelope for key in fields)
    _mixed_require(
        isinstance(value, dict)
        and all(
            (key in value) == (key in envelope) and value.get(key) == envelope.get(key)
            for key in fields
        ),
        "writer_source_view_binding_invalid",
    )
    if not call:
        _mixed_require(
            ("writer_source_view_source_sha256" in value) == selected
            and (
                not selected
                or isinstance(value["writer_source_view_source_sha256"], str)
                and re.fullmatch(
                    r"[0-9a-f]{64}", value["writer_source_view_source_sha256"]
                )
                is not None
            ),
            "writer_source_view_snapshot_invalid",
        )
        return
    _mixed_require(
        ("writer_source_view_record" in value) == selected,
        "writer_source_view_record_invalid",
    )
    if selected:
        record = value["writer_source_view_record"]
        _mixed_require(
            isinstance(record, dict)
            and set(record)
            == {"contract", "identity", "source_snapshot", "view_sha", "system_sha"}
            and record.get("contract") == envelope["writer_source_view"]
            and record.get("identity") == envelope["writer_source_view_identity"]
            and all(
                isinstance(record.get(key), str)
                and re.fullmatch(r"[0-9a-f]{64}", record[key]) is not None
                for key in ("source_snapshot", "view_sha", "system_sha")
            ),
            "writer_source_view_record_invalid",
        )


def _historical_source_view_request(start, envelope, call):
    """Bind the saved local view record to actual model-visible message bytes."""
    if "writer_source_view" not in envelope:
        return
    from litagent.quality import digest as wire_digest

    _historical_source_view_binding(call, envelope, call=True)
    try:
        messages = start["messages"]
        payload = json.loads(messages[1]["content"])
        system = messages[0]["content"]
        record = call["writer_source_view_record"]
        _mixed_require(
            record["view_sha"] == call.get("payload_sha256") == wire_digest(payload)
            and record["system_sha"]
            == call.get("system_sha256")
            == wire_digest(system),
            "writer_source_view_request_invalid",
        )
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError(
            "frozen_accounting_mixed_writer_source_view_request_invalid"
        ) from exc


def _historical_source_view_archive(journal, envelope, archive):
    """Use the pinned journal snapshot and an available canonical archive packet."""
    if "writer_source_view" not in envelope:
        return
    from litagent.quality import digest as wire_digest

    try:
        expected = journal["writer_source_view_source_sha256"]
        metadata = (archive.get("report") or {}).get("metadata") or {}
        document = metadata.get("writer_document")
        if document is not None:
            packet = document["packet"]
            _mixed_require(
                isinstance(packet, dict) and wire_digest(packet) == expected,
                "writer_source_view_snapshot_invalid",
            )
        for call in journal["calls"].values():
            _historical_source_view_binding(call, envelope, call=True)
            _mixed_require(
                call["writer_source_view_record"]["source_snapshot"] == expected,
                "writer_source_view_snapshot_invalid",
            )
    except (KeyError, TypeError, AttributeError) as exc:
        raise ValueError(
            "frozen_accounting_mixed_writer_source_view_snapshot_authority_missing"
        ) from exc


def _historical_dimensions_binding(value, envelope):
    """Compare pinned dimensions without importing current executable semantics."""
    fields = ("evaluation_dimensions_contract", "evaluation_dimensions_identity")
    _mixed_require(
        isinstance(value, dict)
        and all(
            (key in value) == (key in envelope) and value.get(key) == envelope.get(key)
            for key in fields
        ),
        "evaluation_dimensions_binding_invalid",
    )


def _historical_support_binding(value, envelope):
    """Compare recorded semantic tags with their parent, never today's prompt."""
    _historical_dimensions_binding(value, envelope)
    fields = (
        "support_validation_revision",
        "support_validation_identity",
        "semantic_conflict_policy",
        "semantic_conflict_identity",
    )
    _mixed_require(
        isinstance(value, dict)
        and all(
            (key in value) == (key in envelope) and value.get(key) == envelope.get(key)
            for key in fields
        ),
        "semantic_binding_invalid",
    )


def _historical_guidance_wire_owner(value, envelope):
    """Keep the recorded wire selector pinned without invoking its codec."""
    fields = (
        "repair_guidance_wire_revision",
        "repair_guidance_wire_identity",
        "repair_guidance_capacity_rule",
        "repair_prompt_revision",
        "repair_prompt_identity",
    )
    _mixed_require(
        isinstance(value, dict)
        and all(
            (key in value) == (key in envelope) and value.get(key) == envelope.get(key)
            for key in fields
        ),
        "guidance_wire_owner_invalid",
    )


def _historical_json_positions(raw):
    """Index strict JSON paths as pure data, without loading any writer parser."""
    if not isinstance(raw, str) or len(raw.encode()) > 1048576:
        raise ValueError("historical_json_size_invalid")

    def constant(value):
        raise ValueError("historical_json_constant_invalid")

    decoder = json.JSONDecoder(parse_constant=constant)
    positions = {}

    def whitespace(index):
        while index < len(raw) and raw[index] in " \t\r\n":
            index += 1
        return index

    def visit(index, path, depth=0):
        if depth > 128 or len(positions) > 100000:
            raise ValueError("historical_json_depth_invalid")
        start = whitespace(index)
        if start >= len(raw):
            raise ValueError("historical_json_incomplete")
        token = raw[start]
        if token in "{[":
            object_value = token == "{"
            close = "}" if object_value else "]"
            value = {} if object_value else []
            index = whitespace(start + 1)
            if index < len(raw) and raw[index] == close:
                index += 1
            else:
                while True:
                    if object_value:
                        key, index = decoder.raw_decode(raw, index)
                        if not isinstance(key, str) or key in value:
                            raise ValueError("historical_json_key_invalid")
                        index = whitespace(index)
                        if index >= len(raw) or raw[index] != ":":
                            raise ValueError("historical_json_colon_invalid")
                        index += 1
                    else:
                        key = len(value)
                    child, index = visit(index, (*path, key), depth + 1)
                    if object_value:
                        value[key] = child
                    else:
                        value.append(child)
                    index = whitespace(index)
                    if index >= len(raw):
                        raise ValueError("historical_json_incomplete")
                    if raw[index] == close:
                        index += 1
                        break
                    if raw[index] != ",":
                        raise ValueError("historical_json_separator_invalid")
                    index = whitespace(index + 1)
        else:
            value, index = decoder.raw_decode(raw, start)
        positions[path] = (start, index)
        return value, index

    value, end = visit(0, ())
    if whitespace(end) != len(raw):
        raise ValueError("historical_json_trailing_data")
    return value, positions


def _historical_source_map_call(call, envelope):
    """Authenticate stored lineage as cost evidence, without a scientific replay."""
    if envelope.get("repair_mapping_contract") not in (
        "capacity-source-map-v3",
        "capacity-source-map-v4",
    ):
        return
    from litagent.quality import digest as wire_digest

    keys = (
        "repair_mapping_contract",
        "source_map_identity",
        "source_relations_sha256",
        "source_relation_bindings_sha256",
        "source_relation_bindings",
    )
    canonical = call.get("reference_canonical")
    declared = isinstance(canonical, dict) and "source_relations" in canonical
    schema = call.get("reference_schema")
    source_schema = isinstance(schema, dict) and "source_relations" in schema.get(
        "properties", {}
    )
    parsed = "canonical_sha256" in call or "wire_bindings" in call
    source_raw = False
    if parsed:
        try:
            value = json.loads(call.get("raw", ""))
            source_raw = isinstance(value, dict) and "deleted_sources" in value
        except (TypeError, ValueError):
            # Body EOF compatibility does not create source-map metadata.
            pass
    if not declared:
        _mixed_require(
            not source_schema
            and not source_raw
            and not any(key in call for key in keys),
            "source_map_call_invalid",
        )
        return
    _mixed_require(parsed and source_schema and source_raw, "source_map_call_invalid")
    try:
        raw = call["raw"]
        wire, raw_positions = _historical_json_positions(raw)
        _mixed_require(
            set(wire) == {"outcome", "explanation", "patches", "deleted_sources"},
            "source_map_call_invalid",
        )
        derived = envelope["repair_mapping_contract"] == "capacity-source-map-v4"
        directory_sha, owners = None, {}
        if derived:
            directory = schema["x-source-owner-directory"]
            _mixed_require(
                schema["x-repair-mapping-contract"]
                == envelope["repair_mapping_contract"]
                and type(directory) is dict
                and bool(directory)
                and all(
                    type(tid) is str
                    and tid
                    and type(ids) is list
                    and ids
                    and all(type(uid) is str and uid for uid in ids)
                    for tid, ids in directory.items()
                ),
                "source_owner_directory_invalid",
            )
            ids = [uid for group in directory.values() for uid in group]
            _mixed_require(len(ids) == len(set(ids)), "source_owner_directory_invalid")
            relation_schema = schema["properties"]["source_relations"]["items"][
                "properties"
            ]
            source_ids = relation_schema["source_unit_ids"]["items"]["enum"]
            deleted_ids = schema["properties"]["deleted_sources"]["items"][
                "properties"
            ]["unit_id"]["enum"]
            target_ids = relation_schema["target_id"]["enum"]
            patch_targets = schema["properties"]["patches"]["items"]["properties"][
                "target_id"
            ]["enum"]
            _mixed_require(
                all(
                    type(group) is list
                    and all(type(item) is str for item in group)
                    and len(group) == len(set(group))
                    for group in (source_ids, deleted_ids, target_ids, patch_targets)
                )
                and set(source_ids) == set(deleted_ids) == set(ids)
                and set(target_ids) == set(patch_targets) == set(directory),
                "source_owner_directory_invalid",
            )
            owners = {uid: tid for tid, group in directory.items() for uid in group}
            directory_sha = wire_digest(directory)
            guidance = call["guidance_wire_record"]
            authorized = guidance["authorized_targets"]
            visible = guidance["payload"]["targets"]
            expected_directory = {}
            _mixed_require(
                type(authorized) is dict and type(visible) is dict,
                "source_owner_directory_invalid",
            )
            for tid, target in authorized.items():
                path = target["path"]
                end = target["end"]
                _mixed_require(
                    type(path) is list
                    and len(path) == 3
                    and type(path[0]) is str
                    and all(type(index) is int and index >= 0 for index in path[1:])
                    and type(end) is int
                    and end > path[2]
                    and type(target["original"]) is list
                    and len(target["original"]) == end - path[2]
                    and target["original_sha256"] == wire_digest(target["original"]),
                    "source_owner_authorized_target_invalid",
                )
                expected_directory[tid] = [
                    "writer:" + ":".join(map(str, (path[0], path[1], index)))
                    for index in range(path[2], end)
                ]
                _mixed_require(
                    all(
                        wire_digest(visible[tid][key]) == wire_digest(target[key])
                        for key in ("path", "end", "maximum")
                    )
                    and visible[tid]["original_unit_ids"] == expected_directory[tid],
                    "source_owner_authorized_target_invalid",
                )
            _mixed_require(
                set(visible) == set(authorized)
                and wire_digest(directory) == wire_digest(expected_directory)
                and guidance["payload_sha256"] == wire_digest(guidance["payload"])
                and guidance["identity"]
                == wire_digest(
                    {key: value for key, value in guidance.items() if key != "identity"}
                )
                and call["wire_bindings"]["repair_mapping_contract"]
                == envelope["repair_mapping_contract"]
                and call["wire_bindings"]["source_owner_directory_sha256"]
                == directory_sha,
                "source_owner_directory_invalid",
            )
        groups, relations, ordinals = {}, [], []
        for ordinal, patch in enumerate(wire["patches"]):
            if derived:
                _mixed_require(
                    type(patch) is dict
                    and set(patch)
                    in (
                        {"text", "fact_card_refs", "source_unit_ids"},
                        {"text", "body_unit_refs", "source_unit_ids"},
                    )
                    and type(patch["source_unit_ids"]) is list
                    and bool(patch["source_unit_ids"])
                    and all(
                        type(uid) is str and uid in owners
                        for uid in patch["source_unit_ids"]
                    )
                    and len(patch["source_unit_ids"])
                    == len(set(patch["source_unit_ids"]))
                    and len({owners[uid] for uid in patch["source_unit_ids"]}) == 1,
                    "source_owner_patch_invalid",
                )
                target = owners[patch["source_unit_ids"][0]]
            else:
                target = patch["target_id"]
            groups.setdefault(target, []).append((ordinal, patch))
        for tid, group in groups.items():
            for index, (ordinal, patch) in enumerate(group):
                relations.append(
                    {
                        "target_id": tid,
                        "replacement_index": index,
                        "source_unit_ids": patch["source_unit_ids"],
                    }
                )
                ordinals.append(ordinal)
        metadata = {
            "source_relations": relations,
            "deleted_sources": wire["deleted_sources"],
        }
        bindings = call["source_relation_bindings"]
        raw_sha = hashlib.sha256(raw.encode()).hexdigest()
        _mixed_require(
            set(bindings)
            == {
                "contract",
                "wire_identity",
                "raw_sha256",
                "raw_length",
                "host_schema_sha256",
                "canonical_map_sha256",
                "relations",
                "deleted_sources",
                "coverage_complete",
                "scientific_assessment",
            }
            | ({"source_owner_directory_sha256"} if derived else set())
            and bindings["coverage_complete"] is True
            and (
                not derived
                or bindings["source_owner_directory_sha256"] == directory_sha
            )
            and bindings["scientific_assessment"] == "not_verified"
            and call["repair_mapping_contract"] == envelope["repair_mapping_contract"]
            and call["source_map_identity"] == envelope["source_map_identity"]
            and call["source_relations_sha256"] == wire_digest(metadata)
            and call["source_relation_bindings_sha256"] == wire_digest(bindings)
            and bindings["contract"] == envelope["repair_mapping_contract"]
            and bindings["wire_identity"] == envelope["source_map_identity"]
            and bindings["raw_sha256"] == call["raw_sha256"] == raw_sha
            and bindings["raw_length"] == len(raw)
            and bindings["host_schema_sha256"]
            == call["wire_bindings"]["host_schema_sha256"]
            and bindings["canonical_map_sha256"] == wire_digest(metadata)
            and all(
                wire_digest({key: call[name][key] for key in metadata})
                == wire_digest(metadata)
                for name in ("reference_canonical", "received_canonical")
            )
            and len(bindings["relations"]) == len(relations)
            and (not derived or len(call["wire_bindings"]["units"]) == len(relations))
            and len(bindings["deleted_sources"]) == len(metadata["deleted_sources"]),
            "source_map_call_invalid",
        )
        for index, (relation, binding) in enumerate(
            zip(relations, bindings["relations"], strict=True)
        ):
            ordinal = ordinals[index]
            _mixed_require(
                wire_digest(binding["canonical_path"])
                == wire_digest(["source_relations", index])
                and type(binding["wire_ordinal"]) is int
                and binding["wire_ordinal"] == ordinal
                and type(binding["replacement_index"]) is int
                and binding["replacement_index"] == relation["replacement_index"]
                and binding["canonical_relation_sha256"] == wire_digest(relation)
                and (
                    wire_digest(binding["target_id"])
                    == wire_digest(
                        {
                            "derived": True,
                            "value": relation["target_id"],
                            "source_owner_directory_sha256": directory_sha,
                        }
                    )
                    if derived
                    else binding["target_id"]["wire_path"]
                    == ["patches", ordinal, "target_id"]
                )
                and binding["source_unit_ids"]["wire_path"]
                == ["patches", ordinal, "source_unit_ids"]
                and [b["wire_path"] for b in binding["source_ids"]]
                == [
                    ["patches", ordinal, "source_unit_ids", i]
                    for i in range(len(relation["source_unit_ids"]))
                ],
                "source_map_call_invalid",
            )
            if derived:
                unit = call["wire_bindings"]["units"][index]
                _mixed_require(
                    "target_id_span" not in unit
                    and unit["target_id"] == relation["target_id"]
                    and type(unit["wire_ordinal"]) is int
                    and unit["wire_ordinal"] == ordinal
                    and wire_digest(unit["target_id_derivation"])
                    == wire_digest(
                        {
                            "rule": "unique-source-owner-v1",
                            "source_owner_directory_sha256": directory_sha,
                            "source_relation_index": index,
                        }
                    ),
                    "source_owner_unit_binding_invalid",
                )
        for index, (deletion, binding) in enumerate(
            zip(metadata["deleted_sources"], bindings["deleted_sources"], strict=True)
        ):
            _mixed_require(
                wire_digest(binding["canonical_path"])
                == wire_digest(["deleted_sources", index])
                and binding["canonical_deletion_sha256"] == wire_digest(deletion)
                and binding["wire_binding"]["wire_path"] == ["deleted_sources", index]
                and binding["unit_id"]["wire_path"]
                == ["deleted_sources", index, "unit_id"]
                and binding["reason"]["wire_path"]
                == ["deleted_sources", index, "reason"],
                "source_map_call_invalid",
            )

        def check_spans(value):
            """Check recorded raw positions without loading the frozen parser."""
            if isinstance(value, list):
                for item in value:
                    check_spans(item)
            elif isinstance(value, dict):
                if "wire_path" in value:
                    path = value["wire_path"]
                    _mixed_require(
                        set(value)
                        == {"wire_path", "start", "end", "raw_sha256", "value_sha256"}
                        and isinstance(path, list)
                        and all(type(part) in (str, int) for part in path),
                        "source_map_span_invalid",
                    )
                    actual = wire
                    for part in path:
                        actual = actual[part]
                    start, end = value["start"], value["end"]
                    _mixed_require(
                        type(start) is int
                        and type(end) is int
                        and 0 <= start < end <= len(raw),
                        "source_map_span_invalid",
                    )
                    literal = raw[start:end]
                    _mixed_require(
                        raw_positions.get(tuple(path)) == (start, end)
                        and wire_digest(json.loads(literal)) == wire_digest(actual)
                        and hashlib.sha256(literal.encode()).hexdigest()
                        == value["raw_sha256"]
                        and wire_digest(actual) == value["value_sha256"],
                        "source_map_span_invalid",
                    )
                else:
                    for item in value.values():
                        check_spans(item)

        check_spans(bindings)
    except (KeyError, IndexError, TypeError, ValueError, AttributeError) as exc:
        raise ValueError("frozen_accounting_mixed_source_map_call_invalid") from exc


def _historical_call_binding(call, envelope):
    """Check only semantic fields produced at this recorded call's parse stage."""
    _historical_source_view_binding(call, envelope, call=True)
    _historical_support_binding(call, envelope)
    _historical_source_map_call(call, envelope)
    wire_fields = (
        "repair_guidance_wire_revision",
        "repair_guidance_wire_identity",
        "repair_guidance_capacity_rule",
        "repair_prompt_revision",
        "repair_prompt_identity",
    )
    wire_selected = "repair_guidance_wire_revision" in envelope
    wire_repair = wire_selected and call.get("stage") == "repair"
    if wire_selected or any(
        key in call for key in (*wire_fields, "guidance_wire_record")
    ):
        from litagent.quality import digest as wire_digest

        _mixed_require(
            all(
                (key in call) == (wire_repair and key in envelope)
                and (
                    not wire_repair or key not in envelope or call[key] == envelope[key]
                )
                for key in wire_fields
            )
            and ("guidance_wire_record" in call) == wire_repair,
            "call_guidance_wire_binding_invalid",
        )
        if wire_repair:
            record = call["guidance_wire_record"]
            _mixed_require(
                isinstance(record, dict)
                and record.get("contract") == "host-guidance-wire-receipt-v1"
                and all(
                    (key in record) == (key in envelope)
                    and record.get(key) == envelope.get(key)
                    for key in wire_fields
                )
                and record.get("identity")
                == wire_digest({k: v for k, v in record.items() if k != "identity"})
                and all(
                    key in record
                    and record.get(key + "_sha256") == wire_digest(record[key])
                    for key in ("canonical", "authority", "payload", "receipt")
                )
                and isinstance(record.get("final_payload_sha256"), str)
                and re.fullmatch(r"[0-9a-f]{64}", record["final_payload_sha256"])
                is not None,
                "call_guidance_wire_receipt_invalid",
            )
    if "support_validation_revision" not in envelope:
        return
    fields = ("reference_contract", "transport_revision", "transport_identity")
    parsed = "canonical_sha256" in call or "wire_bindings" in call
    _mixed_require(
        all(
            (key in call) == parsed and (not parsed or call[key] == envelope.get(key))
            for key in fields
        ),
        "call_semantic_binding_invalid",
    )


def _historical_writer_envelope(start, envelope, *, call=None):
    """Check the semantic fields actually projected into a selected writer wire."""
    _historical_source_view_request(start, envelope, call)
    if "support_validation_revision" not in envelope:
        return
    try:
        payload = json.loads(start["messages"][1]["content"])
        value = payload["writing_envelope"]
    except (KeyError, IndexError, TypeError, ValueError) as exc:
        raise ValueError("frozen_accounting_mixed_writer_envelope_invalid") from exc
    fields = (
        "contract",
        "inventory_recovery_contract",
        "reference_contract",
        "transport_revision",
        "support_validation_revision",
        "semantic_conflict_policy",
        "semantic_conflict_identity",
    )
    _mixed_require(
        isinstance(value, dict)
        and all(
            (key in value) == (key in envelope) and value.get(key) == envelope.get(key)
            for key in fields
        ),
        "writer_envelope_invalid",
    )
    if envelope.get("repair_mapping_contract") in (
        "capacity-source-map-v3",
        "capacity-source-map-v4",
    ):
        source_repair = payload.get("cause") == "capacity"
        _mixed_require(
            all(
                (key in value) == source_repair
                and (not source_repair or value[key] == envelope.get(key))
                for key in (
                    "repair_mapping_contract",
                    "source_map_identity",
                    "source_map_json_bytes",
                )
            ),
            "writer_source_map_envelope_invalid",
        )
    guidance_fields = (
        "repair_guidance_contract",
        "repair_guidance_identity",
        "repair_guidance_json_bytes",
    )
    task_view = (
        envelope.get("repair_guidance_wire_revision") == "host-repair-task-view-v1"
    )
    if "repair_guidance_contract" in envelope or any(
        key in value for key in guidance_fields
    ):
        repair_input = "current_body" in payload or "repair_input_contract" in payload
        selected = "repair_guidance_contract" in envelope and repair_input
        _mixed_require(
            (
                not repair_input
                or "current_body" in payload
                and payload.get("repair_input_contract") == "flat-repair-input-v1"
            )
            and all(
                (key in value)
                == (
                    selected and not (task_view and key == "repair_guidance_json_bytes")
                )
                and (
                    not selected
                    or task_view
                    and key == "repair_guidance_json_bytes"
                    or json.dumps(value[key], sort_keys=True, ensure_ascii=False)
                    == json.dumps(envelope.get(key), sort_keys=True, ensure_ascii=False)
                )
                for key in guidance_fields
            ),
            "writer_repair_guidance_envelope_invalid",
        )
    wire_fields = (
        "repair_guidance_wire_revision",
        "repair_guidance_wire_identity",
        "repair_guidance_capacity_rule",
        "repair_prompt_revision",
        "repair_prompt_identity",
    )
    if "repair_guidance_wire_revision" in envelope or any(
        key in value for key in wire_fields
    ):
        selected = (
            "repair_guidance_wire_revision" in envelope and "current_body" in payload
        )
        _mixed_require(
            all(
                (key in value) == (selected and key in envelope)
                and (not selected or key not in envelope or value[key] == envelope[key])
                for key in wire_fields
            ),
            "writer_repair_guidance_wire_envelope_invalid",
        )
        if task_view and selected:
            from litagent.quality import digest as wire_digest

            _mixed_require(
                wire_digest(payload.get("repair_feedback_limits"))
                == wire_digest(
                    {
                        "contract": envelope["repair_feedback_contract"],
                        "aggregate_rule": envelope["repair_guidance_capacity_rule"],
                        "maximum_request_reservation": envelope["request_caps"][
                            "repair"
                        ],
                        "reserved_output_tokens": 20000,
                    }
                ),
                "writer_repair_guidance_admission_rule_invalid",
            )
        if selected and call is not None:
            from litagent.quality import digest as wire_digest

            record = call.get("guidance_wire_record", {})
            _mixed_require(
                record.get("final_payload_sha256") == wire_digest(payload)
                and wire_digest(record.get("payload", {}).get("repair_guidance"))
                == wire_digest(payload.get("repair_guidance")),
                "writer_repair_guidance_wire_receipt_invalid",
            )
            if (
                envelope.get("repair_mapping_contract") == "capacity-source-map-v4"
                and payload.get("cause") == "capacity"
            ):
                _mixed_require(
                    wire_digest(record.get("payload", {}).get("targets"))
                    == wire_digest(payload.get("targets")),
                    "writer_source_owner_targets_invalid",
                )


def _historical_progress_envelope(record, ledger, request_id):
    """Bind the saved entry scope before this inventory's own allocation exists."""
    envelope = ledger["plan"].get("envelope", {})
    if "support_validation_revision" not in envelope:
        return
    index = list(ledger["entries"]).index(request_id)
    previous = [
        event["allocation"]
        for event in ledger.get("capacity_revisions", [])
        if len(event["entries"]) <= index
    ]
    expected = previous[-1]["envelope"] if previous else envelope
    _mixed_require(
        record["checkpoint"]["progress"].get("writer_envelope") == expected,
        "progress_envelope_invalid",
    )


def _historical_source_binding(root, manifest, envelope):
    """Bind selected historical semantics to frozen bytes without importing them."""
    if "support_validation_revision" not in envelope:
        return
    sources = manifest.get("source_hashes")
    _mixed_require(
        isinstance(sources, dict) and "src/litagent/eval/scoped.py" in sources,
        "source_binding_missing",
    )
    source_root = Path(root) / "source"
    _mixed_require(not source_root.is_symlink(), "source_binding_invalid")
    for name, expected in sources.items():
        _mixed_require(
            isinstance(name, str)
            and not Path(name).is_absolute()
            and ".." not in Path(name).parts
            and isinstance(expected, str)
            and re.fullmatch(r"[0-9a-f]{64}", expected),
            "source_binding_invalid",
        )
        path = source_root / name
        _mixed_require(
            path.resolve().is_relative_to(source_root.resolve())
            and not path.is_symlink()
            and path.is_file()
            and hashlib.sha256(path.read_bytes()).hexdigest() == expected,
            "source_binding_invalid",
        )


def conservative_rejection_receipt(root, manifest, state, row, archive, starts, ends):
    """Carry a complete uncertain charge; never fabricate a received usage record."""
    from litagent.observability.context import reset_task_id, set_task_id

    if (
        state.get("status") != "stopped"
        or row.get("publishable") is not False
        or manifest.get("version") != VERSION
        or manifest.get("limits") != LIMITS
        or archive.get("config") != manifest.get("config")
        or archive.get("config_fingerprint")
        != manifest.get("config", {}).get("fingerprint")
        or any(e["event"] == "llm.retry" for e in archive["events"])
    ):
        raise ValueError("frozen_accounting_unknown_proof_missing")
    Budget, reservation_counter, budget_sha = historical_accounting_budget(
        root, manifest
    )
    path = Path("artifacts/targeted-repairs", row["run_id"] + ".json.writer-v2.json")
    journal = json.loads(path.read_text())
    reconstructed = Budget(max_tokens=LIMITS["per_case_tokens"])
    reconstructed.configure_strict(halt_on_unknown_usage=True)
    total_unknown, rejection_calls = 0, []
    indexed = {e["operation_id"]: e for e in ends}
    expected_model = manifest.get("config", {}).get("effective", {}).get("llm", {})
    routes = expected_model.get("model_routing", {})
    for start in starts:
        end = indexed[start["operation_id"]]
        cap = start.get("max_tokens")
        if type(cap) is not int or not 0 < cap <= 20000:
            raise ValueError("frozen_accounting_cap_invalid")
        reservation = reservation_counter(
            start.get("messages"), cap, start.get("response_format")
        )
        if not isinstance(start.get("task_id"), str) or not start["task_id"]:
            raise ValueError("frozen_accounting_task_invalid")
        context = set_task_id(start["task_id"])
        reserved = False
        try:
            reconstructed.reserve(reservation)
            reserved = True
            reconstructed.begin_request()
            if end.get("usage_available") is True:
                reconstructed.record(end)
            else:
                call, receipt, request, response = bound_native_receipt(
                    start, end, journal, reservation_counter=reservation_counter
                )
                code = receipt["status_code"]
                route = (
                    "evaluation" if call.get("stage") == "reviewer" else "generation"
                )
                if (
                    not 400 <= code < 500
                    or not isinstance(response.get("error"), dict)
                    or "usage" in response
                    or response.get("output")
                    or call.get("status") != "failed_no_replay"
                    or call.get("usage", {}).get("usage_available") is not False
                    or end.get("provider_error") != "responses_http_" + str(code)
                    or request.get("model")
                    != routes.get(route, expected_model.get("model"))
                    or receipt["endpoint"] != "/api/v3/responses"
                    or (
                        receipt.get("origin") is not None
                        and receipt["origin"] != expected_model.get("origin")
                    )
                    or cap
                    != (
                        manifest["config"]["effective"]
                        .get("adversarial", {})
                        .get("review_max_tokens")
                        if route == "evaluation"
                        else expected_model.get("max_tokens")
                    )
                    or call.get("thinking_mode") != "enabled"
                    or call.get("reasoning_effort") != "low"
                ):
                    raise ValueError("frozen_accounting_unknown_not_terminal_rejection")
                reconstructed.charge_unknown_usage(reservation)
                total_unknown += reservation
                rejection_calls.append(
                    {
                        "operation_id": start["operation_id"],
                        "reservation": reservation,
                        "request_sha256": receipt["request_sha256"],
                        "response_sha256": receipt["response_sha256"],
                        "http_status": code,
                        "origin_evidence": (
                            "recorded_http_origin"
                            if receipt.get("origin")
                            else "frozen_config_origin_only"
                        ),
                    }
                )
        finally:
            reset_task_id(context)
            # Every successful reserve is released even when proof validation fails.
            if reserved:
                reconstructed.release(reservation)
    snapshot = reconstructed.snapshot()
    if (
        type(row.get("unknown_hold")) is not int
        or row["unknown_hold"] != total_unknown
        or total_unknown <= 0
        or snapshot["used_tokens"] != row["known_tokens"] + total_unknown
        or snapshot["requests_sent"] != row["requests"]
        or row.get("usage_complete") is not False
    ):
        raise ValueError("frozen_accounting_unknown_amount_invalid")
    actual = row.get("budget_snapshot")
    if actual is not None:
        restored = Budget.from_snapshot(actual)
        for key in (
            "max_tokens",
            "used_tokens",
            "requests_sent",
            "phase_usage",
            "usage_uncertain",
            "call_count",
        ):
            if restored.snapshot().get(key) != snapshot.get(key):
                raise ValueError("frozen_accounting_budget_snapshot_invalid")
    return {
        "version": CONSERVATIVE_ACCOUNTING,
        "unknown_hold": total_unknown,
        "journal_sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
        "budget_evidence": (
            "persisted_snapshot"
            if actual is not None
            else "reconstructed_from_frozen_records"
        ),
        "budget_source_sha256": budget_sha,
        "conservative_snapshot": snapshot,
        "rejections": rejection_calls,
    }


def planned_writer_unknown_receipt(root, manifest, state, row, archive, starts, ends):
    """Authenticate sent writer escrows using data, without replay or frozen code."""
    from litagent.llm.responses import RESPONSES_CONTRACT
    from litagent.quality import digest as writer_digest
    from litagent.safety.budget import strict_request_reservation

    def require(condition, reason):
        if not condition:
            raise ValueError("frozen_accounting_planned_" + reason)

    run_id = row["run_id"]
    ledger_path = Path(
        "artifacts/targeted-repairs", run_id + ".json.budget-ledger.json"
    )
    journal_path = Path("artifacts/targeted-repairs", run_id + ".json.writer-v2.json")
    ledger, journal = json.loads(ledger_path.read_text()), json.loads(
        journal_path.read_text()
    )
    require(ledger == row["budget_snapshot"], "persisted_snapshot_conflict")
    require(ledger["plan"] == manifest["budget_plans"][row["case_id"]], "plan_conflict")
    hold = historical_unknown_token_hold(
        ledger,
        row["known_tokens"],
        expected_plan=manifest["budget_plans"][row["case_id"]],
    )
    _historical_source_binding(root, manifest, ledger["plan"].get("envelope", {}))
    _historical_support_binding(journal, ledger["plan"].get("envelope", {}))
    _historical_guidance_wire_owner(journal, ledger["plan"].get("envelope", {}))
    _historical_source_view_binding(journal, ledger["plan"].get("envelope", {}))
    _historical_source_view_archive(
        journal, ledger["plan"].get("envelope", {}), archive
    )
    require(
        hold > 0
        and type(row.get("unknown_hold")) is int
        and row["unknown_hold"] == hold
        and row.get("usage_complete") is False
        and row.get("publishable") is False,
        "unknown_amount_invalid",
    )
    require(
        journal["contract"] == manifest["contract"]
        and journal["prompt_contract"] == manifest["writer_prompt_contract"],
        "journal_contract_conflict",
    )
    sent = {k: v for k, v in ledger["entries"].items() if v["sent"]}
    failed = [e["data"] for e in archive["events"] if e["event"] == "llm.failed"]
    terminals = ends + failed
    require(
        type(row.get("requests")) is int
        and row["requests"] == ledger["requests_sent"] == len(sent) == len(starts)
        and len(terminals) == len(starts)
        and len({x["operation_id"] for x in starts}) == len(starts)
        and len({x["operation_id"] for x in terminals}) == len(terminals)
        and {x["operation_id"] for x in starts}
        == {x["operation_id"] for x in terminals},
        "operation_set_invalid",
    )
    terminal_by_id = {x["operation_id"]: x for x in terminals}
    completed_ids = {x["operation_id"] for x in ends}
    models = manifest["config"]["effective"]["llm"]
    routes = models.get("model_routing", {})
    used, known, calls = set(), 0, []
    for start in starts:
        matches = []
        for name, call in journal["calls"].items():
            _historical_call_binding(call, ledger["plan"].get("envelope", {}))
            request_id = journal["identity"] + ":" + name
            if request_id not in sent:
                continue
            api = call.get("provider_api", "chat")
            mode, cap = call.get("thinking_mode"), call.get("max_tokens")
            limit = call.get("output_limit_parameter")
            if (
                api not in {"chat", "responses"}
                or mode not in {"provider_default", "enabled", "disabled"}
                or type(cap) is not int
                or cap <= 0
                or limit
                != (
                    "max_output_tokens"
                    if api == "responses"
                    else "max_tokens" if mode == "disabled" else "max_completion_tokens"
                )
            ):
                continue
            extra = None if mode == "provider_default" else {"thinking": {"type": mode}}
            rf = start.get("response_format")
            identity = [
                start["messages"],
                rf,
                cap if api == "chat" and mode == "disabled" else {limit: cap},
                extra,
            ]
            effort = call.get("reasoning_effort")
            if effort is not None:
                require(effort == "low" and mode == "enabled", "thinking_invalid")
                identity.append({"reasoning_effort": effort})
            if api == "responses":
                identity.append([api, RESPONSES_CONTRACT])
            if writer_digest(identity) != call.get("input_sha256"):
                continue
            _historical_writer_envelope(
                start, ledger["plan"].get("envelope", {}), call=call
            )
            entry = sent.get(request_id)
            require(entry is not None and request_id not in used, "request_id_invalid")
            expected_stage = (
                "body_initial"
                if name == "body:introduction"
                and "body_initial" in ledger["plan"]["stages"]
                else (
                    "body"
                    if name.startswith("body:")
                    else (
                        "reviewer"
                        if name == "reviewer"
                        else "repair" if name.startswith("repair_") else "summary"
                    )
                )
            )
            require(entry["stage"] == expected_stage, "stage_conflict")
            require(
                cap == ledger["plan"]["stages"][expected_stage]["output_tokens"],
                "output_plan_conflict",
            )
            route = "evaluation" if expected_stage == "reviewer" else "generation"
            require(
                start["model"] == routes.get(route, models.get("model"))
                and start.get("provider_api", "chat") == api
                and start["max_tokens"] == cap
                and start["output_limit_parameter"] == limit
                and start.get("reasoning_effort") == effort,
                "request_options_conflict",
            )
            require(
                rf["type"] == "json_schema"
                and writer_digest(rf["json_schema"]["schema"])
                == call["provider_schema_sha256"],
                "schema_conflict",
            )
            kwargs = {"response_format": rf, limit: cap, "model": start["model"]}
            if extra is not None:
                kwargs["extra_body"] = extra
            if effort is not None:
                kwargs["reasoning_effort"] = effort
            if api == "responses":
                kwargs["provider_api"] = api
            reservation = strict_request_reservation(start["messages"], cap, rf)
            require(
                entry["reserved_tokens"] == call["reservation"] == reservation
                and entry["wire_sha256"]
                == writer_digest([start["messages"], kwargs, api]),
                "wire_conflict",
            )
            matches.append((request_id, entry, reservation))
        require(len(matches) == 1, "writer_binding_ambiguous_or_missing")
        request_id, entry, reservation = matches[0]
        used.add(request_id)
        end = terminal_by_id[start["operation_id"]]
        complete = start["operation_id"] in completed_ids
        if complete and end.get("usage_available") is True:
            usage = {k: end[k] for k in ("prompt_tokens", "completion_tokens")}
            require(
                all(type(v) is int and v >= 0 for v in usage.values())
                and type(end.get("total_tokens")) is int
                and end["total_tokens"] == sum(usage.values())
                and entry["status"] == "settled"
                and entry["actual_tokens"] == end["total_tokens"]
                and entry["usage_sha256"] == writer_digest(usage),
                "known_usage_conflict",
            )
            known += end["total_tokens"]
        else:
            require(
                end.get("usage_available") is False
                and entry["status"] == "interrupted_no_replay"
                and entry["actual_tokens"] is None,
                "unknown_status_conflict",
            )
        calls.append(
            {
                "operation_id": start["operation_id"],
                "request_id": request_id,
                "wire_sha256": entry["wire_sha256"],
                "reservation": reservation,
                "actual_tokens": entry["actual_tokens"],
                "status": entry["status"],
            }
        )
    require(used == set(sent) and known == row["known_tokens"], "usage_set_conflict")
    return {
        "version": PLANNED_UNKNOWN_ACCOUNTING,
        "unknown_hold": hold,
        "plan_identity": ledger["plan"]["identity"],
        "ledger_sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
        "journal_sha256": hashlib.sha256(journal_path.read_bytes()).hexdigest(),
        "requests": calls,
    }


def _mixed_require(condition, reason):
    """Reject incomplete cost evidence without changing durable accounting."""
    if not condition:
        raise ValueError("frozen_accounting_mixed_" + reason)


def _mixed_snapshot_prefix(previous, final, *, expected_plan):
    """Validate a checkpoint without normalizing its pending entries in place."""
    from litagent.safety.planned_accounting import validate_accounting_snapshot

    validate_accounting_snapshot(previous, expected_plan=expected_plan)
    validate_accounting_snapshot(final, expected_plan=expected_plan)
    _mixed_require(
        all(
            previous.get(key) == final.get(key)
            for key in ("schema_version", "strict_policy", "plan", "max_tokens")
        )
        and list(previous["entries"])
        == list(final["entries"])[: len(previous["entries"])]
        and previous.get("capacity_revisions", [])
        == final.get("capacity_revisions", [])[
            : len(previous.get("capacity_revisions", []))
        ]
        and set(previous["closed_stages"]).issubset(final["closed_stages"])
        and all(
            previous[key] <= final[key]
            for key in ("used_tokens", "call_count", "requests_sent")
        )
        and (not previous["usage_uncertain"] or final["usage_uncertain"])
        and (
            not previous["stop_reason"]
            or previous["stop_reason"] == final["stop_reason"]
        ),
        "progress_budget_prefix_invalid",
    )
    revisions = previous.get("capacity_revisions", [])
    _mixed_require(
        previous.get("capacity_allocation")
        == (revisions[-1]["allocation"] if revisions else None),
        "progress_allocation_invalid",
    )
    for key, old in previous["entries"].items():
        new = final["entries"][key]
        _mixed_require(
            all(
                old.get(field) == new.get(field)
                for field in (
                    "stage",
                    "reserved_tokens",
                    "round_number",
                    "round_wire_sha256",
                    "inventory_recovery",
                )
            )
            and (
                old.get("wire_sha256") is None
                or old["wire_sha256"] == new.get("wire_sha256")
            )
            and (not old["sent"] or new["sent"])
            and (old["status"] != "settled" or old == new)
            and (
                old["status"] != "interrupted_no_replay"
                or new["status"] == "interrupted_no_replay"
            ),
            "progress_entry_regressed",
        )


def evaluation_progress_records(run_id, archive):
    """Locate only checkpoints named by this run's actual archive events."""
    _mixed_require(
        archive.get("run_id") == archive.get("session_id") == run_id,
        "progress_run_invalid",
    )
    latest, owners = {}, {}
    for event in archive["events"]:
        if event["event"] != "evaluation.batch":
            continue
        value = event["data"]
        identity = value.get("identity")
        _mixed_require(
            isinstance(identity, str) and re.fullmatch(r"[0-9a-f]{64}", identity),
            "progress_locator_invalid",
        )
        latest[identity] = value
        owners.setdefault(identity, []).append(value.get("session_id"))
    records = []
    for identity, event in latest.items():
        path = Path("artifacts/evaluations", identity, "progress.json")
        raw = path.read_bytes()
        checkpoint = json.loads(raw)
        _mixed_require(
            checkpoint.get("identity") == identity
            and checkpoint.get("started") is True
            and isinstance(checkpoint.get("phase"), str)
            and checkpoint["phase"] == event.get("phase")
            and checkpoint.get("budget") == event.get("budget")
            and isinstance(checkpoint.get("progress"), dict),
            "progress_locator_conflict",
        )
        records.append(
            {
                "identity": identity,
                "path": str(path),
                "sha256": hashlib.sha256(raw).hexdigest(),
                "checkpoint": checkpoint,
                "event_session_ids": owners[identity],
            }
        )
    return records


def _mixed_progress_context(record, ledger, manifest, row, archive, journal):
    """Bind checkpoint provenance without requiring a committed final report."""
    from litagent.quality import digest as wire_digest

    checkpoint = record["checkpoint"]
    pinned_plan = manifest["budget_plans"][row["case_id"]]
    _mixed_snapshot_prefix(checkpoint["budget"], ledger, expected_plan=pinned_plan)
    envelope = pinned_plan.get("envelope", {})
    _historical_support_binding(checkpoint["progress"], envelope)
    if "support_validation_revision" in envelope:
        _historical_support_binding(
            checkpoint["progress"].get("writer_envelope"), envelope
        )
    cases = [value for value in manifest["cases"] if value["case_id"] == row["case_id"]]
    _mixed_require(len(cases) == 1, "progress_case_invalid")
    case = cases[0]
    report = archive.get("report") or {}
    contexts = [report.get("metadata", {})]
    candidates = []
    if isinstance(report.get("survey"), str):
        candidates.append(report["survey"])
    for node in archive.get("nodes", {}).values():
        value = node.get("output")
        if isinstance(value, dict) and isinstance(value.get("writer_document"), dict):
            contexts.append(value)
            candidates.extend(
                value[key]
                for key in ("draft", "final_draft")
                if isinstance(value.get(key), str)
            )
    for value in journal.get("candidates", []):
        survey = value.get("survey")
        _mixed_require(
            isinstance(survey, str)
            and hashlib.sha256(survey.encode()).hexdigest()
            == value.get("report_sha256"),
            "candidate_report_invalid",
        )
        candidates.append(survey)
    _mixed_require(
        checkpoint.get("config_sha256") == wire_digest(manifest["config"])
        and checkpoint.get("coverage_requirements_sha256")
        == wire_digest(case["bundle"]["requirements"])
        and any(
            isinstance(value.get("fact_cards"), dict)
            and value.get("evidence_snapshot_sha256")
            == checkpoint.get("evidence_snapshot_sha256")
            == wire_digest(value["fact_cards"])
            for value in contexts
        )
        and isinstance(checkpoint.get("report_sha256"), str)
        and re.fullmatch(r"[0-9a-f]{64}", checkpoint["report_sha256"]),
        "progress_context_invalid",
    )
    survey = next(
        (
            value
            for value in candidates
            if hashlib.sha256(value.encode()).hexdigest() == checkpoint["report_sha256"]
        ),
        None,
    )
    owner_values = [checkpoint.get("session_id"), *record["event_session_ids"]]
    owner_required = manifest.get(
        "accounting_policy"
    ) == PLANNED_MIXED_UNKNOWN_ACCOUNTING or any(
        value is not None for value in owner_values
    )
    owner_verified = all(value == row["run_id"] for value in owner_values)
    _mixed_require(not owner_required or owner_verified, "progress_owner_invalid")
    record["owner_verified"] = owner_verified
    recovery = checkpoint["progress"].get("inventory_recovery")
    if recovery is None and survey is not None:
        _mixed_require(
            record["identity"]
            == wire_digest([row["run_id"], survey, checkpoint["phase"]]),
            "progress_run_identity_invalid",
        )
        record["normal_identity_preimage"] = "verified"
    else:
        # Interrupted, uncommitted candidates need not have a stored full survey.
        # Their report identity must still close through each actual wire below.
        record["normal_identity_preimage"] = "not_available_cost_binding_only"
    repair_path = Path("artifacts/targeted-repairs", row["run_id"] + ".json")
    if recovery is not None:
        from litagent.quality import targeted_repair_state

        raw = repair_path.read_bytes()
        repair = json.loads(raw)
        repair_state = targeted_repair_state(
            repair["state"], row["run_id"], allow_inflight=True
        )
        _mixed_require(
            isinstance(recovery, dict)
            and set(recovery)
            == {
                "contract",
                "source_report_sha256",
                "previous_inventory_sha256",
                "bindings_sha256",
                "attempt_index",
            }
            and type(recovery.get("attempt_index")) is int,
            "recovery_progress_invalid",
        )
        record["repair_state"] = repair_state
        record["repair_checkpoint_sha256"] = hashlib.sha256(raw).hexdigest()
        artifact = repair.get(
            "inventory_recovery_" + str(recovery["attempt_index"]), {}
        )
        feedback = artifact.get("feedback")
        if feedback is not None:
            _mixed_require(
                all(feedback.get(key) == value for key, value in recovery.items()),
                "recovery_feedback_invalid",
            )
            record["canonical_recovery_feedback"] = feedback
            if survey is not None:
                expected = wire_digest(
                    [
                        wire_digest([row["run_id"], survey, checkpoint["phase"]]),
                        feedback,
                    ]
                )
                _mixed_require(
                    expected == record["identity"], "recovery_progress_identity_invalid"
                )
                record["normal_identity_preimage"] = (
                    "verified_with_canonical_recovery_feedback"
                )
    _mixed_require(
        owner_verified
        or record["normal_identity_preimage"]
        in {"verified", "verified_with_canonical_recovery_feedback"},
        "progress_owner_preimage_missing",
    )
    inputs = [
        value.get("input_sha256")
        for value in checkpoint["progress"].get("requests", [])
    ]
    _mixed_require(
        len(inputs) == len(set(inputs))
        and all(
            isinstance(value, str) and re.fullmatch(r"[0-9a-f]{64}", value)
            for value in inputs
        ),
        "progress_request_identity_invalid",
    )


def _mixed_writer_start(start, ledger, manifest, journal):
    """Authenticate a writer call using the unchanged journal input contracts."""
    from litagent.llm.responses import RESPONSES_CONTRACT
    from litagent.quality import digest as wire_digest
    from litagent.safety.budget import strict_request_reservation

    matches = []
    models = manifest["config"]["effective"]["llm"]
    routes = models.get("model_routing", {})
    for name, call in journal["calls"].items():
        _historical_call_binding(call, ledger["plan"].get("envelope", {}))
        request_id = journal["identity"] + ":" + name
        entry = ledger["entries"].get(request_id)
        if not entry or not entry["sent"]:
            continue
        api, mode, cap = (
            call.get("provider_api", "chat"),
            call.get("thinking_mode"),
            call.get("max_tokens"),
        )
        if (
            api not in {"chat", "responses"}
            or mode not in {"provider_default", "enabled", "disabled"}
            or type(cap) is not int
            or cap <= 0
        ):
            continue
        limit = (
            "max_output_tokens"
            if api == "responses"
            else "max_tokens" if mode == "disabled" else "max_completion_tokens"
        )
        if call.get("output_limit_parameter") != limit:
            continue
        extra = None if mode == "provider_default" else {"thinking": {"type": mode}}
        rf, effort = start.get("response_format"), call.get("reasoning_effort")
        identity = [
            start["messages"],
            rf,
            cap if api == "chat" and mode == "disabled" else {limit: cap},
            extra,
        ]
        if effort is not None:
            identity.append({"reasoning_effort": effort})
        if api == "responses":
            identity.append([api, RESPONSES_CONTRACT])
        if wire_digest(identity) != call.get("input_sha256"):
            continue
        _historical_writer_envelope(
            start, ledger["plan"].get("envelope", {}), call=call
        )
        stage = (
            "body_initial"
            if name == "body:introduction"
            and "body_initial" in ledger["plan"]["stages"]
            else (
                "body"
                if name.startswith("body:")
                else (
                    "reviewer"
                    if name == "reviewer"
                    else "repair" if name.startswith("repair_") else "summary"
                )
            )
        )
        route = "evaluation" if stage == "reviewer" else "generation"
        _mixed_require(
            entry["stage"] == stage
            and cap == ledger["plan"]["stages"][stage]["output_tokens"]
            and start["model"] == routes.get(route, models.get("model"))
            and start.get("provider_api", "chat") == api
            and start.get("max_tokens") == cap
            and start.get("output_limit_parameter") == limit
            and start.get("reasoning_effort") == effort
            and (effort is None or effort == "low" and mode == "enabled")
            and isinstance(rf, dict)
            and rf.get("type") == "json_schema"
            and wire_digest(rf["json_schema"]["schema"])
            == call["provider_schema_sha256"],
            "writer_options_invalid",
        )
        kwargs = {"response_format": rf, limit: cap, "model": start["model"]}
        if extra is not None:
            kwargs["extra_body"] = extra
        if effort is not None:
            kwargs["reasoning_effort"] = effort
        if api == "responses":
            kwargs["provider_api"] = api
        reservation = strict_request_reservation(start["messages"], cap, rf)
        _mixed_require(
            entry["reserved_tokens"] == call["reservation"] == reservation
            and entry.get("wire_sha256")
            == wire_digest([start["messages"], kwargs, api]),
            "writer_wire_invalid",
        )
        matches.append(
            {
                "request_id": request_id,
                "operation_id": start["operation_id"],
                "stage": stage,
                "phase": stage,
                "input_sha256": call["input_sha256"],
                "wire_sha256": entry["wire_sha256"],
                "reservation": reservation,
                "progress_identity": None,
            }
        )
    _mixed_require(len(matches) <= 1, "writer_binding_ambiguous")
    return matches[0] if matches else None


def _planned_evaluation_occurrence_bindings(archive, starts, records, ledger, manifest):
    """Bind repeated Evaluation wires through complete serial checkpoint ranges."""
    from litagent.eval.scoped import _request_identity
    from litagent.quality import digest as wire_digest

    try:
        events = archive["events"]
        _mixed_require(
            isinstance(events, list)
            and all(
                isinstance(event, dict)
                and type(event.get("sequence")) is int
                and event["sequence"] == index
                for index, event in enumerate(events)
            ),
            "evaluation_occurrence_sequence_invalid",
        )
        archived_starts = [
            (index, event["data"])
            for index, event in enumerate(events)
            if event["event"] == "llm.start"
        ]
        _mixed_require(
            wire_digest([value for _, value in archived_starts]) == wire_digest(starts),
            "evaluation_occurrence_sequence_invalid",
        )
        terminal_positions = {}
        for index, event in enumerate(events):
            if event["event"] not in {"llm.complete", "llm.failed"}:
                continue
            operation = event["data"]["operation_id"]
            _mixed_require(
                operation not in terminal_positions,
                "evaluation_occurrence_sequence_invalid",
            )
            terminal_positions[operation] = index
        inputs = [
            _request_identity(value["messages"], value.get("response_format"))
            for _, value in archived_starts
        ]
        sent = [key for key, value in ledger["entries"].items() if value["sent"]]
        result, previous_end = {}, -1
        for record in records:
            checkpoint = record["checkpoint"]
            requests = checkpoint["progress"].get("requests", [])
            located = [
                (index, event["data"])
                for index, event in enumerate(events)
                if event["event"] == "evaluation.batch"
                and event["data"].get("identity") == record["identity"]
            ]
            _mixed_require(
                isinstance(requests, list) and bool(located),
                "evaluation_occurrence_anchor_invalid",
            )
            first_index, first = located[0]
            last_index, last = located[-1]
            _mixed_require(
                first_index > previous_end
                and all(
                    value.get("phase") == checkpoint["phase"] for _, value in located
                )
                and wire_digest(last["budget"]) == wire_digest(checkpoint["budget"]),
                "evaluation_occurrence_anchor_invalid",
            )
            previous_end = last_index
            _mixed_require(
                any(
                    wire_digest(plan) == wire_digest(ledger["plan"])
                    for plan in manifest["budget_plans"].values()
                ),
                "evaluation_occurrence_plan_invalid",
            )
            _mixed_snapshot_prefix(
                first["budget"], ledger, expected_plan=ledger["plan"]
            )
            _mixed_snapshot_prefix(
                checkpoint["budget"], ledger, expected_plan=ledger["plan"]
            )
            before, after = (
                first["budget"]["requests_sent"],
                checkpoint["budget"]["requests_sent"],
            )
            _mixed_require(
                type(before) is int
                and type(after) is int
                and 0 <= before <= after <= len(starts)
                and after - before == len(requests)
                and sum(index < first_index for index, _ in archived_starts) == before
                and sum(index < last_index for index, _ in archived_starts) == after
                and [
                    key
                    for key, value in first["budget"]["entries"].items()
                    if value["sent"]
                ]
                == sent[:before]
                and [
                    key
                    for key, value in checkpoint["budget"]["entries"].items()
                    if value["sent"]
                ]
                == sent[:after],
                "evaluation_occurrence_prefix_invalid",
            )
            if not requests:
                continue
            phases = [request["phase"] for request in requests]
            _mixed_require(
                phases
                == ["inventory"]
                + ["assessment"] * phases.count("assessment")
                + (["global"] if phases.count("global") == 1 else [])
                and inputs.count(requests[0]["input_sha256"]) == 1,
                "evaluation_occurrence_anchor_invalid",
            )
            previous_terminal = first_index
            _mixed_require(
                not any(
                    index < first_index
                    and terminal_positions[value["operation_id"]] >= first_index
                    for index, value in archived_starts
                ),
                "evaluation_occurrence_overlap_invalid",
            )
            for index, request in enumerate(requests):
                ordinal = before + index
                event_index, start = archived_starts[ordinal]
                operation = start["operation_id"]
                terminal_index = terminal_positions[operation]
                _mixed_require(
                    previous_terminal < event_index < terminal_index < last_index
                    and first_index < event_index
                    and inputs[ordinal] == request["input_sha256"]
                    and operation not in result,
                    "evaluation_occurrence_sequence_invalid",
                )
                previous_terminal = terminal_index
                request_id = sent[ordinal]
                entry = ledger["entries"][request_id]
                result[operation] = {
                    "request_id": request_id,
                    "operation_id": operation,
                    "progress_identity": record["identity"],
                    "progress_index": index,
                    "phase": request["phase"],
                    "round_number": entry.get("round_number"),
                    "input_sha256": request["input_sha256"],
                    "wire_sha256": entry.get("wire_sha256"),
                }
        return result
    except (KeyError, IndexError, TypeError, AttributeError, StopIteration) as exc:
        raise ValueError(
            "frozen_accounting_mixed_evaluation_occurrence_invalid"
        ) from exc


def authenticate_planned_evaluation_start(
    start, record, ledger, manifest, *, expected_binding=None
):
    """Bind an archived Evaluation wire to one checkpoint and one sent escrow."""
    from litagent.eval.scoped import _request_identity
    from litagent.quality import digest as wire_digest
    from litagent.safety.budget import strict_request_reservation

    checkpoint = record["checkpoint"]
    rf = start.get("response_format")
    input_sha = _request_identity(start["messages"], rf)
    matches = [
        value
        for value in checkpoint["progress"].get("requests", [])
        if value.get("input_sha256") == input_sha
    ]
    _mixed_require(len(matches) <= 1, "evaluation_progress_ambiguous")
    if not matches:
        _mixed_require(
            expected_binding is None, "evaluation_occurrence_binding_invalid"
        )
        return None
    progress = matches[0]
    expected_request_id = None
    if expected_binding is not None:
        _mixed_require(
            isinstance(expected_binding, dict), "evaluation_occurrence_binding_invalid"
        )
        requests = checkpoint["progress"].get("requests", [])
        sent = [key for key, value in ledger["entries"].items() if value["sent"]]
        index = expected_binding.get("progress_index")
        final_count = checkpoint["budget"].get("requests_sent")
        _mixed_require(
            set(expected_binding)
            == {
                "request_id",
                "operation_id",
                "progress_identity",
                "progress_index",
                "phase",
                "round_number",
                "input_sha256",
                "wire_sha256",
            }
            and type(index) is int
            and 0 <= index < len(requests)
            and type(final_count) is int
            and len(requests) <= final_count <= len(sent)
            and wire_digest(requests[index]) == wire_digest(progress)
            and expected_binding["operation_id"] == start["operation_id"]
            and expected_binding["progress_identity"] == record["identity"]
            and expected_binding["input_sha256"] == input_sha
            and expected_binding["phase"] == progress.get("phase")
            and expected_binding["request_id"]
            == sent[final_count - len(requests) + index],
            "evaluation_occurrence_binding_invalid",
        )
        expected_request_id = expected_binding["request_id"]
    config = manifest["config"]["effective"]
    settings, models = config["eval"], config["llm"]
    cap, phase = settings["max_tokens"], progress.get("phase")
    model = models.get("model_routing", {}).get("evaluation", models.get("model"))
    _mixed_require(
        phase in {"inventory", "assessment", "global"}
        and type(cap) is int
        and cap == 16384
        and start.get("provider_api", "chat") == "chat"
        and start.get("output_limit_parameter") == "max_tokens"
        and start.get("max_tokens") == cap
        and start.get("temperature") == 0
        and start.get("model") == model
        and start.get("reasoning_effort") is None
        and isinstance(rf, dict)
        and rf.get("type") == "json_schema",
        "evaluation_options_invalid",
    )
    kwargs = {
        "response_format": rf,
        "max_tokens": cap,
        "temperature": 0,
        "model": model,
    }
    if settings["thinking_mode"] != "provider_default":
        kwargs["extra_body"] = {"thinking": {"type": settings["thinking_mode"]}}
    wire_sha = wire_digest([start["messages"], kwargs, "chat"])
    entries = [
        (key, value)
        for key, value in ledger["entries"].items()
        if value["sent"]
        and value.get("wire_sha256") == wire_sha
        and (expected_request_id is None or key == expected_request_id)
    ]
    _mixed_require(len(entries) == 1, "evaluation_wire_ambiguous_or_missing")
    request_id, entry = entries[0]
    if expected_binding is not None:
        _mixed_require(
            expected_binding["wire_sha256"] == wire_sha
            and (
                expected_binding["round_number"] is None
                or type(expected_binding["round_number"]) is int
            )
            and expected_binding["round_number"] == entry.get("round_number"),
            "evaluation_occurrence_binding_invalid",
        )
    amount = strict_request_reservation(start["messages"], cap, rf)
    _mixed_require(
        type(progress.get("reserved_tokens")) is int
        and progress["reserved_tokens"] == amount == entry["reserved_tokens"],
        "evaluation_reservation_invalid",
    )
    round_wire = wire_digest([start["messages"], cap, rf])
    if phase in {"assessment", "global"}:
        allocations = [
            event["allocation"]
            for event in ledger.get("capacity_revisions", [])
            if event["allocation"].get("round", {}).get("number")
            == entry.get("round_number")
        ]
        _mixed_require(
            entry["stage"] == phase and len(allocations) == 1,
            "evaluation_round_missing",
        )
        allocation = allocations[0]
        round_ = allocation["round"]
        stage_entries = [
            key for key, value in ledger["entries"].items() if value["stage"] == phase
        ]
        index = stage_entries.index(request_id) - round_["offsets"][phase]
        slots = [value for value in round_["requests"] if value["stage"] == phase]
        expected = {
            "stage": phase,
            "wire_sha256": round_wire,
            "reserved_tokens": amount,
            "output_tokens": cap,
        }
        _mixed_require(
            round_["mode"] == "assess"
            and round_["report_sha256"] == checkpoint["report_sha256"]
            and 0 <= index < len(slots)
            and slots[index] == expected
            and entry.get("round_wire_sha256") == round_wire,
            "evaluation_round_wire_invalid",
        )
    elif "inventory_recovery" in entry:
        binding = entry["inventory_recovery"]
        allocations = [
            value["allocation"]
            for value in ledger.get("capacity_revisions", [])
            if value["allocation"]["identity"] == binding.get("parent_allocation")
        ]
        metadata = checkpoint["progress"]
        wire = json.loads(start["messages"][1]["content"])["inventory_recovery"]
        recovery_metadata = metadata.get("inventory_recovery", {})
        attempts = [
            value
            for value in record["repair_state"]["attempts"]
            if value["index"] == recovery_metadata.get("attempt_index")
        ]
        _mixed_require(
            entry["stage"] == "repair"
            and len(allocations) == 1
            and set(binding)
            == {
                "contract",
                "parent_allocation",
                "source_report_sha256",
                "feedback_sha256",
                "wire_sha256",
            }
            and binding["contract"]
            == ledger["plan"]["envelope"].get("inventory_recovery_contract")
            and binding["wire_sha256"] == round_wire
            and binding["source_report_sha256"]
            == checkpoint["report_sha256"]
            == allocations[0]["round"]["report_sha256"]
            and all(
                isinstance(binding.get(key), str)
                and re.fullmatch(r"[0-9a-f]{64}", binding[key])
                for key in (
                    "parent_allocation",
                    "source_report_sha256",
                    "feedback_sha256",
                    "wire_sha256",
                )
            )
            and all(
                recovery_metadata.get(key) == wire.get(key)
                for key in (
                    "contract",
                    "source_report_sha256",
                    "previous_inventory_sha256",
                    "bindings_sha256",
                    "attempt_index",
                )
            )
            and len(attempts) == 1
            and attempts[0]["stage"] == "evaluation"
            and attempts[0]["input_sha256"] == input_sha
            and attempts[0].get("purpose") == binding["contract"]
            and attempts[0].get("source_report_sha256")
            == binding["source_report_sha256"]
            and (
                "canonical_recovery_feedback" not in record
                or wire_digest(record["canonical_recovery_feedback"])
                == binding["feedback_sha256"]
            ),
            "evaluation_recovery_invalid",
        )
    else:
        _mixed_require(entry["stage"] == "inventory", "evaluation_stage_invalid")
    if phase == "inventory":
        _historical_progress_envelope(record, ledger, request_id)
        wire = json.loads(start["messages"][1]["content"])
        _mixed_require(
            wire.get("host_methodology_validation", {}).get("source_sha256")
            == checkpoint["report_sha256"],
            "inventory_report_identity_invalid",
        )
    semantic_selected = "semantic_conflict_policy" in ledger["plan"].get("envelope", {})
    settlement = progress.get("semantic_conflict_settlement")
    settlement_required = (
        semantic_selected
        and phase == "assessment"
        and progress.get("status") == "response_received"
        and (progress.get("usage") or {}).get("usage_available") is True
    )
    _mixed_require(
        ("semantic_conflict_settlement" in progress) == settlement_required,
        "semantic_settlement_presence_invalid",
    )
    if settlement_required:
        # Authenticate frozen receipts as data. Today's effect code cannot reopen
        # a parent's verdict, and a later unknown request cannot erase known cost.
        envelope = allocation["envelope"]
        prefix = list(ledger["entries"].items())
        prefix = prefix[
            : next(i for i, (key, _) in enumerate(prefix) if key == request_id) + 1
        ]
        expected_receipt = {
            "contract": "host-settled-assessment-receipt-v1",
            "plan_identity": ledger["plan"]["identity"],
            "input_identity": ledger["plan"]["input_identity"],
            "envelope_identity": envelope["identity"],
            "semantic_conflict_policy": envelope["semantic_conflict_policy"],
            "semantic_conflict_identity": envelope["semantic_conflict_identity"],
            "stage": "assessment",
            "request_id": request_id,
            "input_sha256": round_wire,
            "round_number": entry["round_number"],
            "round_wire_sha256": round_wire,
            "wire_sha256": entry["wire_sha256"],
            "usage_sha256": entry["usage_sha256"],
            "reserved_tokens": entry["reserved_tokens"],
            "actual_tokens": entry["actual_tokens"],
            "entry_sha256": wire_digest(entry),
        }
        expected_receipt["identity"] = wire_digest(expected_receipt)
        _mixed_require(
            all(value["status"] == "settled" for _, value in prefix)
            and isinstance(settlement, dict)
            and set(settlement)
            == {
                "receipt",
                "report_sha256",
                "response_sha256",
                "input_sha256",
                "batch_payload_sha256",
            }
            and wire_digest(settlement["receipt"]) == wire_digest(expected_receipt)
            and settlement["report_sha256"] == checkpoint["report_sha256"]
            and settlement["input_sha256"] == input_sha
            and isinstance(progress.get("response"), str)
            and settlement["response_sha256"]
            == hashlib.sha256(progress["response"].encode()).hexdigest()
            and len(
                [
                    batch
                    for batch in checkpoint["progress"].get("batches", [])
                    if batch.get("input_sha256") == settlement["batch_payload_sha256"]
                ]
            )
            == 1,
            "semantic_settlement_binding_invalid",
        )
    return {
        "request_id": request_id,
        "operation_id": start["operation_id"],
        "stage": entry["stage"],
        "phase": phase,
        "input_sha256": input_sha,
        "wire_sha256": wire_sha,
        "reservation": amount,
        "progress_identity": record["identity"],
        "progress_status": progress.get("status"),
        "progress_usage": progress.get("usage"),
    }


def _authenticate_planned_requests(root, manifest, state, row, archive, starts, ends):
    """Authenticate all historical sent requests, whether usage is known or unknown."""
    from litagent.quality import digest as wire_digest
    from litagent.safety.planned_accounting import validate_accounting_snapshot

    run_id = row["run_id"]
    ledger_path = Path(
        "artifacts/targeted-repairs", run_id + ".json.budget-ledger.json"
    )
    journal_path = Path("artifacts/targeted-repairs", run_id + ".json.writer-v2.json")
    ledger = json.loads(ledger_path.read_text())
    journal = json.loads(journal_path.read_text())
    _mixed_require(
        ledger == row["budget_snapshot"]
        and ledger["plan"] == manifest["budget_plans"][row["case_id"]]
        and archive.get("config") == manifest["config"]
        and archive.get("config_fingerprint") == manifest["config"].get("fingerprint")
        and journal["contract"] == manifest["contract"]
        and journal["prompt_contract"] == manifest["writer_prompt_contract"],
        "persisted_identity_invalid",
    )
    validate_accounting_snapshot(
        ledger, expected_plan=manifest["budget_plans"][row["case_id"]]
    )
    envelope = ledger["plan"].get("envelope", {})
    _historical_source_binding(root, manifest, envelope)
    _historical_support_binding(journal, envelope)
    _historical_guidance_wire_owner(journal, envelope)
    _historical_source_view_binding(journal, envelope)
    _historical_source_view_archive(journal, envelope, archive)
    hold = historical_unknown_token_hold(
        ledger,
        row["known_tokens"],
        expected_plan=manifest["budget_plans"][row["case_id"]],
    )
    sent = {key: value for key, value in ledger["entries"].items() if value["sent"]}
    failed = [
        event["data"] for event in archive["events"] if event["event"] == "llm.failed"
    ]
    terminals = ends + failed
    _mixed_require(
        type(row.get("requests")) is int
        and row["requests"]
        == ledger["requests_sent"]
        == len(sent)
        == len(starts)
        == len(terminals)
        and len({value["operation_id"] for value in starts}) == len(starts)
        and len({value["operation_id"] for value in terminals}) == len(terminals)
        and {value["operation_id"] for value in starts}
        == {value["operation_id"] for value in terminals},
        "operation_set_invalid",
    )
    terminal_by_id = {value["operation_id"]: value for value in terminals}
    complete_ids = {value["operation_id"] for value in ends}
    records = evaluation_progress_records(run_id, archive)
    for record in records:
        _mixed_progress_context(record, ledger, manifest, row, archive, journal)
    progress_inputs = [
        value["input_sha256"]
        for record in records
        for value in record["checkpoint"]["progress"].get("requests", [])
    ]
    occurrences = (
        _planned_evaluation_occurrence_bindings(
            archive, starts, records, ledger, manifest
        )
        if len(progress_inputs) != len(set(progress_inputs))
        else None
    )
    used, known, calls = set(), 0, []
    writer_ids = {journal["identity"] + ":" + key for key in journal["calls"]}
    for start in starts:
        binding = _mixed_writer_start(start, ledger, manifest, journal)
        if binding is None:
            expected = (
                occurrences.get(start["operation_id"])
                if occurrences is not None
                else None
            )
            _mixed_require(
                occurrences is None or expected is not None,
                "evaluation_occurrence_binding_missing",
            )
            bindings = [
                value
                for record in records
                if expected is None
                or expected["progress_identity"] == record["identity"]
                if (
                    value := authenticate_planned_evaluation_start(
                        start,
                        record,
                        ledger,
                        manifest,
                        **(
                            {"expected_binding": expected}
                            if expected is not None
                            and expected["progress_identity"] == record["identity"]
                            else {}
                        ),
                    )
                )
                is not None
            ]
            _mixed_require(
                len(bindings) == 1, "evaluation_binding_ambiguous_or_missing"
            )
            binding = bindings[0]
            _mixed_require(
                binding["request_id"] not in writer_ids, "writer_binding_missing"
            )
        request_id = binding["request_id"]
        _mixed_require(request_id not in used, "request_reused")
        used.add(request_id)
        entry, end = sent[request_id], terminal_by_id[start["operation_id"]]
        if start["operation_id"] in complete_ids and end.get("usage_available") is True:
            usage = {key: end[key] for key in ("prompt_tokens", "completion_tokens")}
            _mixed_require(
                all(type(value) is int and value >= 0 for value in usage.values())
                and type(end.get("total_tokens")) is int
                and end["total_tokens"] == sum(usage.values())
                and entry["status"] == "settled"
                and entry["actual_tokens"] == end["total_tokens"]
                and entry["usage_sha256"] == wire_digest(usage),
                "known_usage_invalid",
            )
            known += end["total_tokens"]
            if binding["progress_identity"] is not None:
                _mixed_require(
                    binding["progress_status"] == "response_received"
                    and isinstance(binding["progress_usage"], dict)
                    and all(
                        binding["progress_usage"].get(key) == value
                        for key, value in usage.items()
                    ),
                    "progress_known_usage_invalid",
                )
        else:
            _mixed_require(
                end.get("usage_available") is False
                and entry["status"] == "interrupted_no_replay"
                and entry["actual_tokens"] is None
                and entry.get("usage_sha256") is None,
                "unknown_status_invalid",
            )
            if binding["progress_identity"] is not None:
                _mixed_require(
                    binding["progress_status"] == "interrupted_no_replay"
                    and (
                        binding["progress_usage"] is None
                        or binding["progress_usage"].get("usage_available") is False
                    ),
                    "progress_unknown_usage_invalid",
                )
        calls.append(
            {
                **binding,
                "actual_tokens": entry["actual_tokens"],
                "status": entry["status"],
            }
        )
    _mixed_require(
        used == set(sent) and known == row["known_tokens"], "usage_set_invalid"
    )
    actual_progress = {
        (value["progress_identity"], value["input_sha256"])
        for value in calls
        if value["progress_identity"] is not None
    }
    expected_progress = {
        (record["identity"], value["input_sha256"])
        for record in records
        for value in record["checkpoint"]["progress"].get("requests", [])
    }
    _mixed_require(actual_progress == expected_progress, "progress_request_set_invalid")
    return {
        "version": PLANNED_MIXED_UNKNOWN_ACCOUNTING,
        "unknown_hold": hold,
        "plan_identity": ledger["plan"]["identity"],
        "ledger_sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
        "journal_sha256": hashlib.sha256(journal_path.read_bytes()).hexdigest(),
        "evaluation_progress": [
            {
                key: record[key]
                for key in (
                    "identity",
                    "path",
                    "sha256",
                    "normal_identity_preimage",
                    "owner_verified",
                    "repair_checkpoint_sha256",
                )
                if key in record
            }
            for record in records
        ],
        "requests": calls,
    }


def planned_mixed_unknown_receipt(root, manifest, state, row, archive, starts, ends):
    """Keep the existing unknown receipt format after shared request authentication."""
    receipt = _authenticate_planned_requests(
        root, manifest, state, row, archive, starts, ends
    )
    _mixed_require(
        receipt["unknown_hold"] > 0
        and type(row.get("unknown_hold")) is int
        and row["unknown_hold"] == receipt["unknown_hold"]
        and row.get("usage_complete") is False
        and row.get("publishable") is False,
        "unknown_amount_invalid",
    )
    return receipt


def accounting_history(history, parents, *, unknown_policy="reject"):
    """Carry a verified terminal prefix forward without refunding or duplicating calls."""
    if unknown_policy not in (
        "reject",
        CONSERVATIVE_ACCOUNTING,
        PLANNED_UNKNOWN_ACCOUNTING,
        PLANNED_MIXED_UNKNOWN_ACCOUNTING,
    ):
        raise ValueError("frozen_accounting_policy_invalid")
    history, receipts, seen, seen_archives = dict(history), [], set(), set()
    for parent in parents:
        root = Path(parent)
        if root.resolve() in seen:
            raise ValueError("frozen_accounting_parent_duplicate")
        seen.add(root.resolve())
        mp, sp = root / "manifest.json", root / "state.json"
        manifest, state = json.loads(mp.read_text()), json.loads(sp.read_text())
        terminal_probe_failure = (
            state.get("status") == "blocked"
            and manifest.get("version")
            in {
                "writer-v2-protocol-observation-v2",
                "writer-v2-protocol-observation-v3",
            }
            and state.get("failure", {}).get("family") == "protocol"
            and state.get("requests") == 1
        )
        if (
            manifest.get("history") != history
            or manifest.get("identity")
            != digest({k: v for k, v in manifest.items() if k != "identity"})
            or state.get("identity") != manifest["identity"]
            or (
                state.get("status")
                not in {"stopped", "completed", "protocol_completed"}
                and not terminal_probe_failure
            )
        ):
            raise ValueError("frozen_accounting_prefix_invalid")
        rows = state.get("cases", [state])
        archives = {}
        requests, tokens = 0, 0
        conservative = []
        planned_receipts = []
        for row in rows:
            uncertain = row.get("usage_complete") is not True or row.get(
                "unknown_hold", 0
            )
            snapshot = row.get("budget_snapshot")
            expected_plan = manifest.get("budget_plans", {}).get(row.get("case_id"))
            planned = isinstance(expected_plan, dict) or (
                isinstance(snapshot, dict) and snapshot.get("schema_version") == 2
            )
            if planned:
                if (
                    not isinstance(snapshot, dict)
                    or type(snapshot.get("schema_version")) is not int
                    or snapshot["schema_version"] != 2
                ):
                    raise ValueError("frozen_accounting_budget_snapshot_invalid")
                # Rejection-only callers need no authority to carry a malformed
                # parent. Preserve their existing failure before loading receipts.
                visibly_uncertain = (
                    uncertain or snapshot.get("usage_uncertain") is not False
                )
                if visibly_uncertain and unknown_policy == "reject":
                    raise ValueError("frozen_accounting_usage_unknown")
                if visibly_uncertain and unknown_policy not in {
                    "reject",
                    PLANNED_UNKNOWN_ACCOUNTING,
                    PLANNED_MIXED_UNKNOWN_ACCOUNTING,
                }:
                    raise ValueError(
                        "frozen_accounting_planned_unknown_receipt_unsupported"
                    )
                if not isinstance(expected_plan, dict):
                    raise ValueError("frozen_accounting_planned_plan_conflict")
                if snapshot.get("plan") != expected_plan:
                    expected_plan = _historical_accounting_plan(
                        root, manifest, state, row
                    )
                planned_hold = historical_unknown_token_hold(
                    snapshot, row.get("known_tokens"), expected_plan=expected_plan
                )
                uncertain = (
                    uncertain
                    or snapshot.get("usage_uncertain") is not False
                    or planned_hold > 0
                )
            if uncertain and unknown_policy == "reject":
                raise ValueError("frozen_accounting_usage_unknown")
            if (
                uncertain
                and planned
                and unknown_policy
                not in {PLANNED_UNKNOWN_ACCOUNTING, PLANNED_MIXED_UNKNOWN_ACCOUNTING}
            ):
                # Legacy rejection proofs reconstruct a different phase ledger
                # and cannot authenticate a planned-budget parent. Never refund
                # its escrows or execute arbitrary frozen accounting modules.
                raise ValueError(
                    "frozen_accounting_planned_unknown_receipt_unsupported"
                )
            run_id = row.get("run_id", root.name)
            if Path(run_id).name != run_id:
                raise ValueError("frozen_accounting_run_invalid")
            path = Path("artifacts/runs", run_id + ".json")
            archive = json.loads(path.read_text())
            if archive["run_id"] != run_id or path.resolve() in seen_archives:
                raise ValueError("frozen_accounting_archive_duplicate_or_invalid")
            seen_archives.add(path.resolve())
            events = archive["events"]
            starts = [e["data"] for e in events if e["event"] == "llm.start"]
            ends = [e["data"] for e in events if e["event"] == "llm.complete"]
            if uncertain and planned:
                try:
                    receipt_builder = planned_writer_unknown_receipt
                    if unknown_policy == PLANNED_MIXED_UNKNOWN_ACCOUNTING:
                        journal = json.loads(
                            Path(
                                "artifacts/targeted-repairs",
                                run_id + ".json.writer-v2.json",
                            ).read_text()
                        )
                        writer_ids = {
                            journal["identity"] + ":" + key for key in journal["calls"]
                        }
                        sent_ids = {
                            key
                            for key, value in snapshot["entries"].items()
                            if value["sent"]
                        }
                        if not sent_ids.issubset(writer_ids):
                            receipt_builder = planned_mixed_unknown_receipt
                    planned_receipts.append(
                        receipt_builder(
                            root, manifest, state, row, archive, starts, ends
                        )
                    )
                except (KeyError, TypeError, AttributeError, OSError) as exc:
                    raise ValueError(
                        "frozen_accounting_planned_receipt_malformed"
                    ) from exc
                requests += row["requests"]
                tokens += row["known_tokens"]
                archives[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
                continue
            if (
                type(row.get("requests")) is not int
                or type(row.get("known_tokens")) is not int
                or row["requests"] < 0
                or row["known_tokens"] < 0
                or len(starts) != len(ends)
                or len(starts) != row["requests"]
                or len({e["operation_id"] for e in starts}) != len(starts)
                or {e["operation_id"] for e in starts}
                != {e["operation_id"] for e in ends}
                or any(
                    e.get("usage_available") is not True and not uncertain for e in ends
                )
                or any(
                    any(
                        type(e.get(k)) is not int or e[k] < 0
                        for k in ("prompt_tokens", "completion_tokens", "total_tokens")
                    )
                    or e["total_tokens"] != e["prompt_tokens"] + e["completion_tokens"]
                    for e in ends
                    if e.get("usage_available") is True
                )
                or sum(
                    e["total_tokens"] for e in ends if e.get("usage_available") is True
                )
                != row["known_tokens"]
            ):
                raise ValueError("frozen_accounting_archive_usage_invalid")
            if planned and "support_validation_revision" in expected_plan.get(
                "envelope", {}
            ):
                authenticated = _authenticate_planned_requests(
                    root, manifest, state, row, archive, starts, ends
                )
                _mixed_require(
                    authenticated["unknown_hold"] == 0
                    and row.get("usage_complete") is True
                    and snapshot["usage_uncertain"] is False,
                    "known_usage_invalid",
                )
            if uncertain:
                conservative.append(
                    conservative_rejection_receipt(
                        root, manifest, state, row, archive, starts, ends
                    )
                )
            requests += row["requests"]
            tokens += row["known_tokens"]
            archives[str(path)] = hashlib.sha256(path.read_bytes()).hexdigest()
        receipts.append(
            {
                "root": str(root),
                "manifest_sha256": hashlib.sha256(mp.read_bytes()).hexdigest(),
                "state_sha256": hashlib.sha256(sp.read_bytes()).hexdigest(),
                "archives": archives,
                "requests": requests,
                "known_tokens": tokens,
                "history_before": dict(history),
                **(
                    {
                        "conservative_rejections": conservative,
                        "unknown_hold": sum(r["unknown_hold"] for r in conservative),
                    }
                    if conservative
                    else {}
                ),
                **(
                    {
                        "planned_unknown_receipts": planned_receipts,
                        "unknown_hold": sum(
                            r["unknown_hold"] for r in conservative + planned_receipts
                        ),
                    }
                    if planned_receipts
                    else {}
                ),
            }
        )
        history["model_requests"] += requests
        history["model_known_tokens"] += tokens
        if conservative or planned_receipts:
            history["model_unknown_conservative_tokens"] = history.get(
                "model_unknown_conservative_tokens", 0
            ) + sum(r["unknown_hold"] for r in conservative + planned_receipts)
    return history, receipts


def prepare(
    root,
    config,
    *,
    accounting_parents=(),
    writing_profile=DISABLED_PROFILE,
    writing_plan=None,
    inventory_recovery=False,
    mixed_inventory_recovery=False,
    carry_planned_unknown=False,
    accounting_policy=None,
    classification_review=False,
    repair_feedback=False,
    reference_contract=None,
    transport_revision=None,
    support_validation_revision=None,
    repair_mapping_contract=None,
    repair_guidance_contract=None,
    repair_guidance_wire_revision=None,
    semantic_conflict_policy=None,
    evaluation_response_contract=None,
    repair_prompt_revision=None,
    round_resource_contract=None,
    writer_source_view=None,
    evaluation_dimensions_contract=None,
    closeout_profile=None,
):
    """Freeze all three materials and the same source before any paid request."""
    if accounting_policy is not None and (
        accounting_policy != PLANNED_MIXED_UNKNOWN_ACCOUNTING
        or writing_profile != ROUND_PROFILE
        or carry_planned_unknown is not False
    ):
        raise ValueError("frozen_accounting_policy_invalid")
    check_profile(config, writing_profile, writing_plan=writing_plan)
    if evaluation_dimensions_contract is not None:
        from litagent.eval.batched import response_contract_identity
        from litagent.eval.dimensions import dimensions_identity

        evaluation_dimensions_options(
            {
                "evaluation_dimensions_contract": evaluation_dimensions_contract,
                "evaluation_dimensions_identity": dimensions_identity(),
                "writing_profile": writing_profile,
                "inventory_recovery": inventory_recovery,
                "mixed_inventory_recovery": mixed_inventory_recovery,
                "classification_review": classification_review,
                "repair_feedback": repair_feedback,
                "reference_contract": reference_contract,
                "transport_revision": transport_revision,
                "support_validation_revision": support_validation_revision,
                "evaluation_response_contract": evaluation_response_contract,
                "evaluation_response_identity": (
                    response_contract_identity(contract=evaluation_response_contract)
                    if evaluation_response_contract is not None
                    else None
                ),
            }
        )
    if writer_source_view is not None:
        from litagent.eval.batched import response_contract_identity
        from litagent.writer_source_view import source_view_identity
        from litagent.writing_flat import (
            REPAIR_TASK_VIEW_ADMISSION_RULE,
            repair_guidance_wire_identity,
            repair_prompt_identity,
        )

        writer_source_view_options(
            {
                "writer_source_view": writer_source_view,
                "writer_source_view_identity": source_view_identity(),
                "writing_profile": writing_profile,
                "inventory_recovery": inventory_recovery,
                "mixed_inventory_recovery": mixed_inventory_recovery,
                "classification_review": classification_review,
                "repair_feedback": repair_feedback,
                "reference_contract": reference_contract,
                "transport_revision": transport_revision,
                "support_validation_revision": support_validation_revision,
                "repair_mapping_contract": repair_mapping_contract,
                "repair_guidance_contract": repair_guidance_contract,
                "repair_guidance_wire_revision": repair_guidance_wire_revision,
                "repair_guidance_wire_identity": repair_guidance_wire_identity(
                    "host-repair-task-view-v1"
                ),
                "repair_guidance_capacity_rule": REPAIR_TASK_VIEW_ADMISSION_RULE,
                "evaluation_response_contract": evaluation_response_contract,
                "evaluation_response_identity": (
                    response_contract_identity(contract=evaluation_response_contract)
                    if evaluation_response_contract is not None
                    else None
                ),
                "repair_prompt_revision": repair_prompt_revision,
                "repair_prompt_identity": repair_prompt_identity(),
            }
        )
    if round_resource_contract is not None:
        from litagent.safety.planned_budget import joint_round_escrow_identity

        round_resource_options(
            {
                "round_resource_contract": round_resource_contract,
                "round_resource_identity": joint_round_escrow_identity(),
                "writing_profile": writing_profile,
            }
        )
    if repair_prompt_revision is not None:
        from litagent.writing_flat import (
            REPAIR_TASK_VIEW_ADMISSION_RULE,
            repair_guidance_wire_identity,
            repair_prompt_identity,
        )

        repair_prompt_options(
            {
                "repair_prompt_revision": repair_prompt_revision,
                "repair_prompt_identity": repair_prompt_identity(),
                "writing_profile": writing_profile,
                "inventory_recovery": inventory_recovery,
                "mixed_inventory_recovery": mixed_inventory_recovery,
                "classification_review": classification_review,
                "repair_feedback": repair_feedback,
                "reference_contract": reference_contract,
                "transport_revision": transport_revision,
                "support_validation_revision": support_validation_revision,
                "repair_mapping_contract": repair_mapping_contract,
                "repair_guidance_contract": repair_guidance_contract,
                "repair_guidance_wire_revision": repair_guidance_wire_revision,
                "repair_guidance_wire_identity": repair_guidance_wire_identity(
                    "host-repair-task-view-v1"
                ),
                "repair_guidance_capacity_rule": REPAIR_TASK_VIEW_ADMISSION_RULE,
            }
        )
    if evaluation_response_contract is not None:
        from litagent.eval.batched import response_contract_identity

        evaluation_response_options(
            {
                "evaluation_response_contract": evaluation_response_contract,
                "evaluation_response_identity": response_contract_identity(
                    contract=evaluation_response_contract
                ),
                "writing_profile": writing_profile,
                "inventory_recovery": inventory_recovery,
                "mixed_inventory_recovery": mixed_inventory_recovery,
                "classification_review": classification_review,
                "repair_feedback": repair_feedback,
                "reference_contract": reference_contract,
                "transport_revision": transport_revision,
                "support_validation_revision": support_validation_revision,
            }
        )
    if semantic_conflict_policy is not None:
        from litagent.eval.semantic_conflict import semantic_conflict_identity
        from litagent.writing_flat import (
            REPAIR_TASK_VIEW_ADMISSION_RULE,
            repair_guidance_wire_identity,
        )

        semantic_conflict_options(
            {
                "semantic_conflict_policy": semantic_conflict_policy,
                "semantic_conflict_identity": semantic_conflict_identity(),
                "writing_profile": writing_profile,
                "inventory_recovery": inventory_recovery,
                "mixed_inventory_recovery": mixed_inventory_recovery,
                "classification_review": classification_review,
                "repair_feedback": repair_feedback,
                "reference_contract": reference_contract,
                "transport_revision": transport_revision,
                "support_validation_revision": support_validation_revision,
                "repair_mapping_contract": repair_mapping_contract,
                "repair_guidance_contract": repair_guidance_contract,
                "repair_guidance_wire_revision": repair_guidance_wire_revision,
                "repair_guidance_wire_identity": repair_guidance_wire_identity(
                    repair_guidance_wire_revision
                ),
                "repair_guidance_capacity_rule": REPAIR_TASK_VIEW_ADMISSION_RULE,
            }
        )
    if repair_guidance_wire_revision is not None:
        from litagent.writing_flat import (
            REPAIR_GUIDANCE_WIRE_REVISION,
            REPAIR_TASK_VIEW_ADMISSION_RULE,
            REPAIR_TASK_VIEW_REVISION,
            repair_guidance_wire_identity,
        )

        repair_guidance_wire_options(
            {
                "repair_guidance_wire_revision": repair_guidance_wire_revision,
                "repair_guidance_wire_identity": (
                    repair_guidance_wire_identity()
                    if repair_guidance_wire_revision == REPAIR_GUIDANCE_WIRE_REVISION
                    else repair_guidance_wire_identity(repair_guidance_wire_revision)
                ),
                **(
                    {"repair_guidance_capacity_rule": REPAIR_TASK_VIEW_ADMISSION_RULE}
                    if repair_guidance_wire_revision == REPAIR_TASK_VIEW_REVISION
                    else {}
                ),
                "repair_guidance_contract": repair_guidance_contract,
                "repair_mapping_contract": repair_mapping_contract,
                "support_validation_revision": support_validation_revision,
                "writing_profile": writing_profile,
                "inventory_recovery": inventory_recovery,
                "mixed_inventory_recovery": mixed_inventory_recovery,
                "classification_review": classification_review,
                "repair_feedback": repair_feedback,
                "reference_contract": reference_contract,
                "transport_revision": transport_revision,
            }
        )
    if repair_guidance_contract is not None:
        repair_guidance_options(
            {
                "repair_guidance_contract": repair_guidance_contract,
                "repair_mapping_contract": repair_mapping_contract,
                "support_validation_revision": support_validation_revision,
                "writing_profile": writing_profile,
                "inventory_recovery": inventory_recovery,
                "mixed_inventory_recovery": mixed_inventory_recovery,
                "classification_review": classification_review,
                "repair_feedback": repair_feedback,
                "reference_contract": reference_contract,
                "transport_revision": transport_revision,
            }
        )
    if repair_mapping_contract is not None:
        repair_mapping_options(
            {
                "repair_mapping_contract": repair_mapping_contract,
                "support_validation_revision": support_validation_revision,
                "repair_guidance_contract": repair_guidance_contract,
                "repair_guidance_wire_revision": repair_guidance_wire_revision,
                "writing_profile": writing_profile,
                "inventory_recovery": inventory_recovery,
                "mixed_inventory_recovery": mixed_inventory_recovery,
                "classification_review": classification_review,
                "repair_feedback": repair_feedback,
                "reference_contract": reference_contract,
                "transport_revision": transport_revision,
            }
        )
    if support_validation_revision is not None:
        support_validation_options(
            {
                "support_validation_revision": support_validation_revision,
                "writing_profile": writing_profile,
                "inventory_recovery": inventory_recovery,
                "mixed_inventory_recovery": mixed_inventory_recovery,
                "classification_review": classification_review,
                "repair_feedback": repair_feedback,
                "reference_contract": reference_contract,
                "transport_revision": transport_revision,
            }
        )
    if transport_revision is not None:
        transport_revision_options(
            {
                "transport_revision": transport_revision,
                "writing_profile": writing_profile,
                "inventory_recovery": inventory_recovery,
                "mixed_inventory_recovery": mixed_inventory_recovery,
                "repair_feedback": repair_feedback,
                "reference_contract": reference_contract,
            }
        )
    if reference_contract is not None:
        reference_contract_options(
            {
                "reference_contract": reference_contract,
                "writing_profile": writing_profile,
            }
        )
    if type(inventory_recovery) is not bool or (
        inventory_recovery and writing_profile != ROUND_PROFILE
    ):
        raise ValueError("frozen_inventory_recovery_configuration_invalid")
    if type(mixed_inventory_recovery) is not bool or (
        mixed_inventory_recovery
        and (writing_profile != ROUND_PROFILE or inventory_recovery is not True)
    ):
        raise ValueError("frozen_mixed_inventory_recovery_configuration_invalid")
    if type(carry_planned_unknown) is not bool or (
        carry_planned_unknown and writing_profile != ROUND_PROFILE
    ):
        raise ValueError("frozen_planned_unknown_configuration_invalid")
    if type(classification_review) is not bool or (
        classification_review and writing_profile != ROUND_PROFILE
    ):
        raise ValueError("frozen_classification_review_configuration_invalid")
    if type(repair_feedback) is not bool or (
        repair_feedback and writing_profile != ROUND_PROFILE
    ):
        raise ValueError("frozen_repair_feedback_configuration_invalid")
    root = Path(root)
    if root.exists():
        raise ValueError("frozen_campaign_already_exists")
    rows = []
    for case, run in CASES:
        bundle = read_bundle(Path("artifacts/runs", run + ".json"))
        rows.append({"case_id": case, "bundle": bundle})
    paths = source_paths()
    hashes = {
        str(p): hashlib.sha256(p.read_bytes()).hexdigest()
        for p in sorted(set(paths))
        if p.is_file() and "__pycache__" not in str(p)
    }
    history = json.loads(
        Path(
            "artifacts/acceptance/survey-reliability-20261022-004/acceptance-summary.json"
        ).read_text()
    )["cumulative"]
    policy = (
        CONSERVATIVE_ACCOUNTING
        if writing_profile
        in (
            RESPONSES_PROFILE,
            SHAPE_PROFILE,
            REEMISSION_PROFILE,
            ROLE_PROFILE,
            EOF_PROFILE,
            PLANNED_PROFILE,
            ADAPTIVE_PROFILE,
            BUDGETED_PROFILE,
            FLAT_PROFILE,
            CAPACITY_PROFILE,
            LEGACY_REEMISSION_PROFILE,
            ROUND_PROFILE,
        )
        else "reject"
    )
    if carry_planned_unknown:
        policy = PLANNED_UNKNOWN_ACCOUNTING
    if accounting_policy is not None:
        policy = accounting_policy
    history, receipts = accounting_history(
        history, accounting_parents, unknown_policy=policy
    )
    manifest = {
        "version": VERSION,
        "suite_id": root.name,
        "contract": CONTRACT,
        "writer_prompt_contract": PROMPT_CONTRACT,
        "source_commit": subprocess.check_output(
            ["git", "rev-parse", "HEAD"], text=True
        ).strip(),
        "options": profile_options(writing_profile).model_dump(mode="json"),
        "config": build_config_summary(config, "strict_auto_v2"),
        "source_hashes": hashes,
        "cases": rows,
        "history": history,
        "limits": {
            "cases": 3,
            "per_case_tokens": 500000,
            "per_case_requests": 64,
            "total_tokens": 1500000,
            "total_requests": 192,
            "repairs": 2,
            "retries": 0,
            "formal_cases": 0,
        },
    }
    manifest["limits"] = dict(LIMITS)
    if writing_profile in (
        BUDGETED_PROFILE,
        FLAT_PROFILE,
        CAPACITY_PROFILE,
        ROUND_PROFILE,
    ):
        from litagent.writing_v2 import NATIVE_WIRE_CONTRACT

        manifest["writer_wire_contract"] = (
            "flat-writer-v1"
            if writing_profile in (FLAT_PROFILE, CAPACITY_PROFILE, ROUND_PROFILE)
            else NATIVE_WIRE_CONTRACT
        )
    if writing_profile != DISABLED_PROFILE:
        manifest["writing_profile"] = writing_profile
    if profile_parameters(writing_profile, writing_plan=writing_plan):
        manifest["writing_parameters"] = profile_parameters(
            writing_profile, writing_plan=writing_plan
        )
    if inventory_recovery:
        manifest["inventory_recovery"] = True
        manifest["writing_parameters"].update(inventory_recovery_options(manifest))
    if mixed_inventory_recovery:
        manifest["mixed_inventory_recovery"] = True
        manifest["options"] = manifest_options(manifest)
        manifest["writing_parameters"].update(
            mixed_inventory_recovery_options(manifest)
        )
    if classification_review:
        manifest["classification_review"] = True
        manifest["options"] = manifest_options(manifest)
        manifest["writing_parameters"].update(classification_review_options(manifest))
    if repair_feedback:
        manifest["repair_feedback"] = True
        manifest["options"] = manifest_options(manifest)
        manifest["writing_parameters"].update(repair_feedback_options(manifest))
    if reference_contract is not None:
        manifest["reference_contract"] = reference_contract
        manifest["options"] = manifest_options(manifest)
        manifest["writing_parameters"].update(reference_contract_options(manifest))
    if transport_revision is not None:
        manifest["transport_revision"] = transport_revision
        manifest["options"] = manifest_options(manifest)
        manifest["writing_parameters"].update(transport_revision_options(manifest))
    if support_validation_revision is not None:
        manifest["support_validation_revision"] = support_validation_revision
        manifest["options"] = manifest_options(manifest)
        manifest["writing_parameters"].update(support_validation_options(manifest))
    if repair_mapping_contract is not None:
        manifest["repair_mapping_contract"] = repair_mapping_contract
        # Dependent guidance markers are assembled immediately below. Validate
        # their selected values now without publishing a partial manifest.
        manifest["writing_parameters"].update(
            repair_mapping_options(
                {
                    **manifest,
                    "repair_guidance_contract": repair_guidance_contract,
                    "repair_guidance_wire_revision": repair_guidance_wire_revision,
                }
            )
        )
    if repair_guidance_contract is not None:
        manifest["repair_guidance_contract"] = repair_guidance_contract
        manifest["writing_parameters"].update(repair_guidance_options(manifest))
    if repair_guidance_wire_revision is not None:
        from litagent.writing_flat import (
            REPAIR_GUIDANCE_WIRE_REVISION,
            REPAIR_TASK_VIEW_ADMISSION_RULE,
            REPAIR_TASK_VIEW_REVISION,
            repair_guidance_wire_identity,
        )

        manifest["repair_guidance_wire_revision"] = repair_guidance_wire_revision
        manifest["repair_guidance_wire_identity"] = (
            repair_guidance_wire_identity()
            if repair_guidance_wire_revision == REPAIR_GUIDANCE_WIRE_REVISION
            else repair_guidance_wire_identity(repair_guidance_wire_revision)
        )
        if repair_guidance_wire_revision == REPAIR_TASK_VIEW_REVISION:
            manifest["repair_guidance_capacity_rule"] = REPAIR_TASK_VIEW_ADMISSION_RULE
        manifest["writing_parameters"].update(repair_guidance_wire_options(manifest))
    if semantic_conflict_policy is not None:
        from litagent.eval.semantic_conflict import semantic_conflict_identity

        manifest["semantic_conflict_policy"] = semantic_conflict_policy
        manifest["semantic_conflict_identity"] = semantic_conflict_identity()
        semantic_conflict_options(manifest)
        manifest["writing_parameters"].update(
            {
                key: manifest[key]
                for key in ("semantic_conflict_policy", "semantic_conflict_identity")
            }
        )
    if evaluation_response_contract is not None:
        from litagent.eval.batched import response_contract_identity

        manifest["evaluation_response_contract"] = evaluation_response_contract
        manifest["evaluation_response_identity"] = response_contract_identity(
            contract=evaluation_response_contract
        )
        evaluation_response_options(manifest)
        manifest["writing_parameters"].update(
            {
                key: manifest[key]
                for key in (
                    "evaluation_response_contract",
                    "evaluation_response_identity",
                )
            }
        )
    if repair_prompt_revision is not None:
        from litagent.writing_flat import repair_prompt_identity

        manifest["repair_prompt_revision"] = repair_prompt_revision
        manifest["repair_prompt_identity"] = repair_prompt_identity()
        repair_prompt_options(manifest)
        manifest["writing_parameters"].update(
            {
                key: manifest[key]
                for key in ("repair_prompt_revision", "repair_prompt_identity")
            }
        )
    if writer_source_view is not None:
        from litagent.writer_source_view import source_view_identity

        manifest["writer_source_view"] = writer_source_view
        manifest["writer_source_view_identity"] = source_view_identity()
        writer_source_view_options(manifest)
        manifest["writing_parameters"].update(
            {
                key: manifest[key]
                for key in ("writer_source_view", "writer_source_view_identity")
            }
        )
    if evaluation_dimensions_contract is not None:
        from litagent.eval.dimensions import dimensions_identity

        manifest["evaluation_dimensions_contract"] = evaluation_dimensions_contract
        manifest["evaluation_dimensions_identity"] = dimensions_identity()
        evaluation_dimensions_options(manifest)
        manifest["writing_parameters"].update(
            {
                key: manifest[key]
                for key in (
                    "evaluation_dimensions_contract",
                    "evaluation_dimensions_identity",
                )
            }
        )
    if round_resource_contract is not None:
        from litagent.safety.planned_budget import joint_round_escrow_identity

        manifest["round_resource_contract"] = round_resource_contract
        manifest["round_resource_identity"] = joint_round_escrow_identity()
        manifest["writing_parameters"].update(round_resource_options(manifest))
        manifest["writing_parameters"]["round_resource_identity"] = manifest[
            "round_resource_identity"
        ]
    if closeout_profile is not None:
        from litagent.closeout import profile_identity

        manifest["closeout_profile"] = closeout_profile
        manifest["closeout_profile_identity"] = profile_identity()
        manifest["writing_parameters"].update(closeout_options(manifest))
    if writing_plan is not None:
        manifest["writing_plan"] = writing_plan
        manifest["writing_plan_spec"] = writing_plan_spec(writing_plan)
    if policy != "reject":
        manifest["accounting_policy"] = policy
    if receipts:
        manifest["accounting_receipts"] = receipts
    if writing_profile in PLANNED_PROFILES:
        from litagent.frozen_writing import plan_frozen_budget

        manifest["budget_plans"] = {
            row["case_id"]: plan_frozen_budget(
                row["bundle"],
                **inventory_recovery_options(manifest),
                **mixed_inventory_recovery_options(manifest),
                **classification_review_options(manifest),
                **repair_feedback_options(manifest),
                **reference_contract_options(manifest),
                **transport_revision_options(manifest),
                **support_validation_options(manifest),
                **repair_mapping_options(manifest),
                **repair_guidance_options(manifest),
                **repair_guidance_wire_options(manifest),
                **semantic_conflict_options(manifest),
                **evaluation_response_options(manifest),
                **repair_prompt_options(manifest),
                **round_resource_options(manifest),
                **writer_source_view_options(manifest),
                **evaluation_dimensions_options(manifest),
                **closeout_options(manifest),
                budget_policy=profile_parameters(
                    writing_profile, writing_plan=writing_plan
                )["budget_policy"],
                **(
                    {"capacity": config.capacity}
                    if writing_profile in (CAPACITY_PROFILE, ROUND_PROFILE)
                    else {}
                ),
            )
            for row in rows
        }
        admitted = all(
            p["status"] == "admitted" for p in manifest["budget_plans"].values()
        )
        manifest["limits"] = {
            **LIMITS,
            "per_case_tokens": (
                4500000
                if writing_profile in (FLAT_PROFILE, CAPACITY_PROFILE, ROUND_PROFILE)
                else 1000000
            ),
            "total_tokens": (
                sum(p["planned_tokens"] for p in manifest["budget_plans"].values())
                if admitted
                else (
                    13500000
                    if writing_profile
                    in (FLAT_PROFILE, CAPACITY_PROFILE, ROUND_PROFILE)
                    else 3000000
                )
            ),
        }
    manifest["runtime"] = runtime_identity()
    manifest["identity"] = digest(manifest)
    root.mkdir(parents=True)
    atomic_json(root / "manifest.json", manifest)
    for name, h in hashes.items():
        p = root / "source" / name
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_bytes(Path(name).read_bytes())
    return {
        "identity": manifest["identity"],
        "cases": [
            {
                "id": r["case_id"],
                "source": r["bundle"]["source_run_id"],
                "cards": len(r["bundle"]["selection"]["selected_items"]),
            }
            for r in rows
        ],
    }


def expected_limits(manifest):
    """Recompute new-budget scope instead of trusting rehashed client allowances."""
    if manifest.get("writing_profile") not in PLANNED_PROFILES:
        if "budget_plans" in manifest:
            raise ValueError("frozen_budget_policy_conflict")
        return LIMITS
    from litagent.frozen_writing import plan_frozen_budget

    expected = {
        r["case_id"]: plan_frozen_budget(
            r["bundle"],
            **inventory_recovery_options(manifest),
            **mixed_inventory_recovery_options(manifest),
            **classification_review_options(manifest),
            **repair_feedback_options(manifest),
            **reference_contract_options(manifest),
            **transport_revision_options(manifest),
            **support_validation_options(manifest),
            **repair_mapping_options(manifest),
            **repair_guidance_options(manifest),
            **repair_guidance_wire_options(manifest),
            **semantic_conflict_options(manifest),
            **evaluation_response_options(manifest),
            **repair_prompt_options(manifest),
            **round_resource_options(manifest),
            **writer_source_view_options(manifest),
            **evaluation_dimensions_options(manifest),
            **closeout_options(manifest),
            budget_policy=profile_parameters(
                manifest["writing_profile"], writing_plan=manifest.get("writing_plan")
            )["budget_policy"],
            **(
                {"capacity": manifest["config"]["effective"].get("capacity")}
                if manifest["writing_profile"] in (CAPACITY_PROFILE, ROUND_PROFILE)
                else {}
            ),
        )
        for r in manifest["cases"]
    }
    if manifest.get("writing_profile") in (
        BUDGETED_PROFILE,
        FLAT_PROFILE,
        CAPACITY_PROFILE,
        ROUND_PROFILE,
    ):
        from litagent.frozen_writing import validate_frozen_writer_plan

        validate_frozen_writer_plan(manifest.get("budget_plans"), expected)
    elif manifest.get("budget_plans") != expected:
        raise ValueError("frozen_budget_plan_identity_conflict")
    admitted = all(p["status"] == "admitted" for p in expected.values())
    return {
        **LIMITS,
        "per_case_tokens": (
            4500000
            if manifest.get("writing_profile")
            in (FLAT_PROFILE, CAPACITY_PROFILE, ROUND_PROFILE)
            else 1000000
        ),
        "total_tokens": (
            sum(p["planned_tokens"] for p in expected.values())
            if admitted
            else (
                13500000
                if manifest.get("writing_profile")
                in (FLAT_PROFILE, CAPACITY_PROFILE, ROUND_PROFILE)
                else 3000000
            )
        ),
    }


def validate(root, config):
    """Reject changed sources, materials, runtime config or campaign scope."""
    m = json.loads((root / "manifest.json").read_text())
    if m.get("writing_profile") in (
        BUDGETED_PROFILE,
        FLAT_PROFILE,
        CAPACITY_PROFILE,
        ROUND_PROFILE,
    ):
        from litagent.writing_v2 import NATIVE_WIRE_CONTRACT

        if m.get("writer_wire_contract") != (
            "flat-writer-v1"
            if m.get("writing_profile")
            in (FLAT_PROFILE, CAPACITY_PROFILE, ROUND_PROFILE)
            else NATIVE_WIRE_CONTRACT
        ):
            raise ValueError("frozen_writer_wire_contract_conflict")
    elif "writer_wire_contract" in m:
        raise ValueError("frozen_writer_wire_contract_conflict")
    if (
        m.get("version") != VERSION
        or m.get("limits") != expected_limits(m)
        or m.get("runtime") != runtime_identity()
        or m.get("contract") != CONTRACT
        or m.get("writer_prompt_contract") != PROMPT_CONTRACT
        or set(m.get("source_hashes", {})) != {str(p) for p in source_paths()}
        or digest({k: v for k, v in m.items() if k != "identity"}) != m["identity"]
        or m["config"] != build_config_summary(config, "strict_auto_v2")
        or m["options"] != manifest_options(m)
        or [(r["case_id"], r["bundle"]["source_run_id"]) for r in m["cases"]]
        != list(CASES)
    ):
        raise ValueError("frozen_campaign_identity_conflict")
    check_profile(
        config,
        m.get("writing_profile", DISABLED_PROFILE),
        writing_plan=m.get("writing_plan"),
    )
    if m.get("writing_parameters", {}) != {
        **profile_parameters(
            m.get("writing_profile", DISABLED_PROFILE),
            writing_plan=m.get("writing_plan"),
        ),
        **inventory_recovery_options(m),
        **mixed_inventory_recovery_options(m),
        **classification_review_options(m),
        **repair_feedback_options(m),
        **reference_contract_options(m),
        **transport_revision_options(m),
        **support_validation_options(m),
        **repair_mapping_options(m),
        **repair_guidance_options(m),
        **repair_guidance_wire_options(m),
        **semantic_conflict_options(m),
        **evaluation_response_options(m),
        **repair_prompt_options(m),
        **round_resource_options(m),
        **writer_source_view_options(m),
        **evaluation_dimensions_options(m),
        **closeout_options(m),
        **(
            {"evaluation_dimensions_identity": m["evaluation_dimensions_identity"]}
            if "evaluation_dimensions_identity" in m
            else {}
        ),
        **(
            {"writer_source_view_identity": m["writer_source_view_identity"]}
            if "writer_source_view_identity" in m
            else {}
        ),
        **(
            {"round_resource_identity": m["round_resource_identity"]}
            if "round_resource_identity" in m
            else {}
        ),
        **(
            {"repair_prompt_identity": m["repair_prompt_identity"]}
            if "repair_prompt_identity" in m
            else {}
        ),
        **(
            {"evaluation_response_identity": m["evaluation_response_identity"]}
            if "evaluation_response_identity" in m
            else {}
        ),
        **(
            {"semantic_conflict_identity": m["semantic_conflict_identity"]}
            if "semantic_conflict_identity" in m
            else {}
        ),
    }:
        raise ValueError("frozen_writing_parameters_conflict")
    if m.get("writing_plan_spec") != writing_plan_spec(m.get("writing_plan")):
        raise ValueError("frozen_writing_plan_identity_conflict")
    policy = (
        CONSERVATIVE_ACCOUNTING
        if m.get("writing_profile")
        in (
            RESPONSES_PROFILE,
            SHAPE_PROFILE,
            REEMISSION_PROFILE,
            ROLE_PROFILE,
            EOF_PROFILE,
            PLANNED_PROFILE,
            ADAPTIVE_PROFILE,
            BUDGETED_PROFILE,
            FLAT_PROFILE,
            CAPACITY_PROFILE,
            LEGACY_REEMISSION_PROFILE,
            ROUND_PROFILE,
        )
        else "reject"
    )
    if m.get("accounting_policy") in {
        PLANNED_UNKNOWN_ACCOUNTING,
        PLANNED_MIXED_UNKNOWN_ACCOUNTING,
    }:
        if m.get("writing_profile") != ROUND_PROFILE:
            raise ValueError("frozen_planned_unknown_configuration_invalid")
        policy = m["accounting_policy"]
    if m.get("accounting_policy", "reject") != policy:
        raise ValueError("frozen_accounting_policy_conflict")
    if m.get("accounting_receipts"):
        receipts = m["accounting_receipts"]
        history, verified = accounting_history(
            receipts[0]["history_before"],
            [r["root"] for r in receipts],
            unknown_policy=policy,
        )
        if history != m["history"] or verified != receipts:
            raise ValueError("frozen_campaign_accounting_changed")
    for name, h in m["source_hashes"].items():
        if (
            hashlib.sha256(Path(name).read_bytes()).hexdigest() != h
            or hashlib.sha256((root / "source" / name).read_bytes()).hexdigest() != h
        ):
            raise ValueError("frozen_campaign_source_changed")
    for row in m["cases"]:
        path = Path("artifacts/runs", row["bundle"]["source_run_id"] + ".json")
        if (
            hashlib.sha256(path.read_bytes()).hexdigest()
            != row["bundle"]["archive_sha256"]
        ):
            raise ValueError("frozen_campaign_archive_changed")
    return m


def frozen_runtime_guard(root, manifest, config):
    """Recheck an already validated server seal without rerunning history proofs."""
    from litagent.writing_v2 import WriterError

    sealed = copy.deepcopy(manifest)

    def check():
        try:
            from litagent.frozen_writing import validate_frozen_writer_plan

            current = json.loads((root / "manifest.json").read_text())
            validate_frozen_writer_plan(current, sealed)
            validate_frozen_writer_plan(
                build_config_summary(config, "strict_auto_v2"), sealed["config"]
            )
            for name, sha in sealed["source_hashes"].items():
                if any(
                    hashlib.sha256(path.read_bytes()).hexdigest() != sha
                    for path in (Path(name), root / "source" / name)
                ):
                    raise ValueError("changed_source")
        except (OSError, ValueError, TypeError, KeyError):
            raise WriterError("frozen_campaign_source_changed", "identity") from None

    return check


def safe_reason(reason):
    """Keep exception prose and model content out of public machine-code fields."""
    return (
        reason
        if isinstance(reason, str) and re.fullmatch(r"[a-z][a-z0-9_]{0,127}", reason)
        else "evaluation_not_completed"
    )


def evaluation_diagnostic(report):
    """Return safe scalar diagnostics without publishing evaluator request payloads."""
    details = report.get("evaluation", {}).get("faithfulness", {}).get("details", {})
    reason = next(
        (details[k] for k in ("skipped_reason", "error", "reason") if details.get(k)),
        None,
    )
    if not reason:
        return None
    stages = details.get("stages", {})

    def count(value):
        """Unknown historical counts remain unavailable, not scientific zero."""
        return value if type(value) is int and value >= 0 else None

    return {
        "reason_code": safe_reason(reason),
        "expected_assertions": count(stages.get("expected_assertions")),
        "processed_assertions": count(stages.get("processed_assertions")),
        "scientific_unverified": count(details.get("unverified_count")),
    }


def semantic_conflict_diagnostic(report, envelope):
    """Display pinned model verdicts separately from conservative host effects."""
    if "semantic_conflict_policy" not in envelope:
        return {}
    details = report.get("evaluation", {}).get("faithfulness", {}).get("details", {})
    fields = {
        "contract",
        "policy",
        "policy_identity",
        "code",
        "original_status",
        "original_reason_code",
        "decision_sha256",
        "binding_sha256",
    }

    def effects(assertions):
        found = {}
        for assertion in assertions:
            if not isinstance(assertion, dict):
                return None
            effect = assertion.get("assessment_conflict_effect")
            if effect is None:
                continue
            if (
                not isinstance(effect, dict)
                or set(effect) != fields
                or effect["contract"] != "host-assessment-conflict-effect-v1"
                or effect["policy"] != envelope["semantic_conflict_policy"]
                or effect["policy_identity"] != envelope["semantic_conflict_identity"]
                or effect["code"]
                not in {
                    "positive_status_with_negative_required_aspect",
                    "negative_status_with_complete_positive_diagnostics",
                }
                or assertion.get("status") != "unverifiable"
                or effect["original_status"]
                != (
                    "supported"
                    if effect["code"] == "positive_status_with_negative_required_aspect"
                    else "unsupported"
                )
                or not isinstance(effect["original_reason_code"], str)
                or re.fullmatch(
                    r"[a-z][a-z0-9_]{0,127}", effect["original_reason_code"]
                )
                is None
                or not isinstance(assertion.get("claim_id"), str)
                or any(
                    not isinstance(effect[key], str)
                    or re.fullmatch(r"[0-9a-f]{64}", effect[key]) is None
                    for key in ("decision_sha256", "binding_sha256")
                )
            ):
                return None
            projected = {
                "claim_id": assertion["claim_id"],
                "original_status": effect["original_status"],
                "original_reason_code": effect["original_reason_code"],
                "host_status": "unverified",
                "code": effect["code"],
                "decision_sha256": effect["decision_sha256"],
                "binding_sha256": effect["binding_sha256"],
            }
            previous = found.setdefault(assertion["claim_id"], projected)
            if previous != projected:
                return None
        return list(found.values())

    overall = (
        effects(
            [*details.get("assertions", []), *details.get("unsupported_claims", [])]
        )
        if details.get("assessment_stage") == "host_validated"
        else None
    )
    partial = details.get("partial_assessment")
    partial_effects = (
        effects(partial.get("assertions", [])) if isinstance(partial, dict) else None
    )
    return {
        "semantic_conflict_policy": envelope["semantic_conflict_policy"],
        "semantic_conflict_identity": envelope["semantic_conflict_identity"],
        "semantic_conflict_count": len(overall) if overall is not None else None,
        "semantic_conflict_effects": overall,
        "partial_semantic_conflict_count": (
            len(partial_effects) if partial_effects is not None else None
        ),
        "partial_semantic_conflict_effects": partial_effects,
    }


def transport_diagnostic(run_id):
    """Read only a receipt bound to the exact archive, never free error prose."""
    from litagent.llm.responses import receipt_diagnostic

    if not isinstance(run_id, str) or Path(run_id).name != run_id:
        return None
    archive_path = Path("artifacts/runs", run_id + ".json")
    journal_path = Path("artifacts/targeted-repairs", run_id + ".json.writer-v2.json")
    if not archive_path.exists() or not journal_path.exists():
        return None
    try:
        archive = json.loads(archive_path.read_text())
        journal = json.loads(journal_path.read_text())
        if archive.get("run_id") != run_id:
            return None
        ends = {
            e["data"]["operation_id"]: e["data"]
            for e in archive["events"]
            if e["event"] == "llm.complete"
        }
        for event in reversed(archive["events"]):
            if (
                event["event"] != "llm.start"
                or event["data"].get("provider_api") != "responses"
            ):
                continue
            start = event["data"]
            end = ends.get(start["operation_id"])
            if not end or not end.get("provider_error"):
                continue
            _, receipt, _, _ = bound_native_receipt(start, end, journal)
            return {
                **receipt_diagnostic(receipt),
                "usage_available": end.get("usage_available") is True,
                "archive_sha256": hashlib.sha256(archive_path.read_bytes()).hexdigest(),
                "journal_sha256": hashlib.sha256(journal_path.read_bytes()).hexdigest(),
            }
    except (KeyError, ValueError, TypeError, OSError):
        return None
    return None


def repair_accounting(run_id):
    """Project persisted shared attempts, including failures before a report exists."""
    from litagent.quality import load_targeted_repairs

    if not isinstance(run_id, str) or Path(run_id).name != run_id:
        raise ValueError("frozen_repair_identity_invalid")
    path = Path("artifacts/targeted-repairs", run_id + ".json")
    if not path.exists():
        return None
    state = load_targeted_repairs(path, run_id)
    return {
        "repairs": len(state["attempts"]),
        "repair_history": [
            {
                key: (safe_reason(a[key]) if key == "error_code" else a[key])
                for key in ("index", "stage", "status", "kind", "error_code")
                if key in a
            }
            for a in state["attempts"]
        ],
    }


def writer_protocol_diagnostic(run_id, failure_code):
    """Locate the terminal unresolved JSON error without borrowing recovered calls."""
    if (
        failure_code != "writer_json_invalid"
        or not isinstance(run_id, str)
        or Path(run_id).name != run_id
    ):
        return None
    path = Path("artifacts/targeted-repairs", run_id + ".json.writer-v2.json")
    try:
        journal = json.loads(path.read_text())
    except (OSError, ValueError):
        return None
    if not isinstance(journal, dict) or not isinstance(journal.get("calls"), dict):
        return None
    calls = journal["calls"]
    failure = journal.get("failure", {})
    if failure is None:
        failure = {}
    if not isinstance(failure, dict) or failure.get("code") not in (None, failure_code):
        return None

    def unresolved(call):
        """Retain only the exact failure family that the view is explaining."""
        return (
            isinstance(call, dict)
            and call.get("resolved_by") is None
            and isinstance(call.get("failure"), dict)
            and call["failure"].get("code") == failure_code
        )

    # A failed repair has its own call; failed_call may still name the original.
    if "call" in failure or "failed_call" in failure:
        name = failure.get("call") or failure.get("failed_call")
        if not isinstance(name, str) or not unresolved(calls.get(name)):
            return None
    else:
        candidates = [name for name, call in calls.items() if unresolved(call)]
        # Historical object order cannot prove which of multiple failures is current.
        if len(candidates) != 1:
            return None
        name = candidates[0]
    if not isinstance(name, str) or not re.fullmatch(r"[a-z][a-z0-9:_-]{0,127}", name):
        return None
    call = calls[name]
    if not isinstance(call.get("stage"), str) or call["stage"] not in {
        "synthesis",
        "summary",
        "repair",
        "reviewer",
    }:
        return None
    if call.get("finish_reason") not in (
        None,
        "stop",
        "length",
        "invalid",
        "content_filter",
    ):
        return None
    raw = call.get("raw")
    records = call["failure"].get("diagnostics")
    if not isinstance(raw, str) or not isinstance(records, list) or not records:
        return None
    record = records[0]
    if not isinstance(record, dict):
        return None
    offset = record.get("offset")
    if type(offset) is not int or not 0 <= offset <= len(raw):
        return None
    try:
        raw_sha = hashlib.sha256(raw.encode()).hexdigest()
    except UnicodeError:
        return None
    if call.get("raw_sha256") != raw_sha or record.get("raw_sha256") != raw_sha:
        return None
    try:
        json.loads(raw)
    except json.JSONDecodeError as error:
        if error.pos != offset:
            return None
        line, column = error.lineno, error.colno
    except (ValueError, RecursionError):
        return None
    else:
        return None
    return {
        "call": name,
        "stage": call.get("stage"),
        "raw_response_sha256": raw_sha,
        "offset": offset,
        "line": line,
        "column": column,
        "finish_reason": call.get("finish_reason"),
        "scientific_assessment": "not_run",
    }


def syntax_normalizations(run_id, policy):
    """Project bound normalization receipts without exposing received prose."""
    from litagent.writing_v2 import BODY, EOF_CONTAINER_POLICY

    if (
        policy != EOF_CONTAINER_POLICY
        or not isinstance(run_id, str)
        or not run_id
        or Path(run_id).name != run_id
    ):
        return None
    try:
        journal = json.loads(
            Path(
                "artifacts/targeted-repairs", run_id + ".json.writer-v2.json"
            ).read_text()
        )
        if journal.get("syntax_policy") != policy or not isinstance(
            journal.get("calls"), dict
        ):
            return None
        result = []
        for name, entry in journal["calls"].items():
            receipt = entry.get("syntax_normalization")
            if receipt is None:
                continue
            raw, candidate = entry["raw"], entry["normalized_raw"]
            suffix = receipt["suffix"]
            if (
                receipt["policy"] != policy
                or not re.fullmatch(r"[\]}]{1,64}", suffix)
                or candidate != raw + suffix
                or hashlib.sha256(raw.encode()).hexdigest() != receipt["raw_sha256"]
                or entry["raw_sha256"] != receipt["raw_sha256"]
                or hashlib.sha256(candidate.encode()).hexdigest()
                != receipt["candidate_sha256"]
                or any(
                    receipt[key] != entry[key]
                    for key in (
                        "schema_sha256",
                        "provider_schema_sha256",
                        "input_sha256",
                    )
                )
                or receipt["coverage_complete"] is not True
                or receipt["scientific_assessment"] != "not_verified"
                or not isinstance(receipt["units"], list)
                or not 1 <= len(receipt["units"]) <= 128
                or name not in {*(f"body:{k}" for k in BODY), "summary_initial"}
                or any(
                    not isinstance(receipt.get(key), str)
                    or not re.fullmatch(r"[0-9a-f]{64}", receipt[key])
                    for key in (
                        "raw_sha256",
                        "candidate_sha256",
                        "schema_sha256",
                        "provider_schema_sha256",
                        "semantic_sha256",
                    )
                )
            ):
                return None
            result.append(
                {
                    "call": name,
                    **{
                        key: receipt[key]
                        for key in (
                            "policy",
                            "raw_sha256",
                            "candidate_sha256",
                            "schema_sha256",
                            "provider_schema_sha256",
                            "semantic_sha256",
                        )
                    },
                    "unit_count": len(receipt["units"]),
                    "scientific_assessment": "not_verified",
                }
            )
        return result
    except (OSError, ValueError, KeyError, TypeError, AttributeError, UnicodeError):
        return None


def export(root, state, manifest):
    """Publish one sanitized data source with separate execution and quality outcomes."""
    from litagent.safety.planned_accounting import _digest, _envelope

    executed = state["cases"]
    budget_plans = manifest.get("budget_plans")
    budget_plans = budget_plans if isinstance(budget_plans, dict) else {}

    def display_plan(plan):
        """Check frozen display metadata without requiring admission or live semantics."""
        try:
            if (
                not isinstance(plan, dict)
                or plan.get("contract")
                not in {"planned-budget-v" + str(number) for number in range(2, 8)}
                or plan.get("status") not in {"admitted", "preflight_rejected"}
                or any(
                    type(plan.get(key)) is not int or plan[key] < 0
                    for key in ("required_tokens", "planned_tokens", "maximum_tokens")
                )
                or "reason" not in plan
                or plan.get("identity")
                != _digest(
                    {key: value for key, value in plan.items() if key != "identity"}
                )
            ):
                return False
            if plan["contract"] != "planned-budget-v2" and (
                type(plan.get("repair_limit")) is not int
                or not 0 <= plan["repair_limit"] <= 2
            ):
                return False
            if plan["contract"] not in {"planned-budget-v2", "planned-budget-v3"}:
                _envelope(plan.get("envelope"))
            if "joint_round_escrow" in plan:
                from litagent.safety.planned_accounting import _plan

                _plan(plan)
            return True
        except (ValueError, TypeError, KeyError, AttributeError):
            return False

    display_plans = {
        case: plan for case, plan in budget_plans.items() if display_plan(plan)
    }
    writing_profile = profile_description(
        manifest.get("writing_profile", DISABLED_PROFILE),
        manifest.get("config", {}),
        writing_plan=manifest.get("writing_plan"),
    )
    if "writer_wire_contract" in manifest:
        # Read historical metadata verbatim rather than inferring today's wire.
        writing_profile["writer_wire_contract"] = manifest["writer_wire_contract"]
    if (
        manifest.get("writing_profile")
        in (
            FLAT_PROFILE,
            CAPACITY_PROFILE,
            ROUND_PROFILE,
        )
        and display_plans
    ):
        scope = next(iter(display_plans.values())).get("envelope")
        spec = copy.deepcopy(writing_plan_spec(manifest.get("writing_plan")))
        if (
            spec is not None
            and scope
            and scope["contract"] == "flat-writer-envelope-v1"
        ):
            # Frozen capacity is display data, not authority to resume this writer.
            targets = copy.deepcopy(spec["body_target_units"])
            if "capacity" in scope:
                weights = dict(targets)
                targets = {key: 1 for key in weights}
                for _ in range(scope["capacity"]["target_body_units"] - len(targets)):
                    key = max(weights, key=lambda item: weights[item] / targets[item])
                    targets[key] += 1
            spec.update(
                body_target_units=targets,
                body_max_units=scope["body_max_units"],
                body_section_max_units={
                    key: scope["body_max_units"] - 4 for key in targets
                },
                summary_section_max_units=scope["summary_max_units_per_section"],
                report_max_units=scope["body_max_units"]
                + 2 * scope["summary_max_units_per_section"],
                atomic_claim_capacity=scope["model_assertions"],
            )
            writing_profile["writing_plan"] = spec
            writing_profile["chapter_evidence_lists"] = "recommendations"
    rows = [
        next(
            (dict(r) for r in executed if r["case_id"] == case_id),
            {
                "case_id": case_id,
                "run_id": None,
                "status": "not_run",
                "generated": False,
                "reviewer_completed": False,
                "evaluation": "not_run",
                "publishable": False,
                "supported": None,
                "unsupported": None,
                "unverified": None,
                "requests": 0,
                "known_tokens": 0,
                "unknown_hold": 0,
                "failure_code": (
                    "not_started_after_stop"
                    if state["status"] in {"stopped", "interrupted"}
                    else "not_started"
                ),
            },
        )
        for case_id, _ in CASES
    ]

    def cost_unavailable(row, reason_code):
        """Project untrusted cost records without granting accounting authority."""

        def reported(key, *, positive=False):
            value = row.get(key)
            return (
                value
                if type(value) is int and value >= (1 if positive else 0)
                else None
            )

        was_passed = row.get("publishable") is True or row.get("status") == "passed"
        row["cost_diagnostic"] = {
            "status": "unavailable",
            "reason_code": reason_code,
            "reported_requests": reported("requests"),
            "reported_known_tokens": reported("known_tokens"),
            "reported_unknown_hold": reported("unknown_hold", positive=True),
        }
        row.pop("budget_snapshot", None)
        row.update(
            requests=None,
            known_tokens=None,
            unknown_hold=None,
            request_hold=None,
            usage_complete=False,
            publishable=False,
        )
        if was_passed:
            row["status"] = "blocked"

    executed_ids = {row.get("case_id") for row in executed}
    for row in rows:
        snapshot = row.get("budget_snapshot")
        pinned_plan = budget_plans.get(row.get("case_id"))
        if (
            row.get("case_id") in executed_ids
            and (
                isinstance(pinned_plan, dict)
                or manifest.get("writing_profile") in PLANNED_PROFILES
            )
        ) or (isinstance(snapshot, dict) and snapshot.get("schema_version") == 2):
            if not isinstance(pinned_plan, dict):
                cost_unavailable(row, "budget_plan_missing")
                continue
            if snapshot is None or snapshot == {}:
                cost_unavailable(row, "budget_snapshot_missing")
                continue
            if (
                not isinstance(snapshot, dict)
                or type(snapshot.get("schema_version")) is not int
                or snapshot["schema_version"] != 2
            ):
                cost_unavailable(row, "budget_snapshot_invalid")
                continue
            if snapshot.get("plan") != pinned_plan:
                cost_unavailable(row, "budget_plan_conflict")
                continue
            try:
                row["unknown_hold"] = historical_unknown_token_hold(
                    snapshot, row.get("known_tokens"), expected_plan=pinned_plan
                )
            except ValueError:
                cost_unavailable(row, "budget_snapshot_invalid")
                continue
            if row["unknown_hold"] or snapshot.get("usage_uncertain"):
                row.update(
                    usage_complete=False,
                    publishable=False,
                    failure_family="interrupted",
                    failure_code="usage_unknown",
                )
    for row in rows:
        if not row.get("run_id") or Path(row["run_id"]).name != row["run_id"]:
            continue
        archive_path = Path("artifacts/runs", row["run_id"] + ".json")
        if not archive_path.exists() or not row.get("report_sha256"):
            continue
        archive = json.loads(archive_path.read_text())
        report = archive.get("report") or {}
        survey = report.get("survey")
        if (
            archive.get("run_id") != row["run_id"]
            or not isinstance(survey, str)
            or hashlib.sha256(survey.encode()).hexdigest() != row["report_sha256"]
        ):
            continue
        diagnostic = evaluation_diagnostic(report)
        if diagnostic:
            row["evaluation_diagnostic"] = diagnostic
        row.update(
            semantic_conflict_diagnostic(
                report, budget_plans.get(row["case_id"], {}).get("envelope", {})
            )
        )
        attempts = report.get("metadata", {}).get("evidence_rewrite_attempts", [])
        not_sent = {
            evaluation_attempt_label(a, i): a
            for i, a in enumerate(attempts)
            if a.get("attempted") is False
        }
        if not_sent:
            row["evaluation_history"] = [
                h
                for h in row.get("evaluation_history", [])
                if h.get("stage") not in not_sent
            ]
            row["repair_preflight_failures"] = [
                {"attempted": False, "reason_code": safe_reason(a.get("reason_code"))}
                for a in not_sent.values()
            ]
    for row in rows:
        diagnostic = transport_diagnostic(row.get("run_id"))
        if diagnostic:
            row["transport_diagnostic"] = diagnostic
        for key in ("http_status", "provider_error_code", "rejected_parameter"):
            row[key] = (diagnostic or {}).get(key)
    for row in rows:
        row.pop("protocol_diagnostic", None)
        diagnostic = writer_protocol_diagnostic(
            row.get("run_id"), row.get("failure_code")
        )
        if diagnostic:
            row["protocol_diagnostic"] = diagnostic
        row["protocol_call"] = (diagnostic or {}).get("call")
        row["protocol_raw_response_sha256"] = (diagnostic or {}).get(
            "raw_response_sha256"
        )
        row["protocol_offset"] = (diagnostic or {}).get("offset")

    def display(value):
        """Render absent statistics as unavailable rather than scientific zero."""
        return (
            "—"
            if value is None
            else "Yes" if value is True else "No" if value is False else str(value)
        )

    def reason(row, *, evaluation=False):
        """Keep the terminal cause distinct from upstream evaluator diagnostics."""
        upstream = row.get("evaluation_diagnostic", {}).get("reason_code")
        terminal = row.get("failure_code")
        cause = (
            (upstream or terminal or "") if evaluation else (terminal or upstream or "")
        )
        if row.get("http_status"):
            cause += " / HTTP " + str(row["http_status"])
            cause += " / " + (row.get("provider_error_code") or "unavailable")
            if row.get("rejected_parameter"):
                cause += " / " + row["rejected_parameter"]
        for problem in row.get("summary_progress", {}).get("failures", []):
            cause += (
                " / summary "
                + problem["section"]
                + ": "
                + str(problem["actual"])
                + "/"
                + str(problem["maximum"])
            )
        if not evaluation and "cost_diagnostic" in row:
            cause += (
                (" / " if cause else "")
                + "cost unavailable: "
                + row["cost_diagnostic"]["reason_code"]
            )
        return cause

    for row in rows:
        row["writing_profile"] = writing_profile["id"]
        row["writing_plan"] = manifest.get("writing_plan")
        row["syntax_policy"] = writing_profile.get("syntax_policy")
        receipts = syntax_normalizations(row.get("run_id"), row["syntax_policy"])
        row["syntax_normalizations"] = receipts
        row["syntax_normalization_count"] = (
            len(receipts) if receipts is not None else None
        )
        row["provider_api"] = writing_profile.get("provider_api", "chat")
        row["reasoning_effort"] = writing_profile.get("reasoning_effort")
        row["shape_recovery_contract"] = writing_profile.get("shape_recovery_contract")
        row["reemission_contract"] = writing_profile.get("reemission_contract")
        row["writer_prompt_contract"] = manifest.get("writer_prompt_contract")
        if "writer_wire_contract" in manifest:
            row["writer_wire_contract"] = manifest["writer_wire_contract"]
        for key in (
            "generation_thinking_mode",
            "reviewer_thinking_mode",
            "evaluation_thinking_mode",
            "output_limit_parameter",
            "generation_output_tokens",
            "reviewer_output_tokens",
            "evaluation_output_tokens",
        ):
            row[key] = writing_profile[key]
        row["evaluation_reason"] = reason(row, evaluation=True)
        row["terminal_reason"] = reason(row)
        for key in ("expected_assertions", "processed_assertions"):
            row[key] = row.get("evaluation_diagnostic", {}).get(key)

    def usage_total(key):
        """Never turn unavailable executed costs into a measured zero aggregate."""
        values = [row.get(key, 0) for row in rows]
        return None if any(value is None for value in values) else sum(values)

    public = {
        "version": VERSION,
        "view_version": (
            "frozen-writing-view-v13"
            if manifest.get("writing_profile") == ROUND_PROFILE
            else (
                "frozen-writing-view-v12"
                if manifest.get("writing_profile") in (CAPACITY_PROFILE, ROUND_PROFILE)
                else "frozen-writing-view-v11"
            )
        ),
        "suite_id": root.name,
        "identity": manifest["identity"],
        "source_commit": manifest["source_commit"],
        "contract": CONTRACT,
        "writer_prompt_contract": manifest.get("writer_prompt_contract"),
        "writing_profile": writing_profile,
        "state": state["status"],
        "stop_reason": state.get("stop_reason"),
        "planned": 3,
        "run": len(executed),
        "generated": sum(bool(r.get("generated")) for r in rows),
        "evaluation_completed": sum(r.get("evaluation") == "completed" for r in rows),
        "passed": sum(r.get("publishable", False) for r in rows),
        "not_run": 3 - len(executed),
        "judge": "disabled",
        "jev": "disabled",
        "human_calibration": "not_validated",
        "cases": rows,
        "historical_usage": manifest["history"],
        "new_usage": {
            "requests": usage_total("requests"),
            "known_tokens": usage_total("known_tokens"),
            "unknown_hold": usage_total("unknown_hold"),
            "request_hold": usage_total("request_hold"),
        },
        "limits": manifest["limits"],
    }
    if "writer_wire_contract" in manifest:
        public["writer_wire_contract"] = manifest["writer_wire_contract"]
    semantic_rows = [row for row in rows if "semantic_conflict_policy" in row]
    if "semantic_conflict_policy" in manifest:
        public.update(
            {
                key: manifest[key]
                for key in ("semantic_conflict_policy", "semantic_conflict_identity")
            }
        )
    if budget_plans:
        public["budget_plans"] = display_plans
        public["source_fingerprint"] = digest(manifest["source_hashes"])
        public["source_state"] = manifest.get("source_state", "frozen_files")
        for row in rows:
            plan = display_plans.get(row["case_id"])
            if plan is None:
                continue
            row.update(
                budget_status=plan["status"],
                budget_required_tokens=plan["required_tokens"],
                budget_planned_tokens=plan["planned_tokens"],
                budget_reason=plan["reason"],
            )
            if "joint_round_escrow" in plan:
                domain = plan["joint_round_escrow"]["assessment"]
                row.update(
                    budget_round_max_requests=domain["max_requests"],
                    budget_assessment_request_cap=domain["request_cap"],
                    budget_round_reserved_tokens=domain["reserved_tokens"],
                )
            if plan["contract"] in (
                "planned-budget-v3",
                "planned-budget-v4",
                "planned-budget-v6",
                "planned-budget-v7",
            ):
                row["budget_locked_repairs"] = plan["repair_limit"]
            if plan["contract"] in (
                "planned-budget-v4",
                "planned-budget-v6",
                "planned-budget-v7",
            ):
                row.update(
                    budget_body_units=plan["envelope"]["body_max_units"],
                    budget_model_assertions=plan["envelope"]["model_assertions"],
                    budget_assessment_batches=plan["envelope"]["assessment_batches"],
                )
            if state["status"] == "preflight_rejected":
                row.update(
                    status="preflight_rejected",
                    failure_code=plan["reason"],
                    failure_family="budget",
                )
    if manifest.get("writing_profile") in (CAPACITY_PROFILE, ROUND_PROFILE):
        for row in rows:
            capacity = row.get("capacity") or {}
            plan = display_plans.get(row["case_id"], {})
            envelope = plan.get("envelope", {})
            row.update(
                capacity_status=capacity.get("status"),
                capacity_initial_assertions=capacity.get(
                    "initial_assertions",
                    envelope.get("capacity", {}).get("initial_assertions"),
                ),
                capacity_effective_assertions=capacity.get(
                    "effective_assertions", envelope.get("model_assertions")
                ),
                capacity_actual_assertions=capacity.get("actual_assertions"),
                capacity_batches=capacity.get("actual_batches"),
            )
            if manifest.get("writing_profile") == ROUND_PROFILE:
                row.update(
                    capacity_round=(capacity.get("round") or {}).get("number"),
                    capacity_branch=(capacity.get("round") or {}).get("mode"),
                    capacity_assessment_state=(capacity.get("round") or {}).get(
                        "status"
                    ),
                    capacity_target_assertions=capacity.get("target_assertions"),
                    capacity_target_batches=capacity.get("target_batches"),
                    capacity_assessment_reservation=capacity.get(
                        "assessment_reserved_tokens"
                    ),
                )
    out = Path("benchmarks/survey/results", root.name)
    out.mkdir(parents=True, exist_ok=True)
    atomic_json(out / "benchmark.json", public)
    keys = [
        "case_id",
        "writing_profile",
        "writing_plan",
        "provider_api",
        "reasoning_effort",
        "shape_recovery_contract",
        "reemission_contract",
        "syntax_policy",
        "syntax_normalization_count",
        "writer_prompt_contract",
        "reasoning_tokens",
        "reasoning_usage_complete",
        "reasoning_usage_available_requests",
        "generation_thinking_mode",
        "reviewer_thinking_mode",
        "evaluation_thinking_mode",
        "output_limit_parameter",
        "generation_output_tokens",
        "reviewer_output_tokens",
        "evaluation_output_tokens",
        "run_id",
        "status",
        "generated",
        "evaluation",
        "repairs",
        "publishable",
        "supported",
        "unsupported",
        "unverified",
        "requests",
        "known_tokens",
        "unknown_hold",
        "elapsed_ms",
        "report_sha256",
        "failure_family",
        "failure_code",
        "terminal_reason",
        "protocol_call",
        "protocol_raw_response_sha256",
        "protocol_offset",
        "http_status",
        "provider_error_code",
        "rejected_parameter",
        "evaluation_reason",
        "expected_assertions",
        "processed_assertions",
    ]
    if manifest.get("closeout_profile") is not None:
        keys.extend(
            [
                "reviewer_completed",
                "evaluation_valid",
                "inventory_recoveries",
                "prose_repairs",
            ]
        )
    if manifest.get("writing_profile") in (CAPACITY_PROFILE, ROUND_PROFILE):
        keys.extend(
            [
                "capacity_status",
                "capacity_initial_assertions",
                "capacity_effective_assertions",
                "capacity_actual_assertions",
                "capacity_batches",
                "capacity_revisions",
                "capacity_refinements",
            ]
        )
    if "writer_wire_contract" in manifest:
        keys.insert(keys.index("writer_prompt_contract") + 1, "writer_wire_contract")
    if manifest.get("writing_profile") == ROUND_PROFILE:
        keys.extend(
            [
                "capacity_round",
                "capacity_branch",
                "capacity_assessment_state",
                "capacity_target_assertions",
                "capacity_target_batches",
                "capacity_assessment_reservation",
            ]
        )
    finding_rows = [r for r in rows if located_findings_enabled(r)]
    finding_fields = [
        "assessment_stage",
        "inventory_status",
        "partial_assessment_stage",
        "partial_supported",
        "partial_unsupported",
        "partial_unverified",
    ]
    if finding_rows:
        keys.extend(["global_findings_contract", *finding_fields])
    if "semantic_conflict_policy" in manifest:
        keys.extend(
            [
                "semantic_conflict_policy",
                "semantic_conflict_identity",
                "semantic_conflict_count",
                "semantic_conflict_effects",
                "partial_semantic_conflict_count",
                "partial_semantic_conflict_effects",
            ]
        )
    if any("summary_progress" in row for row in rows):
        keys.append("summary_progress")
    cost_rows = [row for row in rows if "cost_diagnostic" in row]
    if cost_rows:
        keys.append("cost_diagnostic")
    if display_plans:
        keys.extend(
            [
                "budget_status",
                "budget_required_tokens",
                "budget_planned_tokens",
                "budget_reason",
            ]
        )
        if manifest.get("writing_profile") in (
            ADAPTIVE_PROFILE,
            BUDGETED_PROFILE,
            FLAT_PROFILE,
            CAPACITY_PROFILE,
            ROUND_PROFILE,
        ):
            keys.append("budget_locked_repairs")
        if manifest.get("writing_profile") in (
            BUDGETED_PROFILE,
            FLAT_PROFILE,
            CAPACITY_PROFILE,
            ROUND_PROFILE,
        ):
            keys.extend(
                [
                    "budget_body_units",
                    "budget_model_assertions",
                    "budget_assessment_batches",
                ]
            )
    buf = io.StringIO()
    if any("joint_round_escrow" in plan for plan in display_plans.values()):
        keys.extend(
            [
                "budget_round_max_requests",
                "budget_assessment_request_cap",
                "budget_round_reserved_tokens",
            ]
        )
    if manifest.get("inventory_recovery") is True:
        keys.append("evaluation_history")
    writer = csv.DictWriter(buf, fieldnames=keys, extrasaction="ignore")
    writer.writeheader()

    def csv_row(row):
        """Serialize optional diagnostics from the same public row projection."""
        projected = (
            {
                **row,
                **(
                    {
                        "summary_progress": json.dumps(
                            row["summary_progress"], ensure_ascii=False
                        )
                    }
                    if "summary_progress" in row
                    else {}
                ),
                "evaluation_history": json.dumps(
                    row.get("evaluation_history", []), ensure_ascii=False
                ),
            }
            if manifest.get("inventory_recovery") is True
            else dict(row)
        )
        if "cost_diagnostic" in row:
            projected["cost_diagnostic"] = json.dumps(
                row["cost_diagnostic"], ensure_ascii=False
            )
        for key in ("semantic_conflict_effects", "partial_semantic_conflict_effects"):
            if key in row:
                projected[key] = json.dumps(row[key], ensure_ascii=False)
        return projected

    writer.writerows(csv_row(row) for row in rows)
    (out / "benchmark.csv").write_text(buf.getvalue())
    md = f"# 三域冻结正文验收\n\n计划3题，运行{public['run']}题，生成{public['generated']}题，Evaluation完成{public['evaluation_completed']}题，严格交付{public['passed']}/3。状态：{state['status']}。\n\n源码 `{manifest['source_commit']}`，协议 `{CONTRACT}`；Lite审阅/评估，Pro写作，Judge/Jev关闭。固定证据到交付，不包含新检索或提取，不是端到端检索Benchmark。人工校准未验证，不报告科学准确率。\n\n|题目|生成|Evaluation|严格交付|支持/不支持/未核验|原因|\n|---|---|---|---|---|---|\n"
    md += "".join(
        f"|{r['case_id']}|{display(r.get('generated'))}|{r.get('evaluation')}|{display(r.get('publishable'))}|{display(r.get('supported'))}/{display(r.get('unsupported'))}/{display(r.get('unverified'))}|{reason(r)}|\n"
        for r in rows
    )
    md += "\n终态与上游诊断 / Terminal and upstream diagnostics\n\n|Case|Terminal|Evaluation|\n|---|---|---|\n"
    md += "".join(
        f"|{r['case_id']}|{r['terminal_reason'] or '—'}|{r['evaluation_reason'] or '—'}|\n"
        for r in rows
    )
    if cost_rows:
        md += "\n费用证据不可用 / Cost evidence unavailable\n\n下列仅为未经认证的原记录，不作为已核实费用或后续请求授权。Reported records below are unauthenticated, not verified charges or authorization.\n\n|Case|Reason|Reported requests|Reported known tokens|Reported positive unknown hold|\n|---|---|---|---|---|\n"
        md += "".join(
            f"|{row['case_id']}|{row['cost_diagnostic']['reason_code']}|{display(row['cost_diagnostic']['reported_requests'])}|{display(row['cost_diagnostic']['reported_known_tokens'])}|{display(row['cost_diagnostic']['reported_unknown_hold'])}|\n"
            for row in cost_rows
        )
    summary_rows = [r for r in rows if "summary_progress" in r]
    if summary_rows:
        md += "\n摘要契约 / Summary contract\n\n|Case|Body chapters received|Summary completed|Units per section|\n|---|---|---|---|\n"
        md += "".join(
            f"|{r['case_id']}|{r['summary_progress']['body_chapters_completed']}/5|{display(r['summary_progress']['summary_completed'])}|1–{r['summary_progress']['constraints']['max_units']}|\n"
            for r in summary_rows
        )
    if manifest.get("writing_profile") == ROUND_PROFILE:
        md += "\n逐轮分配（预留不是费用；superseded_unverified 表示旧稿未核验） / Round allocations\n\n|Case|Round|Branch|Original assessment|Target assertions/batches|Current reservation|\n|---|---|---|---|---|---|\n"
        md += "".join(
            f"|{r['case_id']}|{display(r['capacity_round'])}|{display(r['capacity_branch'])}|{display(r['capacity_assessment_state'])}|{display(r['capacity_target_assertions'])}/{display(r['capacity_target_batches'])}|{display(r['capacity_assessment_reservation'])}|\n"
            for r in rows
        )
    if finding_rows:
        md += "\n部分诊断非最终分数 / Partial diagnostics are not final scores\n\n|Case|Assessment stage|Inventory|Partial stage|Supported/unsupported/unverified|\n|---|---|---|---|---|\n"
        md += "".join(
            f"|{r['case_id']}|{display(r.get('assessment_stage'))}|{display(r.get('inventory_status'))}|{display(r.get('partial_assessment_stage'))}|{display(r.get('partial_supported'))}/{display(r.get('partial_unsupported'))}/{display(r.get('partial_unverified'))}|\n"
            for r in finding_rows
        )
    if semantic_rows:
        md += "\n模型原判与宿主未核验 / Original model decision and host unverified effect\n\n语义冲突不贡献支持数量；部分诊断非最终分数。Semantic conflicts add no supported assertion; partial diagnostics are not final scores.\n\n|Case|Scope|Claim|Original model status|Original reason|Host effect|Code|\n|---|---|---|---|---|---|---|\n"
        md += "".join(
            f"|{row['case_id']}|{scope}|{effect['claim_id']}|{effect['original_status']}|{effect['original_reason_code']}|unverified|{effect['code']}|\n"
            for row in semantic_rows
            for scope, key in (
                ("overall", "semantic_conflict_effects"),
                ("partial", "partial_semantic_conflict_effects"),
            )
            for effect in (row.get(key) or [])
        )
    if manifest.get("closeout_profile") is not None:
        md += "\n收尾配置：清单最多复核一次、两次共享修复。复核不代表正文修改，完整核验不代表严格通过。\n\n|Case|Reviewer|Valid Evaluation|Inventory rechecks|Prose repairs|\n|---|---|---|---|---|\n"
        md += "".join(
            f"|{r['case_id']}|{display(r.get('reviewer_completed'))}|{display(r.get('evaluation_valid'))}|{display(r.get('inventory_recoveries'))}|{display(r.get('prose_repairs'))}|\n"
            for r in rows
        )
    if manifest.get("inventory_recovery") is True:
        md += "\n核验过程（未完整核验的数量仅为诊断） / Evaluation history\n\n|Case|Stage|Support/unsupported/unverified|Complete|\n|---|---|---|---|\n"
        md += "".join(
            f"|{r['case_id']}|{h['stage']}|{display(h.get('supported'))}/{display(h.get('unsupported'))}/{display(h.get('unverified'))}|{display(h.get('complete'))}|\n"
            for r in rows
            for h in r.get("evaluation_history", [])
        )
    md += (
        "\n冻结模式 / Frozen profile: `"
        + writing_profile["id"]
        + "`。\n\n```json\n"
        + json.dumps(writing_profile, ensure_ascii=False, indent=2)
        + "\n```\n"
    )
    md += (
        "\nWriter prompt: "
        + (manifest.get("writer_prompt_contract") or "unavailable")
        + "\n"
    )
    if "writer_wire_contract" in manifest:
        md += "\nWriter wire: " + str(manifest["writer_wire_contract"]) + "\n"
    if any(r.get("failure_code") == "writer_json_invalid" for r in rows):
        md += "\n终态语法定位 / Terminal syntax location\n\n|Case|Call|Raw SHA|Offset|\n|---|---|---|---|\n"
        md += "".join(
            f"|{r['case_id']}|{display(r['protocol_call'])}|{display(r['protocol_raw_response_sha256'])}|{display(r['protocol_offset'])}|\n"
            for r in rows
            if r.get("failure_code") == "writer_json_invalid"
        )
    if display_plans:
        md += "\n## 预算准入 / Budget admission\n\n| Case | Status | Required reservation | Planned allowance | Reason |\n|---|---|---:|---:|---|\n"
        for case_id, plan in display_plans.items():
            md += f"| {case_id} | {plan['status']} | {plan['required_tokens']} | {plan['planned_tokens']} | {plan['reason'] or '—'} |\n"
        joint_plans = {
            case: plan["joint_round_escrow"]["assessment"]
            for case, plan in display_plans.items()
            if "joint_round_escrow" in plan
        }
        if joint_plans:
            md += "\n每轮锁定核验资源域 / Locked assessment domain per round\n\n|Case|Maximum requests|Per-request cap|Round total reservation|\n|---|---:|---:|---:|\n"
            md += "".join(
                f"|{case}|{domain['max_requests']}|{domain['request_cap']}|{domain['reserved_tokens']}|\n"
                for case, domain in joint_plans.items()
            )
            md += "\n后续修复轮保留同一资源域；完整候选仍须发送前检查。预留不是实际费用。\n"
        if manifest.get("writing_profile") in (
            ADAPTIVE_PROFILE,
            BUDGETED_PROFILE,
            FLAT_PROFILE,
            CAPACITY_PROFILE,
            ROUND_PROFILE,
        ):
            md += (
                "\n计划修复路径（仅准入后锁定） / Planned repairs (locked only after admission): "
                + ", ".join(
                    f"{case}={plan['repair_limit']}/2"
                    for case, plan in display_plans.items()
                )
                + "。首次完整 Evaluation 不可省略；0次修复不代表科学通过。\n"
            )
        md += "\n预留不是实际费用。无法容纳完整流程时，三域均不启动付费生成。\n"
        if manifest.get("writing_profile") in (
            BUDGETED_PROFILE,
            FLAT_PROFILE,
            CAPACITY_PROFILE,
            ROUND_PROFILE,
        ):
            md += (
                "\n整体容量（仅准入后锁定） / Content capacity (locked only after admission): "
                + ", ".join(
                    f"{case}: body≤{p['envelope']['body_max_units']}, assertions≤{p['envelope']['model_assertions']}, batches≤{p['envelope']['assessment_batches']}"
                    for case, p in display_plans.items()
                )
                + "。全部实际内容仍须核验；越界停止，不截断。\n"
            )
    md += f"\n新增已知用量{display(public['new_usage']['known_tokens'])} tokens，未知保守占用{display(public['new_usage']['unknown_hold'])}，{display(public['new_usage']['requests'])}请求。历史用量独立保留，不重复计算提取费用。失败和未运行题不移出分母，不追加到通过。\n\n[JSON](benchmark.json) · [CSV](benchmark.csv) · [HTML](benchmark.html)\n"
    md += "\n共享修复次数 / Shared repair attempts (limit 2; missing history unavailable)\n\n|Case|Attempts|\n|---|---|\n"
    md += "".join(f"|{r['case_id']}|{display(r.get('repairs'))}|\n" for r in rows)
    if writing_profile.get("syntax_policy"):
        md += "\n宿主语法归一化 / Host syntax normalization (scientific verification still required)\n\n|Case|Count|\n|---|---|\n"
        md += "".join(
            f"|{r['case_id']}|{display(r.get('syntax_normalization_count'))}|\n"
            for r in rows
        )
    if any("reasoning_usage_complete" in r for r in rows):
        md += "\n思考明细是 completion 的子集，不再次累计；缺失时不推算。\n\n|Case|Reasoning tokens|Details complete|Available requests / all requests|\n|---|---|---|---|\n"
        md += "".join(
            f"|{r['case_id']}|{display(r.get('reasoning_tokens'))}|{display(r.get('reasoning_usage_complete'))}|{display(r.get('reasoning_usage_available_requests'))}/{display(r.get('requests', 0))}|\n"
            for r in rows
        )
    (out / "benchmark.md").write_text(md)
    (out / "README.md").write_text(md)
    table = "".join(
        "<tr>"
        + "".join(
            "<td>" + html.escape(display(r.get(k))) + "</td>"
            for k in [
                "case_id",
                "generated",
                "evaluation",
                "repairs",
                "publishable",
                "supported",
                "unsupported",
                "unverified",
            ]
        )
        + (
            '<td><a href="http://127.0.0.1:8000/flow-demo?run=archive%3A'
            + r["run_id"]
            + '&view=quality">'
            + (
                "质量与原文 / Report"
                if r.get("report_sha256")
                else "运行诊断 / Diagnostics"
            )
            + "</a></td></tr>"
            if r.get("run_id")
            else "<td>未运行 / Not run</td></tr>"
        )
        for r in rows
    )
    details = "".join(
        "<details><summary>"
        + html.escape(r["case_id"])
        + " · "
        + html.escape(str(reason(r) or r["status"]))
        + "</summary><pre>"
        + html.escape(json.dumps(r, ensure_ascii=False, indent=2))
        + "</pre></details>"
        for r in rows
    )
    if manifest.get("closeout_profile") is not None:
        details += (
            "<section><h2>收尾修复 / Bounded closeout</h2><p>清单最多复核一次，两次共享修改；完整核验与严格交付分别记录。At most one inventory recheck within two shared attempts.</p>"
            + "".join(
                "<p>"
                + html.escape(str(r["case_id"]))
                + ": inventory rechecks "
                + str(r.get("inventory_recoveries", "—"))
                + ", prose repairs "
                + str(r.get("prose_repairs", "—"))
                + "</p>"
                for r in rows
            )
            + "</section>"
        )
    page = """<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>LitAgent · Frozen writing validation</title><style>*{box-sizing:border-box}body{font:16px/1.6 system-ui;background:#f4f6fa;color:#172c43;max-width:1100px;padding:20px;margin:auto}section,header{background:white;padding:20px;margin:16px 0;border:1px solid #dbe2ee;border-radius:12px}h1{font-size:1.7rem}.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(150px,1fr));gap:12px}.stat{background:#edf2f9;padding:14px}.scroll{overflow:auto}table{width:100%;border-collapse:collapse}th,td{padding:10px;border-bottom:1px solid #ccc;text-align:left}pre{white-space:pre-wrap;overflow-wrap:anywhere;font-size:.85em}button,a,summary{cursor:pointer}button{font:inherit;padding:8px}a{color:#135a9e}*:focus-visible{outline:3px solid #1679d2}summary{padding:12px 0}p{overflow-wrap:anywhere}@media(max-width:430px){body{padding:10px}section,header{padding:14px}h1{font-size:1.3rem}}</style><header><button id="locale">English / 中文</button><h1 id="title">三域冻结正文验收</h1><p id="scope">固定证据 → 正文 → Reviewer → Evaluation → 自动交付或失败归档；不重新检索。Judge/Jev关闭。</p><p id="profile"></p><p id="transport"></p><p id="syntax"></p></header><section><div id="stats" class="grid"></div><p id="note">自动评测；人工校准未验证。未核验不等于通过，协议完成不等于科学通过。失败及未运行题均保留在分母。</p><p id="status"></p></section><section class="scroll"><table><thead><tr><th>Case</th><th>Generated</th><th>Evaluation</th><th>Delivered</th><th>Supported</th><th>Unsupported</th><th>Unverified</th><th>Detail</th></tr></thead><tbody>__ROWS__</tbody></table></section><section><h2>诊断 / Diagnostics</h2>__DETAILS__</section><section><a href="benchmark.json">JSON</a> · <a href="benchmark.csv">CSV</a> · <a href="benchmark.md">Markdown</a><pre id="usage"></pre></section><script id="data" type="application/json">__DATA__</script><script>const d=JSON.parse(document.getElementById('data').textContent);let en=false;function draw(){document.documentElement.lang=en?'en':'zh-CN';document.getElementById('title').textContent=d.budget_plans&&d.state==='preflight_rejected'?(en?'Three-domain budget admission · offline review':'三域预算准入 · 离线审查'):(en?'Three-domain frozen writing validation':'三域冻结正文验收');document.getElementById('scope').textContent=en?'Frozen evidence → writing → Reviewer → Evaluation → automatic delivery or archived failure. No retrieval. Judge/Jev disabled.':'固定证据 → 正文 → Reviewer → Evaluation → 自动交付或失败归档；不重新检索。Judge/Jev关闭。';document.getElementById('note').textContent=en?'Automatic evaluation; no human calibration. Unverified is not passed. Protocol completion is not scientific acceptance. Failures and unrun cases remain in the denominator.':'自动评测；人工校准未验证。未核验不等于通过，协议完成不等于科学通过。失败及未运行题均保留在分母。';const t=d.cases.filter(r=>r.transport_diagnostic).map(r=>r.case_id+' · HTTP '+r.http_status+' · '+(r.provider_error_code||'—')+' · '+(r.rejected_parameter||'—')).join('; ');document.getElementById('transport').textContent=t?((en?'Provider diagnosis (usage unknown is not zero cost): ':'供应商诊断（未知用量不等于零费用）：')+t):'';const p=d.writing_profile;document.getElementById('syntax').textContent=p.syntax_policy?((en?'Host EOF container normalization; scientific verification still required: ':'宿主末尾容器闭合；仍须完整科学核验：')+p.syntax_policy):'';document.getElementById('profile').textContent=(en?'Frozen thinking profile: ':'冻结思考配置：')+p.id+' · '+(p.provider_api||'chat')+' · '+(en?'Writing':'写作')+' '+p.generation_thinking_mode+' / Reviewer '+p.reviewer_thinking_mode+' / Evaluation '+p.evaluation_thinking_mode+' · '+(p.reasoning_effort?((p.reviewer_thinking_mode==='disabled'?'writing effort=':'effort=')+p.reasoning_effort+' · '):'')+p.output_limit_parameter+' '+(p.generation_output_tokens??'—')+' / '+(p.reviewer_output_tokens??'—');document.getElementById('status').textContent=d.state+' · '+(d.stop_reason||'');let s=document.getElementById('stats');s.replaceChildren();for(const [k,v]of Object.entries({Planned:3,Run:d.run,Generated:d.generated,Evaluated:d.evaluation_completed,Delivered:d.passed})){let e=document.createElement('div');e.className='stat';e.textContent=k+' '+v+' / 3';s.append(e)}document.getElementById('usage').textContent=JSON.stringify({budget_plans:d.budget_plans??null,new_usage:d.new_usage,historical_usage:d.historical_usage,writing_profile:d.writing_profile,writer_prompt_contract:d.writer_prompt_contract??'unavailable',source_commit:d.source_commit,contract:d.contract},null,2)}document.getElementById('locale').onclick=()=>{en=!en;draw()};draw();</script></html>"""
    page = (
        page.replace("<th>Delivered</th>", "<th>Repairs</th><th>Delivered</th>")
        .replace("__ROWS__", table)
        .replace("__DETAILS__", details)
        .replace(
            "__DATA__", json.dumps(public, ensure_ascii=False).replace("<", "\\u003c")
        )
    )
    if semantic_rows:
        effect_table = "".join(
            "<tr>"
            + "".join(
                "<td>" + html.escape(str(value)) + "</td>"
                for value in (
                    row["case_id"],
                    scope,
                    effect["claim_id"],
                    effect["original_status"],
                    effect["original_reason_code"],
                    "unverified",
                    effect["code"],
                )
            )
            + "</tr>"
            for row in semantic_rows
            for scope, key in (
                ("overall", "semantic_conflict_effects"),
                ("partial", "partial_semantic_conflict_effects"),
            )
            for effect in (row.get(key) or [])
        )
        page = page.replace(
            "<section><h2>诊断 / Diagnostics</h2>",
            '<section class="scroll"><h2>模型原判与宿主未核验 / Original model decision and host unverified effect</h2><p>语义冲突不贡献支持数量；部分诊断非最终分数。Semantic conflicts add no supported assertion; partial diagnostics are not final scores.</p><table><thead><tr><th>Case</th><th>Scope</th><th>Claim</th><th>Original model status</th><th>Original reason</th><th>Host effect</th><th>Code</th></tr></thead><tbody>'
            + effect_table
            + "</tbody></table></section><section><h2>诊断 / Diagnostics</h2>",
            1,
        )
    if display_plans:
        table_rows = "".join(
            "<tr>"
            + "".join(
                "<td>" + html.escape(str(v)) + "</td>"
                for v in (
                    case,
                    p["status"],
                    p["required_tokens"],
                    p["planned_tokens"],
                    p["maximum_tokens"],
                    p["reason"] or "—",
                )
            )
            + "</tr>"
            for case, p in display_plans.items()
        )
        panel = (
            '<section class="scroll"><h2>预算准入 / Budget admission</h2><p>保守预留不是实际费用；未知后续输入使用上下文上界。Conservative reservations are not measured charges; unknown future inputs use the enforced context bound.</p><table><thead><tr><th>Case</th><th>Status</th><th>Reservation</th><th>Allowance</th><th>Ceiling</th><th>Reason</th></tr></thead><tbody>'
            + table_rows
            + "</tbody></table></section>"
        )
        if joint_plans:
            domain_rows = "".join(
                "<tr>"
                + "".join(
                    "<td>" + html.escape(str(value)) + "</td>"
                    for value in (
                        case,
                        domain["max_requests"],
                        domain["request_cap"],
                        domain["reserved_tokens"],
                    )
                )
                + "</tr>"
                for case, domain in joint_plans.items()
            )
            panel += (
                '<section class="scroll"><h2>每轮锁定核验资源域 / Locked assessment domain per round</h2><p>后续修复轮保留同一资源域；完整候选仍须发送前检查。预留不是实际费用。</p><table><tr><th>Case</th><th>Maximum requests</th><th>Per-request cap</th><th>Round total reservation</th></tr>'
                + domain_rows
                + "</table></section>"
            )
        if manifest.get("writing_profile") in (
            ADAPTIVE_PROFILE,
            BUDGETED_PROFILE,
            FLAT_PROFILE,
            CAPACITY_PROFILE,
            ROUND_PROFILE,
        ):
            limits = ", ".join(
                f"{case}={p['repair_limit']}/2" for case, p in display_plans.items()
            )
            panel += (
                "<section><p>计划修复路径（仅准入后锁定） / Planned repairs (locked only after admission): "
                + html.escape(limits)
                + "。首次完整 Evaluation 不可省略；0次修复不代表科学通过。</p></section>"
            )
        if manifest.get("writing_profile") in (
            BUDGETED_PROFILE,
            FLAT_PROFILE,
            CAPACITY_PROFILE,
            ROUND_PROFILE,
        ):
            scopes = ", ".join(
                f"{case}: body≤{p['envelope']['body_max_units']}, assertions≤{p['envelope']['model_assertions']}, batches≤{p['envelope']['assessment_batches']}"
                for case, p in display_plans.items()
            )
            panel += (
                "<section><p>整体容量（仅准入后锁定） / Content capacity (locked only after admission): "
                + html.escape(scopes)
                + (
                    "。这是初始核验容量；结算释放可用于完整扩批，容量不足时有界精简并完整复评。</p></section>"
                    if manifest.get("writing_profile")
                    in (CAPACITY_PROFILE, ROUND_PROFILE)
                    else "。越界停止，不截断；完整核验仍必需。</p></section>"
                )
            )
            panel = panel.replace(
                "未知后续输入使用上下文上界。",
                "声明的整体输出与派生请求规模会被强制检查。",
            )
            panel = panel.replace(
                "unknown future inputs use the enforced context bound.",
                "declared output and derived-request scope is enforced.",
            )
        if manifest.get("writing_profile") in (
            FLAT_PROFILE,
            CAPACITY_PROFILE,
            ROUND_PROFILE,
        ):
            panel += "<p>表中是预算计划的计算结果；启动还需完整离线门禁与冻结输入校验通过。Plan arithmetic alone does not authorize execution or establish scientific acceptance.</p>"
        page = page.replace(
            '<section class="scroll">', panel + '<section class="scroll">', 1
        )
    if cost_rows:
        cost_panel = (
            "<section class='scroll'><h2>费用证据不可用 / Cost evidence unavailable</h2>"
            "<p>仅保留未经认证的原记录；不作为已核实费用或后续请求授权。"
            "Reported records are unauthenticated, not verified charges or authorization.</p>"
            "<table><tr><th>Case</th><th>Reason</th><th>Reported requests</th>"
            "<th>Reported known tokens</th><th>Reported positive unknown hold</th></tr>"
        )
        for row in cost_rows:
            diagnostic = row["cost_diagnostic"]
            cost_panel += (
                "<tr>"
                + "".join(
                    "<td>" + html.escape(display(value)) + "</td>"
                    for value in (
                        row["case_id"],
                        diagnostic["reason_code"],
                        diagnostic["reported_requests"],
                        diagnostic["reported_known_tokens"],
                        diagnostic["reported_unknown_hold"],
                    )
                )
                + "</tr>"
            )
        cost_panel += "</table></section>"
        page = page.replace(
            '<section class="scroll">', cost_panel + '<section class="scroll">', 1
        )
    if finding_rows:
        diagnostic_panel = (
            "<section class='scroll'><h2>部分诊断非最终分数 / Partial diagnostics are not final scores</h2><table><tr><th>Case</th>"
            + "".join(f"<th>{key}</th>" for key in finding_fields)
            + "</tr>"
        )
        for row in finding_rows:
            diagnostic_panel += (
                "<tr><th>"
                + html.escape(row["case_id"])
                + "</th>"
                + "".join(
                    "<td>" + html.escape(display(row.get(key))) + "</td>"
                    for key in finding_fields
                )
                + "</tr>"
            )
        diagnostic_panel += "</table><p>整体无效时不能交付 / An invalid overall evaluation cannot authorize delivery.</p></section>"
        page = page.replace(
            '<section class="scroll">', diagnostic_panel + '<section class="scroll">', 1
        )
    if summary_rows:
        progress_panel = "<section class='scroll'><h2>摘要契约 / Summary contract</h2><p>收到五章草稿不等于完整报告，完成摘要不等于科学通过。Received chapter drafts, compiled reports and scientific acceptance are distinct.</p><table><tr><th>Case</th><th>Body chapters</th><th>Summary completed</th><th>Units per section</th></tr>"
        for row in summary_rows:
            progress = row["summary_progress"]
            progress_panel += (
                "<tr>"
                + "".join(
                    "<td>" + html.escape(str(value)) + "</td>"
                    for value in (
                        row["case_id"],
                        str(progress["body_chapters_completed"]) + "/5",
                        display(progress["summary_completed"]),
                        "1–" + str(progress["constraints"]["max_units"]),
                    )
                )
                + "</tr>"
            )
        progress_panel += "</table></section>"
        page = page.replace(
            '<section class="scroll">', progress_panel + '<section class="scroll">', 1
        )
    (out / "benchmark.html").write_text(page)
    return public


async def execute(root, config):
    """Reject concurrent submissions before touching the durable suite state."""
    with suite_lock(root / "execution.lock"):
        return await _execute(root, config)


def offline_gates_verified(root, manifest):
    """Require source-bound engineering evidence before enabling the new profile."""
    if manifest.get("writing_profile") not in (
        FLAT_PROFILE,
        CAPACITY_PROFILE,
        ROUND_PROFILE,
    ):
        return True
    try:
        gates = json.loads((root / "offline-gates.json").read_text())
        return (
            gates.get("contract") == "flat-writer-offline-gates-v1"
            and gates.get("manifest_identity") == manifest["identity"]
            and gates.get("source_hashes") == manifest["source_hashes"]
            and type(gates.get("pytest_exit_code")) is int
            and gates["pytest_exit_code"] == 0
            and gates.get("future_scope_verified") is True
            and gates.get("cases")
            == {
                key: {"budget_identity": plan["identity"], "scope_verified": True}
                for key, plan in manifest["budget_plans"].items()
            }
        )
    except (OSError, ValueError, TypeError, KeyError):
        return False


async def _execute(root, config):
    """Never replay a started observation or expand the three-case manifest."""
    m = validate(root, config)
    state_path = root / "state.json"
    if state_path.exists():
        state = json.loads(state_path.read_text())
        if state.get("identity") != m["identity"]:
            raise ValueError("frozen_campaign_state_conflict")
        if state["status"] != "prepared":
            if state["status"] == "running":
                state.update(
                    status="interrupted", stop_reason="process_interrupted_no_replay"
                )
                for entry in state["cases"]:
                    if entry["status"] == "started_no_replay":
                        entry.update(
                            status="interrupted_no_replay",
                            failure_family="interrupted",
                            failure_code="process_interrupted_no_replay",
                            unknown_hold=(
                                m.get("budget_plans", {})
                                .get(entry["case_id"], {})
                                .get("planned_tokens", 500000)
                            ),
                            request_hold=64,
                            accounting_scope="unsettled_case_cap",
                            usage_complete=False,
                        )
                atomic_json(state_path, state)
            export(root, state, m)
            return state
    else:
        state = {"identity": m["identity"], "status": "prepared", "cases": []}
    if not offline_gates_verified(root, m):
        state.update(
            status="preflight_rejected", stop_reason="future_request_scope_unverified"
        )
        atomic_json(state_path, state)
        export(root, state, m)
        return state
    if m.get("budget_plans") and any(
        p["status"] != "admitted" for p in m["budget_plans"].values()
    ):
        state.update(
            status="preflight_rejected", stop_reason="budget_preflight_rejected"
        )
        atomic_json(state_path, state)
        export(root, state, m)
        return state
    state["status"] = "running"
    atomic_json(state_path, state)
    for row in m["cases"]:
        run_id = root.name + "-" + row["case_id"]
        entry = {
            "case_id": row["case_id"],
            "run_id": run_id,
            "status": "started_no_replay",
            "generated": False,
            "reviewer_completed": False,
            "evaluation": "not_run",
            "publishable": False,
            "supported": None,
            "unsupported": None,
            "unverified": None,
        }
        state["cases"].append(entry)
        atomic_json(state_path, state)
        recorder = RunRecorder(run_id, row["bundle"]["query"], ArchiveRepository())
        recorder.set_config_summary(build_config_summary(config, "strict_auto_v2"))
        recorder.set_citation_style("ieee")
        tracer = (
            LangFuseTracer(
                config.observability.langfuse_host,
                os.getenv("LANGFUSE_PUBLIC_KEY", ""),
                os.getenv("LANGFUSE_SECRET_KEY", ""),
                trace_seed=run_id,
            )
            if config.observability.enabled
            else None
        )
        hook = CompositeTraceHook(
            recorder,
            (
                RedactingTraceHook(tracer, payload_mode="metadata_only")
                if tracer
                else None
            ),
        )
        agent = LitAgent(
            config, trace_hook=hook, run_policy=RunPolicy.isolated_benchmark()
        )
        start = time.monotonic()
        failure = None
        report = None
        try:
            report = await run_frozen(
                agent,
                row["bundle"],
                run_id,
                **(
                    {
                        "expected_budget_plan": m["budget_plans"][row["case_id"]],
                        "frozen_input_guard": frozen_runtime_guard(root, m, config),
                    }
                    if m.get("writing_profile")
                    in (BUDGETED_PROFILE, FLAT_PROFILE, CAPACITY_PROFILE, ROUND_PROFILE)
                    else {}
                ),
                **profile_parameters(
                    m.get("writing_profile", DISABLED_PROFILE),
                    writing_plan=m.get("writing_plan"),
                ),
                **inventory_recovery_options(m),
                **mixed_inventory_recovery_options(m),
                **classification_review_options(m),
                **repair_feedback_options(m),
                **reference_contract_options(m),
                **transport_revision_options(m),
                **support_validation_options(m),
                **repair_mapping_options(m),
                **repair_guidance_options(m),
                **repair_guidance_wire_options(m),
                **semantic_conflict_options(m),
                **evaluation_response_options(m),
                **repair_prompt_options(m),
                **round_resource_options(m),
                **writer_source_view_options(m),
                **evaluation_dimensions_options(m),
                **closeout_options(m),
            )
        except BaseException as exc:
            failure = classify_failure(exc)
        finally:
            budget = agent._cost_budget.snapshot() if agent._cost_budget else {}
            await agent.cleanup()
        if tracer:
            recorder.set_trace_reference(tracer.trace_id, tracer.trace_url)
            tracer.flush()
        artifact = recorder.finalize(
            report=report, error=(failure["code"] if failure else None)
        )
        events = artifact["events"]
        ends = [
            e["data"]
            for e in events
            if e["event"] == "llm.complete" and e["data"].get("usage_available")
        ]
        known = sum(e["total_tokens"] for e in ends)
        reasoning_available = [
            e for e in ends if e.get("reasoning_usage_available") is True
        ]
        reasoning_complete = bool(ends) and len(reasoning_available) == budget.get(
            "requests_sent", 0
        )
        entry.update(
            requests=budget.get("requests_sent", 0),
            known_tokens=known,
            unknown_hold=unknown_token_hold(budget, known),
            usage_complete=not budget.get("usage_uncertain", False),
            budget_snapshot=budget,
            elapsed_ms=round((time.monotonic() - start) * 1000),
            trace_url=tracer.trace_url if tracer else None,
            reasoning_tokens=(
                sum(e["reasoning_tokens"] for e in reasoning_available)
                if reasoning_complete
                else None
            ),
            reasoning_usage_complete=reasoning_complete,
            reasoning_usage_available_requests=len(reasoning_available),
        )
        if report:
            meta = report.get("metadata", {})
            faith = report.get("evaluation", {}).get("faithfulness", {})
            details = faith.get("details", {})
            entry.update(
                generated=bool(meta.get("writer_document")),
                evaluation=(
                    "completed"
                    if not faith.get("skipped", True)
                    and details.get("stages", {}).get("global_check") == "completed"
                    else "not_complete"
                ),
                publishable=report.get("delivery", {}).get("publishable", False),
                report_sha256=(
                    hashlib.sha256(report["survey"].encode()).hexdigest()
                    if meta.get("writer_document")
                    else None
                ),
                supported=details.get("supported_count"),
                unsupported=details.get("unsupported_count"),
                unverified=details.get("unverified_count"),
                repairs=len(meta.get("targeted_repair_state", {}).get("attempts", [])),
                **(
                    {
                        "capacity": details.get("stages", {}).get("capacity"),
                        "capacity_revisions": len(budget.get("capacity_revisions", [])),
                        "capacity_refinements": sum(
                            a.get("cause") == "capacity"
                            for a in meta.get("targeted_repair_state", {}).get(
                                "attempts", []
                            )
                        ),
                    }
                    if m.get("writing_profile") in (CAPACITY_PROFILE, ROUND_PROFILE)
                    else {}
                ),
                core_coverage=details.get("core_coverage", []),
                global_issues=details.get("global_issues", []),
                evaluation_history=[
                    {
                        "stage": label,
                        "supported": value.get("faithfulness", {})
                        .get("details", {})
                        .get("supported_count"),
                        "unsupported": value.get("faithfulness", {})
                        .get("details", {})
                        .get("unsupported_count"),
                        "unverified": value.get("faithfulness", {})
                        .get("details", {})
                        .get("unverified_count"),
                        **(
                            {
                                "complete": value.get("faithfulness", {})
                                .get("details", {})
                                .get("coverage", {})
                                .get("complete", False)
                            }
                            if m.get("inventory_recovery") is True
                            else {}
                        ),
                    }
                    for label, value in [
                        (
                            "initial",
                            meta.get("evaluation_history", {}).get(
                                "initial_evaluation", {}
                            ),
                        ),
                        *[
                            (
                                evaluation_attempt_label(a, i),
                                a.get("candidate_evaluation") or {},
                            )
                            for i, a in enumerate(
                                meta.get("evidence_rewrite_attempts", [])
                            )
                        ],
                    ]
                ],
            )
            if located_findings_enabled(details):
                partial = details.get("partial_assessment") or {}
                entry.update(
                    global_findings_contract=details["global_findings_contract"],
                    assessment_stage=details.get("assessment_stage"),
                    inventory_status=details.get("inventory_check", {}).get("status"),
                    partial_assessment_stage=partial.get("stage"),
                    partial_supported=partial.get("supported_count"),
                    partial_unsupported=partial.get("unsupported_count"),
                    partial_unverified=partial.get("unverified_count"),
                )
            failure = meta.get("writer_failure") or failure
            journal = Path(
                "artifacts/targeted-repairs", run_id + ".json.writer-v2.json"
            )
            if m.get("closeout_profile") is not None:
                from litagent.closeout import result_summary

                entry.update(result_summary(report))
            if not failure and not entry["publishable"] and journal.exists():
                j = json.loads(journal.read_text())
                failure = j.get("failure") or (
                    None
                    if entry["generated"]
                    else next(
                        (
                            v.get("failure")
                            for v in reversed(list(j["calls"].values()))
                            if v.get("failure")
                        ),
                        None,
                    )
                )
            if not failure and entry["evaluation"] != "completed":
                failure = {
                    "family": (
                        "quality"
                        if located_findings_enabled(details)
                        and details.get("assessment_stage") == "host_validated"
                        else (
                            "resource"
                            if m.get("writing_profile")
                            in (CAPACITY_PROFILE, ROUND_PROFILE)
                            and details.get("stages", {})
                            .get("capacity", {})
                            .get("status")
                            in {"needs_refinement", "exhausted"}
                            else "protocol"
                        )
                    ),
                    "code": (evaluation_diagnostic(report) or {}).get("reason_code")
                    or "evaluation_not_completed",
                }
        journal_path = Path(
            "artifacts/targeted-repairs", run_id + ".json.writer-v2.json"
        )
        if journal_path.exists():
            journal = json.loads(journal_path.read_text())
            if (
                journal.get("summary_constraints", {}).get("contract")
                == "effective-summary-limits-v1"
            ):
                calls = journal.get("calls", {})
                summaries = [c for c in calls.values() if c.get("stage") == "summary"]
                failures = [
                    {k: d[k] for k in ("section", "actual", "maximum")}
                    for call in summaries
                    for d in (call.get("failure") or {}).get("diagnostics", [])
                    if d.get("section") in ("abstract", "conclusion")
                    and type(d.get("actual")) is int
                    and type(d.get("maximum")) is int
                ]
                entry["summary_progress"] = {
                    "constraints": journal["summary_constraints"],
                    "constraints_sha256": journal["summary_constraints_sha256"],
                    "body_chapters_completed": sum(
                        c.get("status") == "response_received" and not c.get("failure")
                        for name, c in calls.items()
                        if name.startswith("body:")
                    ),
                    "summary_completed": any(
                        c.get("status") == "response_received" and not c.get("failure")
                        for c in summaries
                    ),
                    "failures": failures,
                }
            entry["generated"] = (
                entry["generated"] or journal.get("status") == "generated"
            )
            reviewed = journal.get("calls", {}).get("reviewer", {})
            entry["reviewer_completed"] = reviewed.get(
                "status"
            ) == "response_received" and not reviewed.get("failure")
            entry["compiled_report_sha256"] = journal.get("draft_sha256")
            repair_usage = repair_accounting(run_id)
            if repair_usage is not None:
                entry.update(repair_usage)
        if entry["unknown_hold"] or not entry["usage_complete"]:
            failure = {"family": "interrupted", "code": "usage_unknown"}
        failure = failure or (
            None
            if entry["publishable"]
            else {"family": "quality", "code": "strict_quality_failed"}
        )
        entry.update(
            status="passed" if entry["publishable"] else "blocked",
            failure_family=failure["family"] if failure else None,
            failure_code=failure["code"] if failure else None,
        )
        atomic_json(state_path, state)
        export(root, state, m)
        allowed_terminal = {"quality", "evidence"} | (
            {"resource"}
            if m.get("writing_profile") in (CAPACITY_PROFILE, ROUND_PROFILE)
            else set()
        )
        if failure and failure["family"] not in allowed_terminal:
            state.update(status="stopped", stop_reason=failure["code"])
            break
    else:
        state["status"] = "completed"
    atomic_json(state_path, state)
    export(root, state, m)
    return state


def main():
    """Require explicit preparation or bounded live execution of the frozen suite."""
    p = argparse.ArgumentParser()
    p.add_argument("--root", type=Path, required=True)
    p.add_argument("--accounting-parent", type=Path, action="append", default=[])
    p.add_argument("--inventory-recovery", action="store_true")
    p.add_argument("--mixed-inventory-recovery", action="store_true")
    p.add_argument("--classification-review", action="store_true")
    p.add_argument("--repair-feedback", action="store_true")
    p.add_argument(
        "--reference-contract", choices=("ordered-reference-set-v1",), default=None
    )
    p.add_argument(
        "--transport-revision", choices=("host-lossless-transport-v1",), default=None
    )
    p.add_argument(
        "--support-validation-revision",
        choices=("host-support-receipts-v1", "host-support-receipts-v2"),
        default=None,
    )
    p.add_argument(
        "--repair-mapping-contract",
        choices=("capacity-source-map-v3", "capacity-source-map-v4"),
        default=None,
    )
    p.add_argument(
        "--repair-guidance-contract",
        choices=("host-structured-repair-v1",),
        default=None,
    )
    p.add_argument(
        "--repair-guidance-wire-revision",
        choices=("host-unit-ref-guidance-v1", "host-repair-task-view-v1"),
        default=None,
    )
    p.add_argument(
        "--evaluation-response-contract",
        choices=("host-bound-evaluation-v1", "host-bound-evaluation-v2"),
        default=None,
    )
    p.add_argument(
        "--repair-prompt-revision",
        choices=("concise-grounded-repair-v1",),
        default=None,
    )
    p.add_argument(
        "--round-resource-contract",
        choices=("joint-round-escrow-v1",),
        default=None,
    )
    p.add_argument(
        "--writer-source-view",
        choices=("source-only-writer-v1",),
        default=None,
    )
    p.add_argument(
        "--evaluation-dimensions-contract",
        choices=("separated-evaluation-dimensions-v1",),
        default=None,
    )
    p.add_argument(
        "--closeout-profile", choices=("engineering-closeout-v1",), default=None
    )
    accounting = p.add_mutually_exclusive_group()
    accounting.add_argument("--carry-planned-unknown", action="store_true")
    accounting.add_argument(
        "--accounting-policy",
        choices=(PLANNED_MIXED_UNKNOWN_ACCOUNTING,),
        default=None,
    )
    p.add_argument(
        "--writing-plan",
        choices=(FOCUSED_WRITING_PLAN, RANGE_WRITING_PLAN),
        default=None,
    )
    p.add_argument(
        "--writing-profile",
        choices=(
            DISABLED_PROFILE,
            THINKING_PROFILE,
            LOW_THINKING_PROFILE,
            RESPONSES_PROFILE,
            SHAPE_PROFILE,
            REEMISSION_PROFILE,
            ROLE_PROFILE,
            EOF_PROFILE,
            PLANNED_PROFILE,
            ADAPTIVE_PROFILE,
            BUDGETED_PROFILE,
            FLAT_PROFILE,
            CAPACITY_PROFILE,
            LEGACY_REEMISSION_PROFILE,
            ROUND_PROFILE,
        ),
        default=None,
    )
    g = p.add_mutually_exclusive_group(required=True)
    g.add_argument("--prepare", action="store_true")
    g.add_argument("--live", action="store_true")
    args = p.parse_args()
    if args.live and args.closeout_profile is not None:
        p.error("--closeout-profile is frozen during --prepare only")
    if args.closeout_profile is not None:
        from litagent.closeout import profile_flags

        for name, value in profile_flags(args.closeout_profile).items():
            setattr(args, name, value)
        args.writing_profile = ROUND_PROFILE
        args.writing_plan = RANGE_WRITING_PLAN
        args.round_resource_contract = "joint-round-escrow-v1"
    if args.live and args.accounting_parent:
        p.error("--accounting-parent is frozen during --prepare only")
    if args.live and args.writing_profile is not None:
        p.error("--writing-profile is frozen during --prepare only")
    if args.live and args.writing_plan is not None:
        p.error("--writing-plan is frozen during --prepare only")
    if args.live and args.inventory_recovery:
        p.error("--inventory-recovery is frozen during --prepare only")
    if args.live and args.mixed_inventory_recovery:
        p.error("--mixed-inventory-recovery is frozen during --prepare only")
    if args.live and args.classification_review:
        p.error("--classification-review is frozen during --prepare only")
    if args.live and args.repair_feedback:
        p.error("--repair-feedback is frozen during --prepare only")
    if args.live and args.reference_contract is not None:
        p.error("--reference-contract is frozen during --prepare only")
    if args.live and args.transport_revision is not None:
        p.error("--transport-revision is frozen during --prepare only")
    if args.live and args.support_validation_revision is not None:
        p.error("--support-validation-revision is frozen during --prepare only")
    if args.live and args.repair_mapping_contract is not None:
        p.error("--repair-mapping-contract is frozen during --prepare only")
    if args.live and args.repair_guidance_contract is not None:
        p.error("--repair-guidance-contract is frozen during --prepare only")
    if args.live and args.repair_guidance_wire_revision is not None:
        p.error("--repair-guidance-wire-revision is frozen during --prepare only")
    if args.live and args.evaluation_response_contract is not None:
        p.error("--evaluation-response-contract is frozen during --prepare only")
    if args.live and args.repair_prompt_revision is not None:
        p.error("--repair-prompt-revision is frozen during --prepare only")
    if args.live and args.round_resource_contract is not None:
        p.error("--round-resource-contract is frozen during --prepare only")
    if args.live and args.writer_source_view is not None:
        p.error("--writer-source-view is frozen during --prepare only")
    if args.live and args.evaluation_dimensions_contract is not None:
        p.error("--evaluation-dimensions-contract is frozen during --prepare only")
    if args.live and args.carry_planned_unknown:
        p.error("--carry-planned-unknown is frozen during --prepare only")
    if args.live and args.accounting_policy is not None:
        p.error("--accounting-policy is frozen during --prepare only")
    if args.accounting_policy is not None and args.writing_profile != ROUND_PROFILE:
        p.error("--accounting-policy requires the frozen round profile")
    cfg = load_config()
    profile = (
        (args.writing_profile or DISABLED_PROFILE)
        if args.prepare
        else json.loads((args.root / "manifest.json").read_text()).get(
            "writing_profile", DISABLED_PROFILE
        )
    )
    cfg = profile_config(cfg, profile)
    result = (
        prepare(
            args.root,
            cfg,
            accounting_parents=args.accounting_parent,
            writing_profile=profile,
            writing_plan=args.writing_plan,
            inventory_recovery=args.inventory_recovery,
            **(
                {"mixed_inventory_recovery": True}
                if args.mixed_inventory_recovery
                else {}
            ),
            **({"classification_review": True} if args.classification_review else {}),
            **({"repair_feedback": True} if args.repair_feedback else {}),
            **(
                {"reference_contract": args.reference_contract}
                if args.reference_contract is not None
                else {}
            ),
            **(
                {"transport_revision": args.transport_revision}
                if args.transport_revision is not None
                else {}
            ),
            **(
                {"support_validation_revision": args.support_validation_revision}
                if args.support_validation_revision is not None
                else {}
            ),
            **(
                {"repair_mapping_contract": args.repair_mapping_contract}
                if args.repair_mapping_contract is not None
                else {}
            ),
            **(
                {"repair_guidance_contract": args.repair_guidance_contract}
                if args.repair_guidance_contract is not None
                else {}
            ),
            **(
                {"repair_guidance_wire_revision": args.repair_guidance_wire_revision}
                if args.repair_guidance_wire_revision is not None
                else {}
            ),
            **(
                {"semantic_conflict_policy": args.semantic_conflict_policy}
                if args.closeout_profile is not None
                else {}
            ),
            **(
                {"evaluation_response_contract": args.evaluation_response_contract}
                if args.evaluation_response_contract is not None
                else {}
            ),
            **(
                {"repair_prompt_revision": args.repair_prompt_revision}
                if args.repair_prompt_revision is not None
                else {}
            ),
            **(
                {"round_resource_contract": args.round_resource_contract}
                if args.round_resource_contract is not None
                else {}
            ),
            **(
                {"writer_source_view": args.writer_source_view}
                if args.writer_source_view is not None
                else {}
            ),
            **(
                {"evaluation_dimensions_contract": args.evaluation_dimensions_contract}
                if args.evaluation_dimensions_contract is not None
                else {}
            ),
            **(
                {"closeout_profile": args.closeout_profile}
                if args.closeout_profile is not None
                else {}
            ),
            **({"carry_planned_unknown": True} if args.carry_planned_unknown else {}),
            **(
                {"accounting_policy": args.accounting_policy}
                if args.accounting_policy is not None
                else {}
            ),
        )
        if args.prepare
        else asyncio.run(execute(args.root, cfg))
    )
    print(
        json.dumps(
            {
                "suite": args.root.name,
                **{
                    k: v
                    for k, v in result.items()
                    if k in {"identity", "status", "stop_reason", "cases"}
                },
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
