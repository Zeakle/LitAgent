"""Freeze server-owned run allowances independently of historical budgets."""

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from litagent import review_policy

CONTRACT = "configured-run-limits-v1"
UPLIFT_CONTRACT = "configured-run-limits-v2"
EXTENDED_CONTRACT = "configured-run-limits-v3"
FORMAT_CONTRACT = "configured-run-limits-v4"
REQUEST_CONTRACT = "configured-run-limits-v5"
PROFILE = "operational-plus20-v1"
EXTENDED_PROFILE = "extended-repair-v1"
FORMAT_PROFILE = "agent-format-repair-v1"
FORMAT_JSON_POLICY = "agent-format-recovery-v1"
FORMAT_MODELS = ("doubao-seed-2-1-lite-260915", "doubao-seed-2-1-pro-260915")
LOSSLESS_JSON_POLICY = "lossless-json-recovery-v1"
REPAIR_ALLOWANCES = {"capacity": 3, "inventory": 3, "prose": 5}
LEGACY = {
    "max_requests": 64,
    "max_repairs": 2,
    "context_tokens": 196608,
    "max_output_tokens": 32768,
}
UPLIFT = {
    "max_requests": 77,
    "max_repairs": 3,
    "context_tokens": 235930,
    "max_output_tokens": 39322,
}
EXTENDED = {
    **UPLIFT,
    "max_requests": 160,
    "max_repairs": sum(REPAIR_ALLOWANCES.values()),
}
PROFILES = {
    review_policy.PROFILE: (
        review_policy.LIMITS_CONTRACT,
        {**EXTENDED, "max_requests": 226, "max_repairs": 12},
        47_420_000,
    ),
    FORMAT_PROFILE: (FORMAT_CONTRACT, {**EXTENDED, "max_requests": 226}, 47_420_000),
    None: (CONTRACT, LEGACY, None),
    PROFILE: (UPLIFT_CONTRACT, UPLIFT, 12_000_000),
    EXTENDED_PROFILE: (EXTENDED_CONTRACT, EXTENDED, 32_000_000),
}


class PlannedRunLimits(BaseModel):
    """Configure finite resources with an explicit opt-in revision for larger bounds."""

    model_config = ConfigDict(extra="forbid")
    max_tokens: int = Field(default=10_000_000, strict=True, ge=500_000, le=2**63 - 1)
    profile: (
        Literal[
            "operational-plus20-v1",
            "extended-repair-v1",
            "agent-format-repair-v1",
            "separate-review-evaluation-v1",
        ]
        | None
    ) = None
    max_requests: int = Field(default=64, strict=True, ge=1, le=226)
    max_repairs: int = Field(default=2, strict=True, ge=0, le=12)
    context_tokens: int = Field(default=196608, strict=True, ge=32768, le=235930)
    max_output_tokens: int = Field(default=32768, strict=True, ge=1, le=39322)
    syntax_policy: (
        Literal["lossless-json-recovery-v1", "agent-format-recovery-v1"] | None
    ) = None

    @model_validator(mode="after")
    def validate_profile(self):
        """Prevent new values from being smuggled into an old frozen contract."""
        _, expected, maximum = PROFILES[self.profile]
        if any(
            getattr(self, key) != value
            for key, value in expected.items()
            if key != "max_requests" or self.profile is not None
        ):
            raise ValueError("run_limits_profile_mismatch")
        if maximum is not None and self.max_tokens > maximum:
            raise ValueError("run_limits_profile_budget_exceeded")
        if (
            (self.syntax_policy == FORMAT_JSON_POLICY)
            != (self.profile in (FORMAT_PROFILE, review_policy.PROFILE))
            or self.syntax_policy == LOSSLESS_JSON_POLICY
            and self.profile != EXTENDED_PROFILE
        ):
            raise ValueError("run_limits_syntax_policy_requires_matching_profile")
        return self

    @property
    def contract(self) -> str:
        """Identify explicit request ceilings without changing v1 fingerprints."""
        if self.profile is None and self.max_requests != LEGACY["max_requests"]:
            return REQUEST_CONTRACT
        return PROFILES[self.profile][0]

    @model_serializer(mode="wrap")
    def preserve_legacy_identity(self, handler):
        """Leave the original single-field configuration byte identity unchanged."""
        value = handler(self)
        if self.contract == CONTRACT:
            return {"max_tokens": value["max_tokens"]}
        if self.contract == REQUEST_CONTRACT:
            return {key: value[key] for key in ("max_tokens", "max_requests")}
        if self.syntax_policy is None:
            value.pop("syntax_policy", None)
        return value


def validate_run_limits(value):
    """Authenticate a frozen resource contract without upgrading old records."""
    if not isinstance(value, dict) or "contract" not in value:
        raise ValueError("run_limits_invalid")
    contract = value["contract"]
    if contract in (CONTRACT, REQUEST_CONTRACT):
        expected = {"contract", "max_tokens"}
        if contract == REQUEST_CONTRACT:
            expected.add("max_requests")
    else:
        expected = {
            "contract",
            "max_tokens",
            "profile",
            *UPLIFT,
        }
        if contract == review_policy.LIMITS_CONTRACT:
            expected.add("workflow_identity")
            if value.get("workflow_identity") != review_policy.identity():
                raise ValueError("review_workflow_identity_invalid")
    if (
        contract
        not in (
            CONTRACT,
            UPLIFT_CONTRACT,
            EXTENDED_CONTRACT,
            FORMAT_CONTRACT,
            review_policy.LIMITS_CONTRACT,
            REQUEST_CONTRACT,
        )
        or set(value) - {"syntax_policy"} != expected
        or (
            "syntax_policy" in value
            and (
                value["syntax_policy"]
                != {
                    EXTENDED_CONTRACT: LOSSLESS_JSON_POLICY,
                    FORMAT_CONTRACT: FORMAT_JSON_POLICY,
                    review_policy.LIMITS_CONTRACT: FORMAT_JSON_POLICY,
                }.get(contract)
            )
        )
    ):
        raise ValueError("run_limits_contract_invalid")
    limits = PlannedRunLimits.model_validate(
        {k: v for k, v in value.items() if k not in {"contract", "workflow_identity"}}
    )
    if limits.contract != contract:
        raise ValueError("run_limits_profile_mismatch")
    return {
        "contract": contract,
        **limits.model_dump(),
        **(
            {"workflow_identity": review_policy.identity()}
            if contract == review_policy.LIMITS_CONTRACT
            else {}
        ),
    }


def configured_run_limits(config):
    """Read only server settings; absent and v1 settings keep historical semantics."""
    safety = config["safety"] if isinstance(config, Mapping) else config.safety
    limits = (
        safety.get("planned_run_limits")
        if isinstance(safety, Mapping)
        else getattr(safety, "planned_run_limits", None)
    )
    if limits is None:
        return None
    limits = (
        limits
        if isinstance(limits, PlannedRunLimits)
        else PlannedRunLimits.model_validate(limits)
    )
    return {
        "contract": limits.contract,
        **limits.model_dump(),
        **(
            {"workflow_identity": review_policy.identity()}
            if limits.contract == review_policy.LIMITS_CONTRACT
            else {}
        ),
    }


def envelope_run_limits(envelope=None):
    """Resolve the same frozen request and repair ceilings at every consumer."""
    value = (envelope or {}).get("run_limits")
    if value is None:
        return dict(LEGACY)
    checked = validate_run_limits(value)
    return {**LEGACY, **{key: checked[key] for key in UPLIFT if key in checked}}


def repair_allowances(envelope):
    """Return the independent quotas of the explicitly selected native profile."""
    limits = (envelope or {}).get("run_limits") or {}
    if review_policy.enabled(envelope):
        validate_run_limits(limits)
        if not envelope.get("semantic_engine"):
            raise ValueError("extended_repair_requires_native_ragas")
        return dict(review_policy.ALLOWANCES)
    if limits.get("contract") not in (EXTENDED_CONTRACT, FORMAT_CONTRACT):
        return None
    validate_run_limits(limits)
    if not envelope.get("semantic_engine"):
        raise ValueError("extended_repair_requires_native_ragas")
    return dict(REPAIR_ALLOWANCES)


def extended_repair_stages(envelope, repairs):
    """Reserve every permitted edit, summary and native Evaluation round."""
    limits = repair_allowances(envelope)
    if limits is None:
        return None
    if repairs != sum(limits.values()):
        raise ValueError("extended_repair_full_allocation_required")
    batches = (
        envelope["body_max_units"]
        + 2 * envelope["summary_max_units_per_section"]
        + envelope["semantic_engine"]["batch_size"]
        - 1
    ) // envelope["semantic_engine"]["batch_size"]
    if review_policy.enabled(envelope):
        return review_policy.stage_counts(batches)
    rounds = 1 + limits["inventory"] + limits["prose"]
    edits = limits["capacity"] + limits["prose"]
    return {
        "summary": 1 + edits,
        "reviewer": 1,
        "repair": edits,
        "inventory": rounds * batches,
        "assessment": rounds * batches,
        "global": rounds,
    }


def format_recovery_stages(envelope):
    """Reserve two format calls and one exact regeneration per authored response."""
    limits = (envelope or {}).get("run_limits") or {}
    if limits.get("syntax_policy") != FORMAT_JSON_POLICY:
        return {}
    limits = repair_allowances(envelope)
    counts = extended_repair_stages(envelope, sum(limits.values()))
    if review_policy.enabled(envelope):
        return review_policy.format_stages(counts)
    return {
        "format_body": ("body", 3 * 5),
        "format_summary": ("summary", 3 * counts["summary"]),
        "format_repair": ("repair", 3 * counts["repair"]),
    }
