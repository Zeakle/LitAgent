"""Freeze server-owned run allowances independently of historical budgets."""

from collections.abc import Mapping
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field, model_serializer, model_validator

from litagent import review_policy, stage_repair
from litagent.capacity_recovery import CONTRACT as INITIAL_CAPACITY_RECOVERY
from litagent.capacity_recovery import identity as initial_capacity_identity

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
FORMAT_CONTEXT_TOKENS = 245760
FORMAT_GLOBAL_CONTEXT_TOKENS = 307200
FORMAT_OUTPUT_TOKENS = 45056
LOSSLESS_JSON_POLICY = "lossless-json-recovery-v1"
PREVIOUS_WORKFLOW_IDENTITY = (
    "c24dddf369888b41ab4f250bf2bfc2537cd423967b1bc9c668c00b0fb9ffc784"
)
SUMMARY_420_WORKFLOW_IDENTITY = (
    "97f008d6e4a3c7aa58b3788e155f5daa76140b71da80971fff0ebc44caf447e1"
)
REPAIR_ALLOWANCES = {"capacity": 3, "inventory": 3, "prose": 5}
# These workflow and citation identities are pinned pairs, never live selectors.
ARCHIVED_INLINE_CITATION_IDENTITY = (
    "bded3bb82f134756b211bdfaa0a38fd533c60dc898685c753030fbf342f15f49"
)
ARCHIVED_INLINE_WORKFLOWS = {
    "83672a7b02e85acd5142f8a6fa404f18cdf0d91cd217d6579b19dca7f62558c2",
    "aa9f3d9666a59259087cceb4c95392129f30913d903a8f465d43e77a9da50314",
}
ARCHIVED_INLINE_SUMMARY_WORKFLOW = (
    "3e3aea45309a5c546feb601cd43c4c7b8b4baff5c8e387a01058985a1fb45472",
    "a2e4b89fb9c6dcd7c967baea9fedfa852b233c8c0973601d3619f1dd87b6d972",
)
ARCHIVED_INLINE_DEPENDENCY_WORKFLOW = (
    ARCHIVED_INLINE_SUMMARY_WORKFLOW[0],
    "d896c4e5234cbe500142031eff57054ad1db0f04931dea43b0cce1210b4f58d5",
)
ARCHIVED_INLINE_RETRY_WORKFLOW = (
    "cc0979883e7df1793e1282d33a87b14890f9576cffbfbec956d2d9235a3ba156",
    "a47b7ade0011e7529220569258a74bde43fefcb50d73f991af0102a178084a1e",
)
ARCHIVED_INLINE_CANDIDATE_WORKFLOW = (
    "5374d33460689deb891fd06a92171579c51c6db571488ac06e7edc0295a94118",
    "052f6cd08de0024ac1d71cfe7d436f92b8f58b4d2476472f3798649fb507b5a4",
)

ARCHIVED_INLINE_PATCH_WORKFLOW = (
    "7b879006f108e344c2fd2acb0d0ee5fc794d9263023208723a617a966d16a558",
    "6d5906211c1d0bc9b676e575d85784d42699ab5b661637cb9aaf1f307b5d7c9c",
)

# ViT 11 is readable after feedback changes, but never resumable under them.
ARCHIVED_STAGE_REPAIR_IDENTITIES = (
    "ddcc2745445a08b1b4c295cdb5bffafae5d34c0963a82c3c974a1dd52be30b3f",
    "7d262f6c05498461f71700652a202175fce5b0c0bfb74c1a2c1fced29ba3c0cf",
    "90ebd7071774543313a4e5eba1614455fad65834bebba16eab2f6bba89fd6300",
)

# Exact ViT 12 identity before adding source-owner candidate diagnostics.
ARCHIVED_STAGE_REPAIR_VIT12 = (
    "b5e14e357785930e3d582d802389a17964945649acec36369981f4f12888adbb",
    "7d262f6c05498461f71700652a202175fce5b0c0bfb74c1a2c1fced29ba3c0cf",
    "90ebd7071774543313a4e5eba1614455fad65834bebba16eab2f6bba89fd6300",
)

# ViT 13 ended with an unknown request; this pair permits accounting reads only.
ARCHIVED_STAGE_REPAIR_VIT13 = (
    "0e8db61718551a9c43e008b34560e66c4d9e9f8f464e373f0ea454b4804e5876",
    "7d262f6c05498461f71700652a202175fce5b0c0bfb74c1a2c1fced29ba3c0cf",
    "90ebd7071774543313a4e5eba1614455fad65834bebba16eab2f6bba89fd6300",
)

# ViT 14 used SSE; the transport is part of this exact read-only combination.
ARCHIVED_STAGE_REPAIR_VIT14 = (
    "ac740946070437490a36e2a8dc6d27fc3af2852039f9c90df0369f55da43ecc0",
    "7d262f6c05498461f71700652a202175fce5b0c0bfb74c1a2c1fced29ba3c0cf",
    "90ebd7071774543313a4e5eba1614455fad65834bebba16eab2f6bba89fd6300",
    "ark-responses-sse-v1",
    "657c3300822034fb5293a81bb0eda80010793d898d7a683df98938e4311e5254",
)

ARCHIVED_STAGE_REPAIR_PRE_CAPACITY_RETRY = (
    "1e85f0aea61507a397f0f8a6b3bea9f8a74fb7bb9c2623343bf69beab635812a",
    "7d262f6c05498461f71700652a202175fce5b0c0bfb74c1a2c1fced29ba3c0cf",
    "90ebd7071774543313a4e5eba1614455fad65834bebba16eab2f6bba89fd6300",
)
ARCHIVED_STAGE_REPAIR_STREAM_PRE_CAPACITY_RETRY = (
    *ARCHIVED_STAGE_REPAIR_PRE_CAPACITY_RETRY,
    "ark-responses-sse-v1",
    "9ff42600c1ee5dc1d39dc2d97be845320442424f40215a3a3902b9d92eb780ed",
)

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
    context_tokens: int = Field(
        default=196608, strict=True, ge=32768, le=FORMAT_CONTEXT_TOKENS
    )
    global_context_tokens: int | None = Field(
        default=None, strict=True, gt=235930, le=FORMAT_GLOBAL_CONTEXT_TOKENS
    )
    max_output_tokens: int = Field(
        default=32768, strict=True, ge=1, le=FORMAT_OUTPUT_TOKENS
    )
    syntax_policy: (
        Literal["lossless-json-recovery-v1", "agent-format-recovery-v1"] | None
    ) = None

    citation_contract: Literal["flat-writer-v2"] | None = None
    candidate_recovery: Literal["llm-stage-repair-v1"] | None = None
    initial_capacity_recovery: Literal["initial-capacity-source-retry-v1"] | None = None
    responses_transport: Literal["ark-responses-sse-v1"] | None = None

    @model_validator(mode="after")
    def validate_profile(self):
        """Prevent new values from being smuggled into an old frozen contract."""
        if self.citation_contract and self.profile != review_policy.PROFILE:
            raise ValueError("inline_citations_require_separate_review")
        if self.candidate_recovery and self.citation_contract != "flat-writer-v2":
            raise ValueError("stage_repair_requires_inline_citations")
        if self.responses_transport and not self.candidate_recovery:
            raise ValueError("responses_stream_requires_stage_repair")
        if self.initial_capacity_recovery and self.profile != FORMAT_PROFILE:
            raise ValueError("initial_capacity_recovery_requires_format_profile")
        if self.global_context_tokens is not None and self.profile != FORMAT_PROFILE:
            raise ValueError("global_context_requires_format_profile")
        if (
            self.global_context_tokens is not None
            and self.global_context_tokens <= self.context_tokens
        ):
            raise ValueError("global_context_must_exceed_baseline")
        _, expected, maximum = PROFILES[self.profile]
        format_bounds = (
            {
                "context_tokens": FORMAT_CONTEXT_TOKENS,
                "max_output_tokens": FORMAT_OUTPUT_TOKENS,
            }
            if self.profile == FORMAT_PROFILE
            else {}
        )
        if any(
            getattr(self, key) not in (value, format_bounds.get(key, value))
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
        if self.candidate_recovery is None:
            value.pop("candidate_recovery", None)
        if self.initial_capacity_recovery is None:
            value.pop("initial_capacity_recovery", None)
        if self.global_context_tokens is None:
            value.pop("global_context_tokens", None)
        if self.citation_contract is None:
            value.pop("citation_contract", None)
        if self.responses_transport is None:
            value.pop("responses_transport", None)
        return value


def validate_run_limits(value, *, archived=False):
    """Authenticate a frozen resource contract without upgrading old records."""
    if not isinstance(value, dict) or "contract" not in value:
        raise ValueError("run_limits_invalid")
    archived_stream = archived and tuple(
        value.get(key)
        for key in (
            "candidate_recovery_identity",
            "citation_identity",
            "workflow_identity",
            "responses_transport",
            "responses_transport_identity",
        )
    ) in (ARCHIVED_STAGE_REPAIR_VIT14, ARCHIVED_STAGE_REPAIR_STREAM_PRE_CAPACITY_RETRY)
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
            if "citation_contract" in value:
                expected.update(("citation_contract", "citation_identity"))
                from litagent.inline_citations import CONTRACT as INLINE_CONTRACT
                from litagent.inline_citations import identity

                historical = archived and (
                    (
                        value.get("citation_identity")
                        == ARCHIVED_INLINE_CITATION_IDENTITY
                        and value.get("workflow_identity") in ARCHIVED_INLINE_WORKFLOWS
                    )
                    or (value.get("citation_identity"), value.get("workflow_identity"))
                    in (
                        ARCHIVED_INLINE_SUMMARY_WORKFLOW,
                        ARCHIVED_INLINE_DEPENDENCY_WORKFLOW,
                        ARCHIVED_INLINE_RETRY_WORKFLOW,
                        ARCHIVED_INLINE_CANDIDATE_WORKFLOW,
                        ARCHIVED_INLINE_PATCH_WORKFLOW,
                    )
                )
                if value["citation_contract"] != INLINE_CONTRACT or (
                    not historical and value.get("citation_identity") != identity()
                ):
                    raise ValueError("inline_citation_identity_invalid")
                workflow_ids = (
                    (value["workflow_identity"],)
                    if historical
                    else (review_policy.identity(inline=True),)
                )
            else:
                workflow_ids = (
                    review_policy.identity(),
                    PREVIOUS_WORKFLOW_IDENTITY,
                    SUMMARY_420_WORKFLOW_IDENTITY,
                )
            if value.get("workflow_identity") not in workflow_ids:
                raise ValueError("review_workflow_identity_invalid")
    if "candidate_recovery" in value:
        from litagent.stage_repair import CONTRACT as REPAIR_CONTRACT
        from litagent.stage_repair import identity as repair_identity

        expected.update(("candidate_recovery", "candidate_recovery_identity"))
        if (
            value["candidate_recovery"] != REPAIR_CONTRACT
            or (
                value.get("candidate_recovery_identity") != repair_identity()
                and not (
                    archived
                    and (
                        (
                            "responses_transport" not in value
                            and tuple(
                                value.get(key)
                                for key in (
                                    "candidate_recovery_identity",
                                    "citation_identity",
                                    "workflow_identity",
                                )
                            )
                            in (
                                ARCHIVED_STAGE_REPAIR_IDENTITIES,
                                ARCHIVED_STAGE_REPAIR_VIT12,
                                ARCHIVED_STAGE_REPAIR_VIT13,
                                ARCHIVED_STAGE_REPAIR_PRE_CAPACITY_RETRY,
                            )
                        )
                        or archived_stream
                    )
                )
            )
            or "citation_contract" not in value
        ):
            raise ValueError("stage_repair_identity_invalid")
    if "initial_capacity_recovery" in value:
        expected.update(
            ("initial_capacity_recovery", "initial_capacity_recovery_identity")
        )
        if (
            value["initial_capacity_recovery"] != INITIAL_CAPACITY_RECOVERY
            or value.get("initial_capacity_recovery_identity")
            != initial_capacity_identity()
        ):
            raise ValueError("initial_capacity_recovery_identity_invalid")
    if "responses_transport" in value:
        from litagent.llm.responses import STREAM_CONTRACT, transport_identity

        expected.update(("responses_transport", "responses_transport_identity"))
        if value["responses_transport"] != STREAM_CONTRACT or (
            not archived_stream
            and value.get("responses_transport_identity") != transport_identity()
        ):
            raise ValueError("responses_transport_identity_invalid")
    if "global_context_tokens" in value:
        expected.add("global_context_tokens")
        if type(value["global_context_tokens"]) is not int:
            raise ValueError("global_context_tokens_invalid")
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
        {
            k: v
            for k, v in value.items()
            if k
            not in {
                "contract",
                "workflow_identity",
                "citation_identity",
                "candidate_recovery_identity",
                "initial_capacity_recovery_identity",
                "responses_transport_identity",
            }
        }
    )
    if limits.contract != contract:
        raise ValueError("run_limits_profile_mismatch")
    return {
        "contract": contract,
        **limits.model_dump(),
        **(
            {
                "initial_capacity_recovery_identity": value[
                    "initial_capacity_recovery_identity"
                ]
            }
            if limits.initial_capacity_recovery
            else {}
        ),
        **(
            {"responses_transport_identity": value["responses_transport_identity"]}
            if limits.responses_transport
            else {}
        ),
        **(
            {"candidate_recovery_identity": value["candidate_recovery_identity"]}
            if limits.candidate_recovery
            else {}
        ),
        **(
            {"citation_identity": value["citation_identity"]}
            if limits.citation_contract
            else {}
        ),
        **(
            {"workflow_identity": value["workflow_identity"]}
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
    from litagent.inline_citations import identity
    from litagent.llm.responses import transport_identity

    return {
        "contract": limits.contract,
        **limits.model_dump(),
        **(
            {"initial_capacity_recovery_identity": initial_capacity_identity()}
            if limits.initial_capacity_recovery
            else {}
        ),
        **(
            {"responses_transport_identity": transport_identity()}
            if limits.responses_transport
            else {}
        ),
        **(
            {"candidate_recovery_identity": stage_repair.identity()}
            if limits.candidate_recovery
            else {}
        ),
        **({"citation_identity": identity()} if limits.citation_contract else {}),
        **(
            {
                "workflow_identity": review_policy.identity(
                    inline=bool(limits.citation_contract)
                )
            }
            if limits.contract == review_policy.LIMITS_CONTRACT
            else {}
        ),
    }


def envelope_run_limits(envelope=None, *, archived=False):
    """Resolve the same frozen request and repair ceilings at every consumer."""
    value = (envelope or {}).get("run_limits")
    if value is None:
        return dict(LEGACY)
    checked = validate_run_limits(value, archived=archived)
    return {
        **LEGACY,
        **{
            key: checked[key]
            for key in (*UPLIFT, "global_context_tokens")
            if key in checked
        },
    }


def repair_allowances(envelope, *, archived=False):
    """Return the independent quotas of the explicitly selected native profile."""
    limits = (envelope or {}).get("run_limits") or {}
    if review_policy.enabled(envelope):
        validate_run_limits(limits, archived=archived)
        if not envelope.get("semantic_engine"):
            raise ValueError("extended_repair_requires_native_ragas")
        return dict(review_policy.ALLOWANCES)
    if limits.get("contract") not in (EXTENDED_CONTRACT, FORMAT_CONTRACT):
        return None
    validate_run_limits(limits, archived=archived)
    if not envelope.get("semantic_engine"):
        raise ValueError("extended_repair_requires_native_ragas")
    return dict(REPAIR_ALLOWANCES)


def extended_repair_stages(envelope, repairs, *, archived=False):
    """Reserve every permitted edit, summary and native Evaluation round."""
    limits = repair_allowances(envelope, archived=archived)
    if limits is None:
        return None
    if repairs != sum(limits.values()):
        raise ValueError("extended_repair_full_allocation_required")
    if review_policy.enabled(envelope):
        from litagent.eval.engine_policy import reserved_phase_batches

        counts = review_policy.stage_counts(reserved_phase_batches(envelope))
        from litagent.eval.native_capacity import selected

        if selected(envelope):
            counts["capacity/inventory"] = limits["capacity"] * reserved_phase_batches(
                envelope
            )
        return counts
    batches = (
        envelope["body_max_units"]
        + 2 * envelope["summary_max_units_per_section"]
        + envelope["semantic_engine"]["batch_size"]
        - 1
    ) // envelope["semantic_engine"]["batch_size"]
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


def format_recovery_stages(envelope, *, archived=False):
    """Reserve two format calls and one exact regeneration per authored response."""
    limits = (envelope or {}).get("run_limits") or {}
    if limits.get("syntax_policy") != FORMAT_JSON_POLICY:
        return {}
    limits = repair_allowances(envelope, archived=archived)
    counts = extended_repair_stages(envelope, sum(limits.values()), archived=archived)
    if review_policy.enabled(envelope):
        return review_policy.format_stages(counts)
    return {
        "format_body": ("body", 3 * 5),
        "format_summary": ("summary", 3 * counts["summary"]),
        "format_repair": ("repair", 3 * counts["repair"]),
    }
