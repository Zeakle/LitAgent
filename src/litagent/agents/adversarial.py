"""Coordinate evidence-aware synthesis, review, and revision rounds."""

from __future__ import annotations

import asyncio
import hashlib
import json
from collections.abc import Mapping
from contextlib import asynccontextmanager
from typing import Any

from litagent.agents.reviewer import ReviewerWorker
from litagent.agents.synthesis import SynthesisWorker
from litagent.context.evidence_selector import EvidenceSelection
from litagent.evidence import build_citation_aliases
from litagent.llm.client import BaseLLMClient
from litagent.logging import get_logger
from litagent.observability.citations import (
    decode_citations_traced,
    normalize_citations_traced,
)
from litagent.observability.context import reset_task_id, set_task_id
from litagent.orchestrator.scheduler import Worker
from litagent.orchestrator.task_graph import SubTask
from litagent.quality import GroundedRequestBudgetError
from litagent.review_structure import (
    assemble_methodology,
    validate_authored_methodology,
)
from litagent.safety.budget import CostBudget
from litagent.workbench import ReportLanguage, is_strict_policy

logger = get_logger("agents.adversarial")


class AdversarialReviewWorker(Worker):
    """Refine a survey through bounded synthesis and review rounds."""

    def __init__(
        self,
        llm: BaseLLMClient,
        synthesis: SynthesisWorker,
        reviewer: ReviewerWorker,
        max_rounds: int = 3,
        pass_threshold: float = 0.8,
        thinking_mode: str = "disabled",
        trace_hook=None,
    ):
        """Initialize the adversarial review worker."""
        if thinking_mode not in {"provider_default", "enabled", "disabled"}:
            raise ValueError("unsupported thinking_mode")
        self._synthesis = synthesis
        self._reviewer = reviewer
        self._llm = llm
        self._max_rounds = max_rounds
        self._pass_threshold = pass_threshold
        self._thinking_mode = thinking_mode
        self._trace_hook = trace_hook

    @property
    def agent_type(self) -> str:
        """Return the task-graph agent type handled by this worker."""
        return "adversarial_review"

    def _emit(self, event: str, data: dict) -> None:
        """Emit a trace event."""
        if self._trace_hook:
            try:
                self._trace_hook(event, data)
            except Exception as e:
                logger.debug(f"Trace hook failed for '{event}': {e}")

    @asynccontextmanager
    async def _sub_span(self, parent_task_id: str, component: str, round_num: int):
        """Trace one synthesis or review round as a nested span."""
        sub_tid = f"{parent_task_id}:{component}:r{round_num}"
        self._emit(
            "subspan.start",
            {
                "task_id": sub_tid,
                "parent_task_id": parent_task_id,
                "name": f"{component}.r{round_num}",
                "round": round_num,
            },
        )
        token = set_task_id(sub_tid)
        error = None
        error_type = None
        try:
            yield
        except asyncio.CancelledError:
            error = "cancelled"
            error_type = "CancelledError"
            raise
        except Exception as e:
            error = str(e)
            error_type = type(e).__name__
            raise
        finally:
            reset_task_id(token)
            self._emit(
                "subspan.end",
                {"task_id": sub_tid, "error": error, "error_type": error_type},
            )

    async def execute(self, task: SubTask) -> Any:
        """Run synthesis and bounded evidence-aware review rounds."""
        # Initial synthesis is required; only later failures may return a draft.
        async with self._sub_span(task.task_id, "synthesis", 1):
            synthesis_result = await self._synthesis.execute(task)
        draft = synthesis_result.get("draft")
        if not isinstance(draft, str) or not draft.strip():
            raise ValueError("synthesis returned an empty draft")
        report_language: ReportLanguage = task.input_data.get("report_language", "en")
        review_structure = task.input_data.get("review_structure", "legacy")
        review_methodology = synthesis_result.get("review_methodology", {})
        strict = is_strict_policy(task.input_data.get("quality_policy"))
        strict_metadata = (
            {
                k: synthesis_result.get(k)
                for k in (
                    "verification",
                    "body_quality",
                    "analysis_plan",
                    "quality_policy",
                    "fact_cards",
                    "evidence_snapshot_sha256",
                    "coverage_requirements",
                    "coverage_matrix",
                    "extraction_verification",
                    "writing_bindings",
                    "targeted_repair_state",
                )
                if k
                not in {
                    "targeted_repair_state",
                    "verification",
                    "body_quality",
                    "analysis_plan",
                }
                or k in synthesis_result
            }
            if strict
            else {}
        )
        degradation_reasons = [
            str(item)
            for item in synthesis_result.get("degradation_reasons", [])
            if isinstance(item, str) and item
        ]

        selection_raw = synthesis_result.get("evidence_selection")
        try:
            async with self._sub_span(task.task_id, "evidence_selection", 1):
                if not isinstance(selection_raw, Mapping):
                    raise ValueError(
                        "synthesis evidence_selection is missing or not a mapping"
                    )
                evidence_selection = EvidenceSelection.from_dict(selection_raw)
        except (ValueError, TypeError) as e:
            logger.warning("Stopping review: malformed evidence_selection: %s", e)
            error = {"reason_code": "evidence_selection_invalid", "message": str(e)}
            # A broken host contract must not become an empty ledger for the LLM.
            return {
                "report_language": report_language,
                "review_structure": review_structure,
                "review_methodology": review_methodology,
                "final_draft": draft,
                "rounds": [],
                "total_rounds": 0,
                "final_score": 0,
                "accepted": False,
                "evidence_selection": {"valid": False, "error": error},
                "degradation_reasons": list(
                    dict.fromkeys([*degradation_reasons, error["reason_code"]])
                ),
            }

        draft = assemble_methodology(
            draft.strip(), review_structure, report_language, review_methodology
        )

        logger.info(
            "Adversarial start: draft=%d chars selection_items=%d/%d",
            len(draft),
            len(evidence_selection.selected_items),
            evidence_selection.candidate_count,
        )

        rounds: list[dict] = []
        reviewed_draft: str | None = None
        seen_drafts = {hashlib.sha256(draft.encode("utf-8")).hexdigest()}

        try:
            for round_num in range(1, self._max_rounds + 1):
                async with self._sub_span(task.task_id, "reviewer", round_num):
                    if round_num == 1:
                        review_task = SubTask(
                            task_id=task.task_id,
                            description=f"Review round {round_num}",
                            agent_type="reviewer",
                            input_data={
                                "query": task.input_data.get("query", ""),
                                "report_language": report_language,
                                "body_quality": task.input_data.get(
                                    "body_quality", "legacy"
                                ),
                                "quality_policy": task.input_data.get(
                                    "quality_policy", "legacy"
                                ),
                                "review_structure": review_structure,
                                "review_methodology": review_methodology,
                                "upstream_results": {
                                    "synthesis": {
                                        "draft": draft,
                                        "coverage_requirements": synthesis_result.get(
                                            "coverage_requirements"
                                        ),
                                        "evidence_selection": (
                                            evidence_selection.to_dict()
                                        ),
                                    },
                                },
                            },
                        )
                        review = await self._reviewer.execute(review_task)
                    else:
                        review_kwargs: dict[str, Any] = {
                            "query": task.input_data.get("query", "")
                        }
                        if strict:
                            review_kwargs["quality_policy"] = task.input_data[
                                "quality_policy"
                            ]
                            review_kwargs["body_quality"] = task.input_data.get(
                                "body_quality", "legacy"
                            )
                            review_kwargs["coverage_requirements"] = (
                                synthesis_result.get("coverage_requirements")
                            )
                        if report_language != "en":
                            review_kwargs["report_language"] = report_language
                        if review_structure != "legacy":
                            review_kwargs.update(
                                review_structure=review_structure,
                                review_methodology=review_methodology,
                            )
                        review = await self._reviewer.review_revision(
                            draft,
                            rounds[-1]["review"],
                            evidence_selection,
                            **review_kwargs,
                        )

                rounds.append(
                    {"round": round_num, "review": review, "draft_length": len(draft)}
                )
                reviewed_draft = draft
                degradation_reasons.extend(
                    str(item)
                    for item in review.get("degradation_reasons", [])
                    if isinstance(item, str) and item
                )

                score = review.get("score", 0)
                verdict = review.get("verdict", "revise")

                ev_passed = (
                    review.get("evidence_compliance", {}).get("passed", False)
                    if isinstance(review.get("evidence_compliance"), dict)
                    else False
                )
                logger.info(
                    f"Round {round_num}: score={score}, verdict={verdict}, "
                    f"evidence_passed={ev_passed}"
                )

                if self._review_passes(review):
                    logger.info(f"Accepted at round {round_num}")
                    break

                if review.get("review_unverified") is True:
                    logger.warning(
                        "Reviewer remained schema-invalid after compact retry; "
                        "preserving the current draft"
                    )
                    break

                if round_num < self._max_rounds:
                    if strict:
                        if not review.get("bound_feedback"):
                            break
                        review_text = json.dumps(
                            review["bound_feedback"], ensure_ascii=False
                        )
                    else:
                        review_text = self._format_review_for_revision(review)
                    try:
                        revise_kwargs: dict[str, Any] = {}
                        if strict:
                            revise_kwargs.update(
                                quality_policy=task.input_data["quality_policy"],
                                body_quality=task.input_data.get(
                                    "body_quality", "legacy"
                                ),
                                fact_cards=synthesis_result.get("fact_cards", {}),
                                coverage_requirements=synthesis_result.get(
                                    "coverage_requirements"
                                ),
                            )
                        revise_kwargs["query"] = task.input_data.get("query", "")
                        if report_language != "en":
                            revise_kwargs["report_language"] = report_language
                        if review_structure != "legacy":
                            revise_kwargs.update(
                                review_structure=review_structure,
                                review_methodology=review_methodology,
                            )
                        messages = self._synthesis.revise(
                            draft,
                            review_text,
                            evidence_selection,
                            **revise_kwargs,
                        )
                    except (ValueError, TypeError) as e:
                        if task.input_data.get("quality_policy") in {
                            "strict_auto_v1",
                            "strict_auto_v2",
                        }:
                            budget = getattr(self._llm, "_cost_budget", None)
                            if isinstance(budget, CostBudget):
                                budget.stop_after_worker_failure()
                        logger.warning(f"revise rejected malformed selection: {e}")
                        break

                    try:
                        async with self._sub_span(
                            task.task_id, "synthesis", round_num + 1
                        ):
                            kwargs: dict[str, Any] = {}
                            if strict:
                                from litagent.quality import (
                                    check_grounded_request_budget,
                                    prepare_grounded_request,
                                )

                                messages, response_format = prepare_grounded_request(
                                    messages
                                )
                                kwargs.update(
                                    max_tokens=20000,
                                    response_format=response_format,
                                    transport_max_attempts=1,
                                )
                                check_grounded_request_budget(
                                    self._llm,
                                    messages,
                                    response_format,
                                    trace_hook=self._trace_hook,
                                    parent_task_id=task.task_id,
                                    stage="revision",
                                )
                            if self._thinking_mode != "provider_default":
                                kwargs["extra_body"] = {
                                    "thinking": {"type": self._thinking_mode}
                                }
                            if strict:
                                resp = await asyncio.wait_for(
                                    self._llm.chat(messages, **kwargs), 180
                                )
                            else:
                                resp = await self._llm.chat(messages, **kwargs)

                        new_draft = (resp.content or "").strip()
                        if strict:
                            from litagent.quality import render_grounded_report_traced

                            revision_bindings = {}
                            new_draft = render_grounded_report_traced(
                                new_draft,
                                synthesis_result["fact_cards"],
                                review_structure,
                                report_language,
                                evidence_selection=evidence_selection,
                                trace_hook=self._trace_hook,
                                parent_task_id=task.task_id,
                                stage="revision",
                                require_native=True,
                                require_source_quotes=True,
                                require_full_sources=task.input_data.get(
                                    "quality_policy"
                                )
                                == "strict_auto_v2",
                                analysis_mode=task.input_data.get("body_quality")
                                == "analysis_v1",
                                require_scope_notices=task.input_data.get(
                                    "quality_policy"
                                )
                                == "strict_auto_v2",
                                bindings=revision_bindings,
                            )
                        if not new_draft:
                            logger.warning(
                                "Revision LLM returned empty content, "
                                "keeping current draft"
                            )
                            degradation_reasons.append("revision_empty")
                            break
                        aliases = build_citation_aliases(
                            list(evidence_selection.selected_items)
                        )
                        decoded = decode_citations_traced(
                            new_draft,
                            aliases,
                            trace_hook=self._trace_hook,
                            parent_task_id=task.task_id,
                            stage="revision",
                            round_num=round_num + 1,
                        ).text
                        candidate = normalize_citations_traced(
                            decoded,
                            evidence_selection.selected_items,
                            trace_hook=self._trace_hook,
                            parent_task_id=task.task_id,
                            stage="revision",
                            round_num=round_num + 1,
                        ).text
                        validate_authored_methodology(candidate, review_structure)
                        candidate = assemble_methodology(
                            candidate,
                            review_structure,
                            report_language,
                            review_methodology,
                        )
                        candidate_hash = hashlib.sha256(
                            candidate.encode("utf-8")
                        ).hexdigest()
                        if candidate_hash in seen_drafts:
                            reason = (
                                "revision_no_progress"
                                if candidate == draft
                                else "revision_cycle_detected"
                            )
                            degradation_reasons.append(reason)
                            self._emit(
                                "revision.no_progress",
                                {
                                    "task_id": task.task_id,
                                    "round": round_num + 1,
                                    "candidate_hash": candidate_hash,
                                    "reason_code": reason,
                                },
                            )
                            break
                        seen_drafts.add(candidate_hash)
                        draft = candidate
                        if strict:
                            revision_bindings["report_sha256"] = candidate_hash
                            strict_metadata["writing_bindings"] = revision_bindings
                    except Exception as e:
                        if (
                            strict
                            and reviewed_draft is not None
                            and isinstance(e, GroundedRequestBudgetError)
                            and e.permits_evaluation(
                                getattr(self._llm, "_cost_budget", None)
                            )
                        ):
                            degradation_reasons.append("writing_phase_budget_exhausted")
                            break
                        if task.input_data.get("quality_policy") in {
                            "strict_auto_v1",
                            "strict_auto_v2",
                        }:
                            budget = getattr(self._llm, "_cost_budget", None)
                            if isinstance(budget, CostBudget):
                                budget.stop_after_worker_failure()
                        logger.warning(
                            f"Revision LLM call failed: {e}, keeping current draft"
                        )
                        degradation_reasons.append("revision_failed")
                        break
                    logger.info(f"Revision {round_num}: {len(draft)} chars")
        except Exception as e:
            preserve = (
                strict
                and reviewed_draft is not None
                and isinstance(e, GroundedRequestBudgetError)
                and e.permits_evaluation(getattr(self._llm, "_cost_budget", None))
            )
            if preserve:
                degradation_reasons.append("writing_phase_budget_exhausted")
            elif task.input_data.get("quality_policy") in {
                "strict_auto_v1",
                "strict_auto_v2",
            }:
                budget = getattr(self._llm, "_cost_budget", None)
                if isinstance(budget, CostBudget):
                    budget.stop_after_worker_failure()
            logger.warning(
                f"Adversarial loop aborted ({e}), returning current draft as fallback"
            )
            if not preserve:
                degradation_reasons.append("adversarial_loop_degraded")

        if reviewed_draft is not None and draft != reviewed_draft:
            draft = reviewed_draft
            degradation_reasons.append("unreviewed_revision_discarded")

        last_review = rounds[-1]["review"] if rounds else {}

        sel_summary = {
            "valid": True,
            "candidate_count": evidence_selection.candidate_count,
            "selected_count": len(evidence_selection.selected_items),
            "section_evidence_ids": evidence_selection.section_evidence_ids,
            "method_by_section": evidence_selection.method_by_section,
            "omitted_count": evidence_selection.omitted_count,
            "estimated_tokens": evidence_selection.estimated_tokens,
        }

        return {
            **strict_metadata,
            "report_language": report_language,
            "review_structure": review_structure,
            "review_methodology": review_methodology,
            "final_draft": draft,
            "rounds": rounds,
            "total_rounds": len(rounds),
            "final_score": last_review.get("score", 0),
            "accepted": (self._review_passes(last_review) if last_review else False),
            "evidence_selection": sel_summary,
            "degradation_reasons": list(dict.fromkeys(degradation_reasons)),
        }

    def _review_passes(self, review: dict[str, Any]) -> bool:
        """Accept only when score, verdict, evidence, and language checks pass."""
        compliance = review.get("evidence_compliance")
        language = review.get("language_compliance", {"passed": True})
        return (
            review.get("score", 0) >= self._pass_threshold
            and review.get("verdict") == "accept"
            and isinstance(compliance, dict)
            and compliance.get("passed") is True
            and isinstance(language, dict)
            and language.get("passed") is True
        )

    def _format_review_for_revision(self, review: dict) -> str:
        """Render structured review feedback for the revision prompt."""
        parts: list[str] = []
        if review.get("weaknesses"):
            parts.append(
                "Weaknesses:\n" + "\n".join(f"- {w}" for w in review["weaknesses"])
            )
        if review.get("issues"):
            parts.append(
                "Issues:\n"
                + "\n".join(
                    f"- [{i.get('severity', 'minor')}] "
                    f"{i.get('section', '')}: {i.get('issue', '')}"
                    for i in review["issues"]
                )
            )
        if review.get("missing_coverage"):
            parts.append(
                "Missing:\n" + "\n".join(f"- {m}" for m in review["missing_coverage"])
            )

        compliance = review.get("evidence_compliance")
        if isinstance(compliance, dict):
            unknown = compliance.get("unknown_evidence_ids", [])
            if isinstance(unknown, list) and unknown:
                parts.append(
                    "Unknown evidence IDs (remove or rebind these):\n"
                    + "\n".join(f"- [E:{eid}]" for eid in unknown)
                )
            malformed = compliance.get("malformed_evidence_refs", [])
            if isinstance(malformed, list) and malformed:
                parts.append(
                    "Malformed evidence markers (rewrite every ID as its own "
                    "[E:<id>] marker):\n"
                    + "\n".join(f"- {marker}" for marker in malformed)
                )
            placement = compliance.get("citation_placement_issues", [])
            if isinstance(placement, list) and placement:
                parts.append(
                    "Citation placement: the allowed evidence pool is not a mandatory "
                    "citation list. Remove inventory dumping from summaries; cite only "
                    "direct support next to each scientific assertion. Do not relocate "
                    "unrelated references or delete core supported facts.\n"
                    + "\n".join(
                        f"- {item.get('section', '?')}: {item.get('reason_code', '')}"
                        for item in placement
                        if isinstance(item, Mapping)
                    )
                )
            unsupported = compliance.get("unsupported_claims", [])
            if isinstance(unsupported, list) and unsupported:
                safe_claims = [c for c in unsupported if isinstance(c, Mapping)]
                if safe_claims:
                    parts.append(
                        "Unsupported claims:\n"
                        + "\n".join(
                            f"- [{c.get('section', '?')}] {c.get('claim', '')}"
                            f"  reason: {c.get('reason', '')}"
                            f"  action: {c.get('action', 'verify')}"
                            f"  source_quote: {c.get('source_quote', 'unmapped')}"
                            f"  source_start: {c.get('source_start', 'unmapped')}"
                            f"  source_end: {c.get('source_end', 'unmapped')}"
                            "  offset_scope: frozen canonical report before citation alias encoding"
                            f"  existing_evidence_ids: {c.get('existing_evidence_ids', [])}"
                            for c in safe_claims
                        )
                    )

        return "\n\n".join(parts) if parts else "No specific feedback."
