"""Freeze the explicitly selected semantic engine without changing old defaults."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

from litagent.eval.ragas_global_wire import UNIT_CONTRACT

CONTRACT = "ragas-cited-faithfulness-v1"
ANCHORED_GLOBAL_REVIEW = "anchored-global-findings-v2"
KEYED_GLOBAL_UNITS = "ragas-global-unit-keys-v1"
KEYED_GLOBAL_ANCHORS = "ragas-global-unit-keys-v2"
KEYED_GLOBAL_QUOTED_OMISSIONS = "ragas-global-unit-keys-v3"
KEYED_ASSESSMENT = "ragas-assessment-statement-keys-v1"
ARCHIVED_SHORT_ANCHOR_ENGINES = {
    (None, None): "45541dbf153f36ebdee1df6a472db12acc1c4df48f45c1d160b4bf1b26a7c711",
    (None, KEYED_ASSESSMENT): (
        "cc8db5429de65932a3ff157507bcc69e4ba742883997aba70d0166400cb7c5e3"
    ),
    (UNIT_CONTRACT, None): (
        "76ca1acaf9108d85e14d9ef68388f9ed0339b1bafe5f93b7de6884d9c68acbb7"
    ),
    (UNIT_CONTRACT, KEYED_ASSESSMENT): (
        "baf8740e30ce0692f1fde7dc2329f3ad4717b858673a1796deb6fb529b87231e"
    ),
}
# Retained for offline verification of settled 076/077 plans only.
ARCHIVED_ENGINE_IDENTITY = (
    "87ac486054ab2be4e45f0491d5cbef2f440917c2f851e6a9ea7ec218727dc481"
)
ARCHIVED_PREFIXED_ENGINE_IDENTITY = (
    "ba31dfb2e3cdc2a1fdb1864ac4795d1b12ab20e1ad64497f1c209a72f48c883c"
)
# Exact pre-unit-view descriptors remain valid only for settled accounting reads.
ARCHIVED_FLAT_ENGINES = {
    None: "1175d0f0ea67c64b07be98befd2fd3a6e312d4e22dd24cb0d03b6c4867d99fee",
    ANCHORED_GLOBAL_REVIEW: "2ec3bae96542901a8310c34d1175be9a75f036ad3638d81a4885e51c14bb842c",
}
# Exact descriptors used by the previous list-based provider response.
ARCHIVED_LIST_ENGINES = {
    (None, None): "fac02b3ace24da435b109c7a5240ce6f7fd1233e2ccc50f2b81ecc59fafd6046",
    (ANCHORED_GLOBAL_REVIEW, None): (
        "b8de89f42bf81d10410ca0b4e6e0a5eeac314b69fc3e85720398872ae6330df9"
    ),
    (ANCHORED_GLOBAL_REVIEW, UNIT_CONTRACT): (
        "11f25b70eb169f587bf33a7fd349282108fbd0983e8db024948be4aa1d988bce"
    ),
}
# Last descriptors that required the model to echo the statement text.
ARCHIVED_ECHO_ENGINES = {
    (
        None,
        None,
        None,
    ): "b0fe0888dc60a4edb7340546d7ce8f4e0673b7ee38ee3d943d926bc0ecef6e3a",
    (
        ANCHORED_GLOBAL_REVIEW,
        None,
        None,
    ): "920101eb3ca2e669c9681dde579c54f959baadb7822965ff64c8a3b0196dc680",
    (
        ANCHORED_GLOBAL_REVIEW,
        UNIT_CONTRACT,
        None,
    ): "1ed811fe0a7452d68ed12354fb0e3ac8f478f94154161e6aebceed92ad3df3cb",
    (
        ANCHORED_GLOBAL_REVIEW,
        None,
        KEYED_GLOBAL_UNITS,
    ): "6fc064ac1ac0454ce0bdcda531b8a7200bf9310765181ea39138190ac0e9e83e",
    (
        ANCHORED_GLOBAL_REVIEW,
        UNIT_CONTRACT,
        KEYED_GLOBAL_UNITS,
    ): "368420178225aead830db091670f1137d347f67b65d497253dd3e9b50ca672db",
}
# Previous implementation by findings, input and unit wire; each pair is
# (text-echo assessment, keyed assessment).
ARCHIVED_LONG_ANCHOR_ENGINES = {
    (None, None, None): (
        "637842905aa790b3e142dca63ae1500e3c8c851b4e5de26f2ff93fab1b2ac1a4",
        "ab425de53746a97538f0ffaa662d476b3b707f135b1b0bf7573359f1672c34bd",
    ),
    (ANCHORED_GLOBAL_REVIEW, None, None): (
        "84dcda43d65fbfe76b38dd2d9d35f7a8ab3efbb60b3e75b950fcefc057ae58ac",
        "6c62c9cd73abc8a73f63135ff82bc935f540f739cd67ccb887c0cf5b84ad8390",
    ),
    (ANCHORED_GLOBAL_REVIEW, UNIT_CONTRACT, None): (
        "240156ec0f9267235b7bd9257729b4d15eaf077bc928977c0df38335213d568a",
        "537073bfe452a01b4a585b69c163c9b5c9f0ca230cb7ce5c137fbb63b2c9570d",
    ),
    (ANCHORED_GLOBAL_REVIEW, None, KEYED_GLOBAL_UNITS): (
        "80db65c283c84ee219917650c025789c03e93221f78f4e511de525dae4540a49",
        "2ac6532b5ce606987264ba8ab8e1b59fce52e35af0870fa14b83f39a1d7de5f6",
    ),
    (ANCHORED_GLOBAL_REVIEW, UNIT_CONTRACT, KEYED_GLOBAL_UNITS): (
        "58784bcd93c50e5d5b6589be4949b61006190412b4ff03748ab4f350c7d4b7e6",
        "6dbc1c8c9af824e64154e696d8406401daa12f0bf3e1dc1035ecadce1e585ce4",
    ),
}


def engine_descriptor(
    batch_size=16,
    *,
    global_findings_contract=None,
    global_input_contract=None,
    global_unit_response_contract=None,
    assessment_response_contract=None,
):
    """Bind the installed processor and the complete adapter implementation."""
    if type(batch_size) is not int or not 1 <= batch_size <= 20:
        raise ValueError("ragas_batch_size_invalid")
    if global_findings_contract not in (None, ANCHORED_GLOBAL_REVIEW):
        raise ValueError("ragas_global_findings_contract_invalid")
    if global_input_contract is not None and (
        global_input_contract != UNIT_CONTRACT
        or global_findings_contract != ANCHORED_GLOBAL_REVIEW
    ):
        raise ValueError("ragas_global_input_contract_invalid")
    if global_unit_response_contract is not None and (
        global_unit_response_contract
        not in (KEYED_GLOBAL_UNITS, KEYED_GLOBAL_ANCHORS, KEYED_GLOBAL_QUOTED_OMISSIONS)
        or global_findings_contract != ANCHORED_GLOBAL_REVIEW
    ):
        raise ValueError("ragas_global_unit_response_contract_invalid")
    if assessment_response_contract not in (None, KEYED_ASSESSMENT):
        raise ValueError("ragas_assessment_response_contract_invalid")
    current = version("ragas")
    if current != "0.4.3":
        raise ValueError("ragas_dependency_version_mismatch")
    files = [
        Path(__file__).with_name(name)
        for name in (
            "ragas_adapter.py",
            "ragas_native.py",
            "ragas_repair.py",
            "ragas_global_wire.py",
        )
    ]
    files.extend([Path(__file__), Path(__file__).parents[1] / "writing_flat.py"])
    identity = hashlib.sha256(
        json.dumps(
            [
                CONTRACT,
                current,
                global_findings_contract,
                *([global_input_contract] if global_input_contract is not None else []),
                *(
                    [global_unit_response_contract]
                    if global_unit_response_contract is not None
                    else []
                ),
                *(
                    [assessment_response_contract]
                    if assessment_response_contract is not None
                    else []
                ),
                *[hashlib.sha256(p.read_bytes()).hexdigest() for p in files],
            ]
        ).encode()
    ).hexdigest()
    return {
        "contract": CONTRACT,
        "ragas_version": current,
        "identity": identity,
        "batch_size": batch_size,
        **(
            {"assessment_response_contract": assessment_response_contract}
            if assessment_response_contract is not None
            else {}
        ),
        **(
            {"global_findings_contract": global_findings_contract}
            if global_findings_contract is not None
            else {}
        ),
        **(
            {"global_unit_response_contract": global_unit_response_contract}
            if global_unit_response_contract is not None
            else {}
        ),
        **(
            {"global_input_contract": global_input_contract}
            if global_input_contract is not None
            else {}
        ),
    }


def validate_engine(value, *, archived=False):
    """Reject changed semantics rather than silently restoring an old checkpoint."""
    if not isinstance(value, dict):
        raise ValueError("ragas_engine_identity_invalid")
    current = engine_descriptor(
        value.get("batch_size"),
        global_findings_contract=value.get("global_findings_contract"),
        global_input_contract=value.get("global_input_contract"),
        global_unit_response_contract=value.get("global_unit_response_contract"),
        assessment_response_contract=value.get("assessment_response_contract"),
    )
    historical = (
        ARCHIVED_PREFIXED_ENGINE_IDENTITY
        if current.get("global_findings_contract") == ANCHORED_GLOBAL_REVIEW
        else ARCHIVED_ENGINE_IDENTITY
    )
    archived_ids = []
    if archived:
        findings = current.get("global_findings_contract")
        input_contract = current.get("global_input_contract")
        units_contract = current.get("global_unit_response_contract")
        wire = findings, input_contract, units_contract
        keyed_assessment = "assessment_response_contract" in current
        if units_contract == KEYED_GLOBAL_ANCHORS:
            archived_ids.append(
                ARCHIVED_SHORT_ANCHOR_ENGINES[
                    input_contract, current.get("assessment_response_contract")
                ]
            )
        elif wire in ARCHIVED_LONG_ANCHOR_ENGINES:
            echo_id, keyed_id = ARCHIVED_LONG_ANCHOR_ENGINES[wire]
            archived_ids.append(keyed_id if keyed_assessment else echo_id)
            if not keyed_assessment:
                archived_ids.append(ARCHIVED_ECHO_ENGINES[wire])
                if units_contract is None:
                    archived_ids.append(ARCHIVED_LIST_ENGINES[findings, input_contract])
                    if input_contract is None:
                        archived_ids.extend(
                            (historical, ARCHIVED_FLAT_ENGINES[findings])
                        )
    if value != current and not any(
        value == {**current, "identity": identity} for identity in archived_ids
    ):
        raise ValueError("ragas_engine_identity_invalid")
    return dict(value)


def configured_engine(config):
    """Read only the server configuration, never a user-supplied endpoint."""
    evaluation = getattr(config, "eval", None)
    selected = getattr(evaluation, "semantic_engine", None)
    input_contract = getattr(evaluation, "native_global_input", None)
    unit_response = getattr(evaluation, "native_global_unit_response", None)
    if selected is None:
        if input_contract is not None:
            raise ValueError("ragas_global_input_contract_invalid")
        if unit_response is not None:
            raise ValueError("ragas_global_unit_response_contract_invalid")
        return None
    if selected != CONTRACT:
        raise ValueError("ragas_engine_invalid")
    from litagent.run_limits import FORMAT_JSON_POLICY, configured_run_limits

    limits = configured_run_limits(config) or {}
    return engine_descriptor(
        config.eval.strict_batch_size,
        global_findings_contract=(
            ANCHORED_GLOBAL_REVIEW
            if limits.get("syntax_policy") == FORMAT_JSON_POLICY
            else None
        ),
        global_input_contract=input_contract,
        global_unit_response_contract=unit_response,
        assessment_response_contract=KEYED_ASSESSMENT,
    )


def anchored_native_review(envelope):
    """Select the new report-review wire only from the frozen engine descriptor."""
    engine = (envelope or {}).get("semantic_engine", {})
    return engine.get("global_findings_contract") == ANCHORED_GLOBAL_REVIEW


def unit_native_review(envelope):
    """Select the colocated wire only from the explicitly frozen descriptor."""
    engine = (envelope or {}).get("semantic_engine", {})
    return engine.get("global_input_contract") == UNIT_CONTRACT


def native_unit_response_contract(envelope):
    """Select the frozen global unit and finding response protocol."""
    engine = (envelope or {}).get("semantic_engine", {})
    return engine.get("global_unit_response_contract")


def phase_batches(envelope):
    """Reserve complete per-unit generator and NLI rounds including summaries."""
    validate_engine(envelope["semantic_engine"], archived=True)
    units = envelope["body_max_units"] + 2 * envelope["summary_max_units_per_section"]
    size = envelope["semantic_engine"]["batch_size"]
    if (envelope.get("capacity") or {}).get("batch_size", 16) != size:
        raise ValueError("ragas_capacity_batch_mismatch")
    return (units + size - 1) // size


def native_writer_budget_plan(*args, **kwargs):
    """Use one server-owned writer preset for new native workbench runs."""
    from litagent.closeout import PRESET
    from litagent.writing_v2 import writer_budget_plan

    if (
        kwargs.get("semantic_engine") is None
        or kwargs.get("budget_policy") != "planned-budget-v7"
    ):
        raise ValueError("ragas_engine_requires_v7")
    return writer_budget_plan(*args, **{**kwargs, **PRESET})
