"""Freeze the explicitly selected semantic engine without changing old defaults."""

import hashlib
import json
from importlib.metadata import version
from pathlib import Path

from litagent.eval.conflict_review import CONTRACT as CONFLICT_ONLY_SCOPE
from litagent.eval.ragas_global_wire import (
    ORDERED_UNIT_CONTRACT,
    STATUS_CONTRACT,
    UNIT_CONTRACT,
)
from litagent.eval.responsibilities import REPORT_REVIEW_SCOPE

CONTRACT = "ragas-cited-faithfulness-v1"
ANCHORED_GLOBAL_REVIEW = "anchored-global-findings-v2"
KEYED_GLOBAL_UNITS = "ragas-global-unit-keys-v1"
KEYED_GLOBAL_ANCHORS = "ragas-global-unit-keys-v2"
KEYED_GLOBAL_QUOTED_OMISSIONS = "ragas-global-unit-keys-v3"
KEYED_GLOBAL_LOCATED_OMISSIONS = "ragas-global-unit-keys-v4"
KEYED_ASSESSMENT = "ragas-assessment-statement-keys-v1"
BOUND_ORGANIZATION = "host-report-heading-claims-v1"
DIRECT_INVENTORY_PROMPT = "ragas-direct-inventory-v1"
STAGE_BATCH_LIMITS = "ragas-stage-batch-limits-v1"
PRE_INLINE_FILE_HASHES = (
    "b7e6c8746519096c8b42d47791a43679f740dff6e712d56a2171aa17837730a2",
    "1fbadf4d02d9a748b44ef35e0554580c4cc7e72da624b65df888c36b08c1808b",
    "6c03c2336adcccaa9ba2b1ac594e208a16bc3d29e89e58ddb6149b975417efbc",
    "078234ea7b2266d230a4da15f27b7891b75db03c5b20397b7662f95cb73dea19",
    "731f8cffeff389df497e990732241a311f306420968f312371dbe39f7e2aac1c",
    "86ef11d49ec17a8943585f56fafc432dd6e300dd9eff3be32f04f1dfb5e77a24",
)
# The contradiction-only branches do not change any older request contract.
PRE_CONFLICT_ONLY_FILE_HASHES = {
    "ragas_native.py": "c31f27f3018136639bfb8483e8d4623053e4e22568db35095647c1a2a143ef95",
    "ragas_repair.py": "959aab5725451a5c3e0c88d9d5750195c8ab289d47dfe049f152ad5e3b5e0c9a",
    "engine_policy.py": "e15608ee0bfb86b3d6feebb9ab975b3fd5c716bde69da22f31a3ed69866882d2",
}
# Profiles whose inventory prompt did not change keep their existing wire identity.
PRE_DIRECT_INVENTORY_FILE_HASHES = {
    "ragas_adapter.py": "81e11a74cfe54a3f4a807813a5ec406f7ba89e6e7bf2e3c8a5e74e46523c9395",
    "ragas_native.py": "4bbb88c8bb4a6e2f99bd1e4c09bc2ddc598701f3c84d2628404c1bd10c12ef21",
    "engine_policy.py": "80583368fdee3a06099b009f04eb498e1df189db6fc26eb29346ce7ff7d9a05e",
}
PRE_STAGE_BATCH_FILE_HASHES = {
    "ragas_adapter.py": "e454d7ea1cf980c03697c34e475ed53de3fdd0ca0e8ef6bccfd01fa66f47a911",
    "ragas_native.py": "93271806308feaabf43e9daca08a085f4450b961380e44b463fa2599fdad79a2",
    "engine_policy.py": "f7be35da453853647a212bc43f08f5839418e22c27e12d63b1c756da48e0dbc8",
}
# The first inline ViT trials used provider NLI even for empty citation scopes.
ARCHIVED_INLINE_VIT_ENGINE = {
    "contract": CONTRACT,
    "ragas_version": "0.4.3",
    "identity": "0ce30ccbdd733881c9b9d57907ec5ebd0623ce68e32b4fca3fb74ab16d1120f6",
    "batch_size": 20,
    "citation_contract": "flat-writer-v2",
    "assessment_response_contract": KEYED_ASSESSMENT,
    "global_findings_contract": ANCHORED_GLOBAL_REVIEW,
    "global_unit_response_contract": KEYED_GLOBAL_LOCATED_OMISSIONS,
}
# The third inline trial used host negatives but still transmitted status labels.
ARCHIVED_INLINE_UNCITED_ENGINE = {
    **ARCHIVED_INLINE_VIT_ENGINE,
    "identity": "7ae695c0f37c9eb34b0574d44eade254c3e7e02d59c38f9c7a222fc37c4be8f9",
}
ARCHIVED_INLINE_STATUS_ENGINE = {
    **ARCHIVED_INLINE_VIT_ENGINE,
    "identity": "95b3b8e103819fe0b458da72b7f97c965f6eeff29c54fcd21d4680999f0f851e",
    "global_status_contract": STATUS_CONTRACT,
}
ARCHIVED_INLINE_EDITORIAL_ENGINES = (
    {
        **ARCHIVED_INLINE_STATUS_ENGINE,
        "identity": "53e37dde502c1931d6be9b386034621f229d85b402e8b1516b0b7688d4c9c98c",
    },
    {
        **ARCHIVED_INLINE_STATUS_ENGINE,
        "identity": "c7d26e5ba348d36a885464f2382b890cfc772ae5edfa17faa22ca51bfee3e2e2",
        "capacity_recovery_contract": "native-capacity-recovery-v1",
    },
)
ARCHIVED_INLINE_SUMMARY_ENGINE = {
    **ARCHIVED_INLINE_STATUS_ENGINE,
    "identity": "72bfafaa18f93573ef1ae10ea1f2b011ae904730f576c703f3eb6d3d85bc88f9",
    "capacity_recovery_contract": "native-capacity-recovery-v1",
}
ARCHIVED_INLINE_DEPENDENCY_ENGINE = {
    **ARCHIVED_INLINE_SUMMARY_ENGINE,
    "identity": "47fef048b4ccf36b5a65f61e14c7cb95ef17f7eda596b62fe6dea223bcf75178",
}
ARCHIVED_INLINE_RETRY_ENGINE = {
    **ARCHIVED_INLINE_DEPENDENCY_ENGINE,
    "identity": "c2e8738f569170dee0e02181d8a15c6371b671952f89985156fe4633ac8232b8",
}
ARCHIVED_INLINE_CANDIDATE_ENGINE = {
    **ARCHIVED_INLINE_RETRY_ENGINE,
    "identity": "08d697a5a8db08699174051f6f3303563a8dfccb150e0e5ede4eb126dd1b0e7d",
}
# Exact pre-v4 descriptors are readable, never resumable under a new identity.
ARCHIVED_QUOTED_OMISSION_ENGINES = {
    (None, None): "3c117b941d70981dced95f8e7c5e47b8ff73da8ef2713decbb25a63cd3946a9b",
    (None, KEYED_ASSESSMENT): (
        "830412b7ea9ffbff8851115c4c5f05af22ce11f8925870c2ced99ace493dbf06"
    ),
    (UNIT_CONTRACT, None): (
        "89aaabba62252d84c9b9eab4903a6e5fa34da65c56ccd6517ceb8397afd5ed86"
    ),
    (UNIT_CONTRACT, KEYED_ASSESSMENT): (
        "382e2908e0cf7c71648b6353f9cb8ec3947e4d9441add4c1bd172747cf9f93bf"
    ),
}
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
    global_scope=None,
    global_findings_contract=None,
    global_input_contract=None,
    global_unit_response_contract=None,
    assessment_response_contract=None,
    citation_contract=None,
    capacity_recovery_contract=None,
    organization_contract=None,
    _previous_inventory_prompt=False,
    _previous_stage_batch_policy=False,
):
    """Bind the installed processor and the complete adapter implementation."""
    from litagent import inline_citations
    from litagent.eval.native_capacity import CONTRACT as CAPACITY_RECOVERY
    from litagent.eval.native_capacity import INVENTORY_CONTEXT_CONTRACT

    scope_modules = {
        "question-coverage-v1": "question_coverage.py",
        REPORT_REVIEW_SCOPE: "responsibilities.py",
        CONFLICT_ONLY_SCOPE: "conflict_review.py",
    }
    if global_scope is not None and global_scope not in scope_modules:
        raise ValueError("ragas_global_scope_invalid")
    if capacity_recovery_contract not in (None, CAPACITY_RECOVERY) or (
        capacity_recovery_contract is not None and citation_contract is None
    ):
        raise ValueError("native_capacity_contract_invalid")

    if organization_contract not in (None, BOUND_ORGANIZATION) or (
        organization_contract is not None
        and global_unit_response_contract
        not in (KEYED_GLOBAL_QUOTED_OMISSIONS, KEYED_GLOBAL_LOCATED_OMISSIONS)
    ):
        raise ValueError("ragas_organization_contract_invalid")
    if citation_contract not in (None, inline_citations.CONTRACT):
        raise ValueError("ragas_citation_contract_invalid")
    if type(batch_size) is not int or not 1 <= batch_size <= 20:
        raise ValueError("ragas_batch_size_invalid")
    if global_findings_contract not in (None, ANCHORED_GLOBAL_REVIEW):
        raise ValueError("ragas_global_findings_contract_invalid")
    if global_input_contract is not None and (
        global_input_contract not in (UNIT_CONTRACT, ORDERED_UNIT_CONTRACT)
        or global_findings_contract != ANCHORED_GLOBAL_REVIEW
        or (
            global_input_contract == ORDERED_UNIT_CONTRACT and citation_contract is None
        )
    ):
        raise ValueError("ragas_global_input_contract_invalid")
    if global_unit_response_contract is not None and (
        global_unit_response_contract
        not in (
            KEYED_GLOBAL_UNITS,
            KEYED_GLOBAL_ANCHORS,
            KEYED_GLOBAL_QUOTED_OMISSIONS,
            KEYED_GLOBAL_LOCATED_OMISSIONS,
        )
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
    if capacity_recovery_contract:
        files.append(Path(__file__).with_name("native_capacity.py"))
    direct_inventory = (
        global_scope == CONFLICT_ONLY_SCOPE
        and capacity_recovery_contract is None
        and not _previous_inventory_prompt
    )
    frozen_hashes = {} if direct_inventory else PRE_DIRECT_INVENTORY_FILE_HASHES
    stage_batch_limits = direct_inventory and not _previous_stage_batch_policy
    if direct_inventory and not stage_batch_limits:
        frozen_hashes = PRE_STAGE_BATCH_FILE_HASHES
    if global_scope != CONFLICT_ONLY_SCOPE:
        frozen_hashes = {**frozen_hashes, **PRE_CONFLICT_ONLY_FILE_HASHES}
    file_hashes = [
        (
            frozen_hashes[p.name]
            if p.name in frozen_hashes
            else hashlib.sha256(p.read_bytes()).hexdigest()
        )
        for p in files
    ]
    identity = hashlib.sha256(
        json.dumps(
            [
                CONTRACT,
                current,
                global_findings_contract,
                *(
                    [
                        global_scope,
                        hashlib.sha256(
                            Path(__file__)
                            .with_name(scope_modules[global_scope])
                            .read_bytes()
                        ).hexdigest(),
                    ]
                    if global_scope
                    else []
                ),
                *([global_input_contract] if global_input_contract is not None else []),
                *([DIRECT_INVENTORY_PROMPT] if direct_inventory else []),
                *([STAGE_BATCH_LIMITS] if stage_batch_limits else []),
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
                *(
                    [
                        citation_contract,
                        STATUS_CONTRACT,
                        *(
                            [capacity_recovery_contract]
                            if capacity_recovery_contract
                            else []
                        ),
                        *([organization_contract] if organization_contract else []),
                        inline_citations.identity(),
                        *file_hashes,
                    ]
                    if citation_contract
                    else (
                        *PRE_INLINE_FILE_HASHES,
                        *(
                            [
                                organization_contract,
                                *file_hashes,
                            ]
                            if organization_contract
                            else []
                        ),
                    )
                ),
            ]
        ).encode()
    ).hexdigest()
    return {
        "contract": CONTRACT,
        "ragas_version": current,
        "identity": identity,
        "batch_size": batch_size,
        **({"global_scope": global_scope} if global_scope else {}),
        **(
            {"inventory_prompt_contract": DIRECT_INVENTORY_PROMPT}
            if direct_inventory
            else {}
        ),
        **({"stage_batch_contract": STAGE_BATCH_LIMITS} if stage_batch_limits else {}),
        **({"citation_contract": citation_contract} if citation_contract else {}),
        **({"global_status_contract": STATUS_CONTRACT} if citation_contract else {}),
        **(
            {"organization_contract": organization_contract}
            if organization_contract
            else {}
        ),
        **(
            {"capacity_recovery_contract": capacity_recovery_contract}
            if capacity_recovery_contract
            else {}
        ),
        **(
            {"inventory_context_contract": INVENTORY_CONTEXT_CONTRACT}
            if capacity_recovery_contract
            else {}
        ),
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


ARCHIVED_INLINE_PATCH_ENGINE = {
    "contract": "ragas-cited-faithfulness-v1",
    "ragas_version": "0.4.3",
    "identity": "9db5fb786c51a9407aa8baf7d572b74aede3e990dc513a1bfc9f85fe30020477",
    "batch_size": 20,
    "citation_contract": "flat-writer-v2",
    "global_status_contract": "ragas-global-status-codes-v1",
    "capacity_recovery_contract": "native-capacity-recovery-v1",
    "inventory_context_contract": "ragas-inventory-local-context-v1",
    "assessment_response_contract": "ragas-assessment-statement-keys-v1",
    "global_findings_contract": "anchored-global-findings-v2",
    "global_unit_response_contract": "ragas-global-unit-keys-v4",
}


ARCHIVED_INLINE_STAGE_ENGINE = {
    **ARCHIVED_INLINE_PATCH_ENGINE,
    "identity": "fb4d208a99c5085e1e548bb3330b4f94e695f8a48b27fcee1e3dd983df366cdc",
}


ARCHIVED_INLINE_PRE_ORGANIZATION_ENGINE = {
    **ARCHIVED_INLINE_PATCH_ENGINE,
    "identity": "553f399b44f8c4c48faeae3d61335c38d175109ece20b113792987432257ae4f",
}

ARCHIVED_INLINE_BOUND_ORGANIZATION_ENGINE = {
    **ARCHIVED_INLINE_PATCH_ENGINE,
    "identity": "c90fc3501dafbc377369d55941d25d48597b8adbf8e09edeffbd21a80a74e849",
    "organization_contract": BOUND_ORGANIZATION,
}

ARCHIVED_PRO_INVENTORY_ENGINE = {
    **ARCHIVED_INLINE_BOUND_ORGANIZATION_ENGINE,
    "identity": "6d3a1e69514b762d59e710198ca7a185dbdeca58692f0f90d948b96e3e016173",
}

# Source-hashed organization profiles from before the ordered global transport.
ARCHIVED_ORGANIZATION_ENGINES = tuple(
    {
        "contract": CONTRACT,
        "ragas_version": "0.4.3",
        "batch_size": 20,
        "identity": identity,
        "organization_contract": BOUND_ORGANIZATION,
        "assessment_response_contract": KEYED_ASSESSMENT,
        "global_findings_contract": ANCHORED_GLOBAL_REVIEW,
        "global_unit_response_contract": response,
    }
    for response, identity in (
        (
            KEYED_GLOBAL_QUOTED_OMISSIONS,
            "221f44a565bb91c92b34ee27ed07f7fd9bbe721d2095e64bd9aa4bd1eb610af4",
        ),
        (
            KEYED_GLOBAL_LOCATED_OMISSIONS,
            "88a48a9fee925f616cdd28a53d730173d176d3bc4809bf5e85191c91def6069f",
        ),
        (
            KEYED_GLOBAL_QUOTED_OMISSIONS,
            "ae5e99b2622a6680eaf982f9e1027f3b225b073d907108d6481b43c5d7da0c18",
        ),
        (
            KEYED_GLOBAL_LOCATED_OMISSIONS,
            "800f4c7bdb08d8312335ee8d35186288c2f26d47728124db66435e8b82cfdb3b",
        ),
    )
)


def validate_engine(value, *, archived=False):
    """Reject changed semantics rather than silently restoring an old checkpoint."""
    if not isinstance(value, dict):
        raise ValueError("ragas_engine_identity_invalid")
    if archived and value in (
        {
            **ARCHIVED_ORGANIZATION_ENGINES[0],
            "global_scope": "question-coverage-v1",
            "identity": "5292f369cdc647277a0825af93ab54e9071aeef1336aa5e74cefb458a1d1c830",
        },
        {
            **ARCHIVED_ORGANIZATION_ENGINES[0],
            "identity": "7f45165503b1bccd89164b65400999f885c218dc39da482f7d8aee682d470098",
        },
        {
            **ARCHIVED_PRO_INVENTORY_ENGINE,
            "global_input_contract": ORDERED_UNIT_CONTRACT,
            "global_scope": "question-coverage-v1",
            "identity": "de72e77aa6825f762b171da20c0fe5cfd68aeee18ea5cee8bd8b63d9897f86f9",
        },
        {
            **ARCHIVED_PRO_INVENTORY_ENGINE,
            "global_input_contract": ORDERED_UNIT_CONTRACT,
            "identity": "477e1184960bc41a7d260ec1f64bfc3413f223e75708774772f91e390f7bd420",
        },
        {
            **ARCHIVED_PRO_INVENTORY_ENGINE,
            "global_input_contract": ORDERED_UNIT_CONTRACT,
            "identity": "50054c4c2718e23900836d2ba51219b1ecaf080d3a79b4a4b8f3a44a8872d144",
        },
        ARCHIVED_PRO_INVENTORY_ENGINE,
        *ARCHIVED_ORGANIZATION_ENGINES,
        ARCHIVED_INLINE_PATCH_ENGINE,
        ARCHIVED_INLINE_STAGE_ENGINE,
        ARCHIVED_INLINE_PRE_ORGANIZATION_ENGINE,
        ARCHIVED_INLINE_BOUND_ORGANIZATION_ENGINE,
        ARCHIVED_INLINE_VIT_ENGINE,
        ARCHIVED_INLINE_UNCITED_ENGINE,
        ARCHIVED_INLINE_STATUS_ENGINE,
        *ARCHIVED_INLINE_EDITORIAL_ENGINES,
        ARCHIVED_INLINE_SUMMARY_ENGINE,
        ARCHIVED_INLINE_DEPENDENCY_ENGINE,
        ARCHIVED_INLINE_RETRY_ENGINE,
        ARCHIVED_INLINE_CANDIDATE_ENGINE,
    ):
        return dict(value)
    options = dict(
        batch_size=value.get("batch_size"),
        global_scope=value.get("global_scope"),
        global_findings_contract=value.get("global_findings_contract"),
        global_input_contract=value.get("global_input_contract"),
        global_unit_response_contract=value.get("global_unit_response_contract"),
        assessment_response_contract=value.get("assessment_response_contract"),
        citation_contract=value.get("citation_contract"),
        capacity_recovery_contract=value.get("capacity_recovery_contract"),
        organization_contract=value.get("organization_contract"),
    )
    current = engine_descriptor(**options)
    if (
        archived
        and "inventory_prompt_contract" in current
        and "inventory_prompt_contract" not in value
        and value == engine_descriptor(**options, _previous_inventory_prompt=True)
    ):
        return dict(value)
    if (
        archived
        and "stage_batch_contract" in current
        and "stage_batch_contract" not in value
        and value == engine_descriptor(**options, _previous_stage_batch_policy=True)
    ):
        return dict(value)
    historical = (
        ARCHIVED_PREFIXED_ENGINE_IDENTITY
        if current.get("global_findings_contract") == ANCHORED_GLOBAL_REVIEW
        else ARCHIVED_ENGINE_IDENTITY
    )
    archived_ids = []
    if (
        archived
        and "global_scope" not in current
        and "citation_contract" not in current
        and "organization_contract" not in current
    ):
        findings = current.get("global_findings_contract")
        input_contract = current.get("global_input_contract")
        units_contract = current.get("global_unit_response_contract")
        wire = findings, input_contract, units_contract
        keyed_assessment = "assessment_response_contract" in current
        if units_contract == KEYED_GLOBAL_QUOTED_OMISSIONS:
            archived_ids.append(
                ARCHIVED_QUOTED_OMISSION_ENGINES[
                    input_contract, current.get("assessment_response_contract")
                ]
            )
        elif units_contract == KEYED_GLOBAL_ANCHORS:
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
        if getattr(evaluation, "native_global_scope", None):
            raise ValueError("ragas_global_scope_invalid")
        if getattr(evaluation, "native_capacity_recovery", False):
            raise ValueError("native_capacity_contract_invalid")
        if input_contract is not None:
            raise ValueError("ragas_global_input_contract_invalid")
        if unit_response is not None:
            raise ValueError("ragas_global_unit_response_contract_invalid")
        return None
    if selected != CONTRACT:
        raise ValueError("ragas_engine_invalid")
    from litagent.run_limits import FORMAT_JSON_POLICY, configured_run_limits

    limits = configured_run_limits(config) or {}
    recovery = getattr(config.eval, "native_capacity_recovery", False)
    if recovery and (
        limits.get("citation_contract") != "flat-writer-v2"
        or limits.get("contract") != "configured-run-limits-v6"
    ):
        raise ValueError("native_capacity_contract_invalid")
    return engine_descriptor(
        config.eval.strict_batch_size,
        global_scope=getattr(evaluation, "native_global_scope", None),
        global_findings_contract=(
            ANCHORED_GLOBAL_REVIEW
            if limits.get("syntax_policy") == FORMAT_JSON_POLICY
            else None
        ),
        global_input_contract=input_contract,
        global_unit_response_contract=unit_response,
        assessment_response_contract=KEYED_ASSESSMENT,
        citation_contract=limits.get("citation_contract"),
        capacity_recovery_contract=(
            "native-capacity-recovery-v1" if recovery else None
        ),
        organization_contract=(
            BOUND_ORGANIZATION
            if unit_response
            in (KEYED_GLOBAL_QUOTED_OMISSIONS, KEYED_GLOBAL_LOCATED_OMISSIONS)
            else None
        ),
    )


def anchored_native_review(envelope):
    """Select the new report-review wire only from the frozen engine descriptor."""
    engine = (envelope or {}).get("semantic_engine", {})
    return engine.get("global_findings_contract") == ANCHORED_GLOBAL_REVIEW


def unit_native_review(envelope):
    """Select the writing guidance originally paired with the v2 unit view."""
    engine = (envelope or {}).get("semantic_engine", {})
    return engine.get("global_input_contract") == UNIT_CONTRACT


def native_global_input_contract(envelope):
    """Read the request representation from the frozen evaluator identity."""
    return ((envelope or {}).get("semantic_engine") or {}).get("global_input_contract")


def native_host_uncited(envelope):
    """Determine no-source verdicts locally under the frozen evidence policy."""
    from litagent.inline_citations import selected

    value = envelope or {}
    return selected(value) or (
        (value.get("semantic_engine") or {}).get("organization_contract")
        == BOUND_ORGANIZATION
    )


def native_unit_response_contract(envelope):
    """Select the frozen global unit and finding response protocol."""
    engine = (envelope or {}).get("semantic_engine", {})
    return engine.get("global_unit_response_contract")


def phase_batches(envelope):
    """Reserve complete per-unit generator and NLI rounds including summaries."""
    validate_engine(envelope["semantic_engine"], archived=True)
    return reserved_phase_batches(envelope)


def reserved_phase_batches(envelope):
    """Compute batch slots from an already validated or projected envelope."""
    inline = (envelope.get("run_limits") or {}).get("citation_contract")
    if envelope["semantic_engine"].get("citation_contract") != inline:
        raise ValueError("ragas_citation_contract_mismatch")
    units = envelope["body_max_units"] + 2 * envelope["summary_max_units_per_section"]
    size = envelope["semantic_engine"]["batch_size"]
    if (envelope.get("capacity") or {}).get("batch_size", 16) != size:
        raise ValueError("ragas_capacity_batch_mismatch")
    batches = (units + size - 1) // size
    return batches + 1 if inline else batches


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
