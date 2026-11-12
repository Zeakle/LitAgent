"""Keep schema semantics and real response diagnostics across local-ref inlining."""

import copy
import hashlib
import json
from pathlib import Path

import pytest
from jsonschema import Draft202012Validator
from test_native_array_integration import request
from test_writing_passages import facts

from litagent.body_quality import NATIVE_SCHEMA_ENCODING_CONTRACT
from litagent.demo import project_flow_demo_view
from litagent.native_wire import inline_local_schema
from litagent.observability.recorder import ArchiveRepository, RunRecorder
from litagent.quality import project_grounded_response
from litagent.workbench import RunOptions

FIXTURE = json.loads(
    Path("tests/fixtures/native-array-production-failure-real.json").read_text()
)


def protocol_document(inline):
    """Build a non-scientific shape fixture from actual frozen permissions."""
    doc = {"title": "PROTOCOL_PROBE", "sections": {}}
    for key, section in inline["properties"]["sections"]["properties"].items():
        if key in {"abstract", "conclusion"}:
            unit = {
                "text": "PROTOCOL_PROBE",
                "body_references": [
                    {"section_key": "methods", "block_index": 0, "sentence_index": 0}
                ],
            }
        else:
            sentence = section["properties"]["blocks"]["items"]["properties"][
                "sentences"
            ]["items"]
            unit = {
                "text": "PROTOCOL_PROBE",
                "kind": "scientific",
                "fact_card_ids": sentence["properties"]["fact_card_ids"]["items"][
                    "enum"
                ][:1],
                "scope_notice_id": "none",
            }
        doc["sections"][key] = {
            "blocks": [{"heading": "", "type": "paragraph", "sentences": [unit]}]
        }
    return doc


@pytest.mark.parametrize(
    "fault",
    [
        None,
        "missing_section",
        "foreign_card",
        "wrong_type",
        "extra_field",
        "wrong_kind",
        "wrong_ref_type",
        "empty_blocks",
        "wrong_number",
    ],
)
def test_received_schema_acceptance_and_rejection_sets_are_preserved(fault):
    before = copy.deepcopy(FIXTURE["schema"])
    new = inline_local_schema(before)
    doc = protocol_document(new)
    methods = doc["sections"]["methods"]["blocks"][0]["sentences"][0]
    if fault == "missing_section":
        del doc["sections"]["experiments"]
    elif fault == "foreign_card":
        methods["fact_card_ids"] = ["not-permitted"]
    elif fault == "wrong_type":
        methods["text"] = 33
    elif fault == "extra_field":
        methods["invented"] = "not silently removed"
    elif fault == "wrong_kind":
        methods["kind"] = "verified"
    elif fault == "wrong_ref_type":
        doc["sections"]["abstract"]["blocks"][0]["sentences"][0]["body_references"][0][
            "sentence_index"
        ] = True
    elif fault == "empty_blocks":
        doc["sections"]["methods"]["blocks"] = []
    elif fault == "wrong_number":
        methods["text"] = "EViT gains 90% on DeiT-S."
    old_valid = Draft202012Validator(before).is_valid(doc)
    assert Draft202012Validator(new).is_valid(doc) == old_valid
    assert old_valid is (fault in {None, "empty_blocks", "wrong_number"})
    assert before == FIXTURE["schema"] and "$defs" not in new
    # Schema validity, including a wrong scientific number, is never support.
    assert '"$ref"' not in json.dumps(new)


def test_literal_refs_property_names_and_shared_definitions_are_not_corrupted():
    schema = {
        "type": "object",
        "$defs": {"a/b": {"type": "object", "const": {"$ref": "literal"}}},
        "properties": {
            "$ref": {"$ref": "#/$defs/a~1b"},
            "other": {"$ref": "#/$defs/a~1b"},
        },
        "required": ["$ref", "other"],
        "additionalProperties": False,
    }
    original = copy.deepcopy(schema)
    result = inline_local_schema(schema)
    value = {"$ref": {"$ref": "literal"}, "other": {"$ref": "literal"}}
    assert Draft202012Validator(schema).is_valid(value) and Draft202012Validator(
        result
    ).is_valid(value)
    result["properties"]["other"]["const"]["$ref"] = "changed"
    assert result["properties"]["$ref"]["const"]["$ref"] == "literal"
    assert schema == original


@pytest.mark.parametrize(
    "fault", ["external", "missing", "cycle", "siblings", "nested_defs", "deep"]
)
def test_unsupported_reference_cases_fail_without_dropping_constraints(fault):
    schema = {
        "type": "object",
        "$defs": {"a": {"type": "string"}},
        "properties": {"x": {"$ref": "#/$defs/a"}},
    }
    if fault == "external":
        schema["properties"]["x"]["$ref"] = "https://example.invalid/schema.json"
    elif fault == "missing":
        schema["properties"]["x"]["$ref"] = "#/$defs/no-such-definition"
    elif fault == "cycle":
        schema["$defs"]["a"] = {"$ref": "#/$defs/a"}
    elif fault == "siblings":
        schema["properties"]["x"]["minLength"] = 2
    elif fault == "nested_defs":
        schema["$defs"]["a"]["$defs"] = {"nested": {"type": "string"}}
    else:
        node = schema["$defs"]["a"]
        for _ in range(65):
            node["properties"] = {"next": {"type": "object"}}
            node = node["properties"]["next"]
    before = copy.deepcopy(schema)
    with pytest.raises(ValueError, match="native_schema_inline"):
        inline_local_schema(schema)
    assert schema == before


@pytest.mark.parametrize("language", ["en", "zh-CN"])
def test_only_explicit_new_analysis_wire_is_inlined(language):
    messages, response_format = request(facts(), language, array_wire=True)
    schema = response_format["json_schema"]["schema"]
    assert "$defs" not in schema and '"$ref"' not in json.dumps(schema)
    assert (
        json.loads(messages[1]["content"])["native_schema_encoding_contract"]
        == NATIVE_SCHEMA_ENCODING_CONTRACT
    )
    old_messages, old_format = request(facts(), language)
    assert "$defs" in old_format["json_schema"]["schema"]
    assert "native_schema_encoding_contract" not in json.loads(
        old_messages[1]["content"]
    )


def test_inline_encoding_changes_analysis_identity_without_changing_legacy(monkeypatch):
    import litagent.body_quality as body

    old = RunOptions()
    current = RunOptions(
        quality_policy="strict_auto_v2",
        review_structure="academic_v2",
        body_quality="analysis_v1",
    )
    a, b = old.fingerprint(), current.fingerprint()
    monkeypatch.setattr(
        body, "NATIVE_SCHEMA_ENCODING_CONTRACT", "inline-local-schema-future"
    )
    assert old.fingerprint() == a and current.fingerprint() != b


def test_real_json_failure_is_not_completed_or_given_an_evaluation_score(tmp_path):
    raw = FIXTURE["raw_response"]
    assert hashlib.sha256(raw.encode()).hexdigest() == FIXTURE["raw_sha256"]
    rf = {
        "type": "json_schema",
        "json_schema": {
            "name": "grounded_report_v23_arrays",
            "schema": inline_local_schema(FIXTURE["schema"]),
            "strict": True,
        },
    }
    events = []
    with pytest.raises(ValueError, match="array_wire_contract_invalid") as failed:
        project_grounded_response(
            raw,
            rf,
            trace_hook=lambda event, data: events.append(
                {"event": event, "data": data}
            ),
            parent_task_id="write",
        )
    assert isinstance(failed.value.__cause__, json.JSONDecodeError)
    error = events[-1]["data"]
    assert error["json_position"] == 8299 and error["json_column"] == 8300
    assert (
        error["subreason"] == "json_decode_error"
        and error["error_type"] == "JSONDecodeError"
    )
    recorder = RunRecorder(
        "inline-failure", "public shape test", ArchiveRepository(tmp_path)
    )
    recorder(
        "worker.start",
        {"task_id": "write", "agent_type": "synthesis", "description": "write"},
    )
    recorder(events[-1]["event"], error)
    artifact = recorder.snapshot()
    assert (
        artifact["nodes"]["worker:write"]["metadata"]["writing_wire_failures"][0]
        == error
    )
    artifact["report"] = {
        "survey": "placeholder",
        "evaluation": {
            "faithfulness": {
                "passed": False,
                "skipped": True,
                "details": {"skipped_reason": "final_report_unavailable"},
            }
        },
    }
    before = copy.deepcopy(artifact)
    view = project_flow_demo_view(artifact, source="archive")
    pipeline = view["pipeline_verification"]
    assert pipeline["stages"]["evaluation"] == "not_run"
    diagnostic = pipeline["writing_diagnostics"][0]
    assert diagnostic["location"] == "native_response"
    assert diagnostic["candidate_sha256"] == FIXTURE["raw_sha256"]
    assert "character=8299" in diagnostic["raw_json_path"]
    assert "column=8300" in diagnostic["raw_json_path"]
    assert diagnostic["candidate_text"] == "" and artifact == before
    assert view["evaluation"] == artifact["report"]["evaluation"]


@pytest.mark.parametrize(
    "keyword", ["$id", "$anchor", "$dynamicAnchor", "$dynamicRef", "$recursiveRef"]
)
def test_scoped_resolution_is_rejected_rather_than_incorrectly_flattened(keyword):
    schema = {
        "type": "object",
        "$defs": {
            "a": {
                keyword: (
                    "https://scope.invalid/root"
                    if keyword == "$id"
                    else (
                        "different"
                        if keyword in {"$anchor", "$dynamicAnchor"}
                        else "#different"
                    )
                ),
                "type": "string",
            }
        },
        "properties": {"x": {"$ref": "#/$defs/a"}},
    }
    with pytest.raises(ValueError, match="native_schema_inline_reference_invalid"):
        inline_local_schema(schema)


@pytest.mark.parametrize("locale", ["en", "zh-CN"])
def test_real_wire_event_reaches_existing_quality_dom_without_inventing_scores(locale):
    import shutil
    import subprocess

    event = {
        "event": "writing.wire.failed",
        "data": {
            "phase": "initial",
            "candidate_sha256": FIXTURE["raw_sha256"],
            "reason_code": "array_wire_contract_invalid",
            "json_position": 8299,
            "json_line": 1,
            "json_column": 8300,
        },
    }
    report = {
        "survey": "placeholder",
        "evaluation": {
            "faithfulness": {
                "score": 0,
                "passed": False,
                "skipped": True,
                "details": {"skipped_reason": "final_report_unavailable"},
            }
        },
    }
    artifact = {"run_id": "response-failure", "events": [event], "report": report}
    before = copy.deepcopy(artifact)
    view = project_flow_demo_view(artifact, source="archive")
    script = r"""
const fs=require('node:fs');
(async()=>{
class E{constructor(){this.children=[];this.dataset={};this.textContent='';}append(...x){this.children.push(...x);}replaceChildren(...x){this.children=x;}addEventListener(){}setAttribute(){}}
global.document={createElement:()=>new E()};
const {renderQuality}=await import('data:text/javascript;base64,'+fs.readFileSync(process.argv[1]+'/quality-view.js').toString('base64'));
const view=JSON.parse(process.argv[2]),root=new E(),walk=e=>[e,...e.children.flatMap(walk)],en=process.argv[3]==='en';
renderQuality(root,view.evaluation,view.quality,view.delivery,()=>{},x=>x,{locale:process.argv[3],pipelineVerification:view.pipeline_verification});
const text=walk(root).map(e=>e.textContent).join(' '),issue=view.pipeline_verification.writing_diagnostics[0];
for(const value of ['native_response','character=8299','column=8300',issue.candidate_sha256,'array_wire_contract_invalid',en?'not Evaluation scores':'不是 Evaluation 评分'])if(!text.includes(value))throw Error(value);
if(text.includes('0.000')||text.includes(en?'Historical evaluation scope':'历史评估口径'))throw Error('invented evaluation');
})().catch(e=>{console.error(e);process.exit(1)});
"""
    result = subprocess.run(
        [
            shutil.which("node") or "node",
            "-e",
            script,
            str(Path("src/litagent/static/flow-demo").resolve()),
            json.dumps(view),
            locale,
        ],
        capture_output=True,
        text=True,
        timeout=15,
    )
    assert result.returncode == 0, result.stderr
    assert artifact == before


def test_missing_report_and_missing_positions_keep_response_diagnostics_read_only():
    event = {
        "event": "writing.wire.failed",
        "data": {
            "reason_code": "array_wire_contract_invalid",
            "raw_response_sha256": FIXTURE["raw_sha256"],
        },
    }
    artifact = {"run_id": "no-report", "events": [event]}
    before = copy.deepcopy(artifact)
    result = project_flow_demo_view(artifact, source="archive")
    issue = result["pipeline_verification"]["writing_diagnostics"][0]
    assert issue["raw_json_path"] == "native_response"
    assert issue["candidate_sha256"] == FIXTURE["raw_sha256"]
    assert result["evaluation"] == {} and artifact == before
