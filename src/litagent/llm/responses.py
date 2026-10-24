"""Map a fixed native schema to Responses without changing its business contract."""

from __future__ import annotations

import base64
import copy
import hashlib
import json
import math

from litagent.safety.budget import strict_request_reservation

RESPONSES_CONTRACT = "ark-responses-mapping-v1"


def build_responses_request(
    messages,
    *,
    model,
    output_limit,
    response_format,
    temperature,
    extra_body=None,
    reasoning_effort=None,
):
    """Validate the complete mapping before any reservation or HTTP attempt."""
    if (
        not isinstance(messages, list)
        or not messages
        or any(
            not isinstance(m, dict)
            or set(m) != {"role", "content"}
            or m["role"] not in ("system", "user")
            or not isinstance(m["content"], str)
            or not m["content"]
            for m in messages
        )
    ):
        raise ValueError("responses_input_invalid")
    if not isinstance(model, str) or not model:
        raise ValueError("responses_model_invalid")
    if type(output_limit) is not int or output_limit <= 0:
        raise ValueError("output_limit_invalid")
    if (
        not isinstance(response_format, dict)
        or set(response_format) != {"type", "json_schema"}
        or response_format["type"] != "json_schema"
    ):
        raise ValueError("responses_schema_invalid")
    schema = response_format["json_schema"]
    if (
        not isinstance(schema, dict)
        or set(schema) != {"name", "schema", "strict"}
        or schema["strict"] is not True
        or not isinstance(schema["schema"], dict)
        or not isinstance(schema["name"], str)
        or not schema["name"]
    ):
        raise ValueError("responses_schema_invalid")
    if type(temperature) not in (int, float) or not math.isfinite(temperature):
        raise ValueError("responses_temperature_invalid")
    if extra_body is not None and extra_body not in (
        {"thinking": {"type": "enabled"}},
        {"thinking": {"type": "disabled"}},
    ):
        raise ValueError("responses_extra_body_invalid")
    if reasoning_effort not in (None, "low") or (
        reasoning_effort is not None and extra_body != {"thinking": {"type": "enabled"}}
    ):
        raise ValueError("responses_reasoning_invalid")
    body = {
        "model": model,
        "input": copy.deepcopy(messages),
        "text": {"format": {"type": "json_schema", **copy.deepcopy(schema)}},
        "max_output_tokens": output_limit,
        "temperature": temperature,
        "store": False,
        "background": False,
        "truncation": "disabled",
        "stream": False,
        **(copy.deepcopy(extra_body) if extra_body else {}),
        **({"reasoning": {"effort": reasoning_effort}} if reasoning_effort else {}),
    }
    # Keep the same canonical reservation shared by caller-owned repairs and SDK.
    # Refuse an oversized envelope instead of silently borrowing downstream funds.
    encoded = json.dumps(body, ensure_ascii=False, allow_nan=False).encode("utf-8")
    if len(encoded) + output_limit > strict_request_reservation(
        messages, output_limit, response_format
    ):
        raise ValueError("responses_envelope_budget_exceeded")
    return body


def _object(pairs):
    """Reject duplicate envelope keys instead of silently selecting the last value."""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError("responses_duplicate_key")
        result[key] = value
    return result


def _invalid_constant(value):
    """Reject non-JSON numeric constants without normalizing them into null."""
    raise ValueError("responses_numeric_constant_invalid")


def normalize_responses(raw: bytes) -> dict:
    """Keep billing separate from output validity, using uncoerced wire types."""
    result = {
        "content": "",
        "model": "",
        "finish_reason": "invalid",
        "provider_error": "responses_envelope_invalid",
        "provider_status": None,
        "usage": {"prompt_tokens": 0, "completion_tokens": 0, "usage_available": False},
    }
    try:
        body = json.loads(
            raw, object_pairs_hook=_object, parse_constant=_invalid_constant
        )
    except (ValueError, UnicodeDecodeError):
        return result
    if not isinstance(body, dict):
        return result
    usage = body.get("usage")
    if (
        isinstance(usage, dict)
        and all(
            type(usage.get(k)) is int and usage[k] >= 0
            for k in ("input_tokens", "output_tokens", "total_tokens")
        )
        and usage["input_tokens"] + usage["output_tokens"] == usage["total_tokens"]
    ):
        result["usage"] = {
            "prompt_tokens": usage["input_tokens"],
            "completion_tokens": usage["output_tokens"],
            "usage_available": True,
        }
        detail = usage.get("output_tokens_details")
        if isinstance(detail, dict) and detail.get("reasoning_tokens") is not None:
            value = detail["reasoning_tokens"]
            valid = type(value) is int and 0 <= value <= usage["output_tokens"]
            result["usage"]["reasoning_usage_available"] = valid
            if valid:
                result["usage"]["reasoning_tokens"] = value
    model = body.get("model")
    if isinstance(model, str) and model:
        result["model"] = model
    status = body.get("status")
    if isinstance(status, str):
        result["provider_status"] = (
            status
            if status
            in {
                "completed",
                "incomplete",
                "failed",
                "cancelled",
                "queued",
                "in_progress",
            }
            else "unknown"
        )
    output = body.get("output")
    texts, invalid = [], False
    if not isinstance(output, list) or not output:
        invalid = True
    else:
        for item in output:
            if not isinstance(item, dict):
                invalid = True
                continue
            if item.get("type") == "reasoning":
                # Reasoning summaries are private receipt data, never evidence.
                if item.get("status") not in (None, "completed"):
                    invalid = True
                continue
            if (
                item.get("type") != "message"
                or item.get("role") != "assistant"
                or item.get("status") != "completed"
                or not isinstance(item.get("content"), list)
                or not item["content"]
            ):
                invalid = True
                continue
            for part in item["content"]:
                if (
                    not isinstance(part, dict)
                    or part.get("type") != "output_text"
                    or not isinstance(part.get("text"), str)
                    or not part["text"]
                ):
                    invalid = True
                else:
                    texts.append(part["text"])
    # Preserve all text in order, even on failure; never choose a convenient part.
    result["content"] = "".join(texts)
    if (
        body.get("object") != "response"
        or not isinstance(body.get("id"), str)
        or not body["id"]
    ):
        result["provider_error"] = "responses_envelope_invalid"
    elif any(
        key in body
        and (
            body[key] is not expected
            if type(expected) is bool
            else body[key] != expected
        )
        for key, expected in (
            ("store", False),
            ("background", False),
            ("truncation", "disabled"),
        )
    ):
        result["provider_error"] = "responses_settings_conflict"
    elif status == "incomplete":
        result["provider_error"] = "responses_incomplete"
        if (
            isinstance(body.get("incomplete_details"), dict)
            and body["incomplete_details"].get("reason") == "max_output_tokens"
        ):
            result["finish_reason"] = "length"
    elif status != "completed":
        result["provider_error"] = "responses_status_invalid"
    elif body.get("error") is not None or body.get("incomplete_details") is not None:
        result["provider_error"] = "responses_completion_conflict"
    elif invalid or not texts:
        result["provider_error"] = "responses_output_invalid"
    elif not result["model"]:
        result["provider_error"] = "responses_model_missing"
    else:
        result.update(finish_reason="stop", provider_error=None)
    return result


def private_receipt(response) -> dict:
    """Retain exact HTTP bytes locally, excluding credentials and cookie headers."""
    request = response.request
    sent, received = request.content, response.content
    return {
        "contract": RESPONSES_CONTRACT,
        "endpoint": request.url.path,
        "status_code": response.status_code,
        "request_body_base64": base64.b64encode(sent).decode("ascii"),
        "response_body_base64": base64.b64encode(received).decode("ascii"),
        "request_sha256": hashlib.sha256(sent).hexdigest(),
        "response_sha256": hashlib.sha256(received).hexdigest(),
        "headers": {
            name: response.headers[name]
            for name in (
                "x-request-id",
                "x-tt-logid",
                "date",
                "content-type",
            )
            if name in response.headers
        },
    }
