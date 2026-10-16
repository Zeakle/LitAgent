"""Exercise root trace lifecycle through the real tracer with a local SDK double."""

import asyncio
import json
from unittest.mock import Mock

import pytest
from test_frozen_quality_repair import factories, source

from litagent.config import load_config
from litagent.observability.tracing import LangFuseTracer
from scripts.run_frozen_quality_repair import prepare, run_trial


class Observation:
    """Capture only the real tracer's actual SDK operations."""

    def __init__(self):
        self.updates = []
        self.ended = False

    def update_trace(self, **kwargs):
        self.updates.append(kwargs)

    def update(self, **kwargs):
        self.updates.append(kwargs)

    def end(self):
        self.ended = True


class SDK:
    """Require survey.start to create a root and resolve its trace identity."""

    def __init__(self):
        self.starts = []
        self.root = Observation()
        self.flush = Mock()

    def create_trace_id(self, **kwargs):
        return "local-trace"

    def start_observation(self, **kwargs):
        self.starts.append(kwargs)
        return self.root

    def get_trace_url(self, **kwargs):
        return "http://localhost:3000/local-trace"


@pytest.mark.asyncio
@pytest.mark.parametrize("outcome", ["blocked", "completed", "failed", "cancelled"])
async def test_real_trace_root_starts_closes_and_links_without_scientific_payload(
    tmp_path, outcome
):
    src = source(tmp_path)
    cfg = load_config()
    crash = RuntimeError("private source body") if outcome == "failed" else None
    if outcome == "cancelled":
        crash = asyncio.CancelledError()
    deps, _ = factories(committed=outcome == "completed", crash=crash)
    sdk = SDK()
    tracer = LangFuseTracer("", "", "", trace_seed="test")
    tracer._client = sdk
    deps["trace_factory"] = lambda _: tracer
    root = tmp_path / "trial"
    kwargs = dict(
        root=root,
        run_id="trace-trial",
        live=True,
        authorization={
            "actor": "user",
            "binding": prepare(src, cfg, "trace-trial")["plan"]["binding"],
        },
        **deps,
    )
    if outcome == "cancelled":
        with pytest.raises(asyncio.CancelledError):
            await run_trial(src, cfg, **kwargs)
    else:
        await run_trial(src, cfg, **kwargs)
    assert sdk.starts and sdk.root.ended and sdk.flush.called
    assert sdk.starts[0]["input"] == {"query": ""}
    assert "private source body" not in json.dumps(sdk.root.updates)
    if outcome in {"completed", "blocked"}:
        output = next(x["output"] for x in sdk.root.updates if "output" in x)
        assert output["accepted"] is (outcome == "completed")
    artifact = json.loads((root / "runs/trace-trial.json").read_text())
    assert artifact["trace"]["trace_id"] == "local-trace"
    assert artifact["events"][0]["event"] == "survey.start"
    assert artifact["events"][-1]["event"] == (
        "survey.error" if crash else "survey.complete"
    )
