"""Check the published repair case and its exact archived lineage."""

import json
from collections import Counter
from html.parser import HTMLParser
from pathlib import Path
from shutil import copyfile

import pytest

from litagent.presentation import PUBLIC_ROOT
from scripts.publish_frozen_repair import JOURNAL, STEM, publication, render_html


class PageLinks(HTMLParser):
    """Collect document anchors without running JavaScript."""

    def __init__(self):
        """Track repeated IDs and actual link targets."""
        super().__init__()
        self.ids = Counter()
        self.links = []

    def handle_starttag(self, tag, attrs):
        """Remember each published ID and navigation link."""
        attrs = dict(attrs)
        if "id" in attrs:
            self.ids[attrs["id"]] += 1
        if tag == "a":
            self.links.append(attrs["href"])


def test_frozen_case_public_data_matches_html_and_links():
    """Keep the real single-case result distinct from aggregate measurements."""
    data = json.loads((PUBLIC_ROOT / f"{STEM}.json").read_bytes())
    page = (PUBLIC_ROOT / f"{STEM}.html").read_text(encoding="utf-8")
    assert render_html(data) == page
    assert data["formal_benchmark"] is False and data["human_accuracy"] is None
    before, after = data["before"], data["after"]
    assert (before["supported"], before["scientific"], before["unsupported"]) == (
        330,
        339,
        9,
    )
    assert before["faithfulness"] is None and not before["publishable"]
    assert after["supported"] == after["scientific"] == 398
    assert after["unsupported"] == after["unverified"] == after["unknown_hold"] == 0
    assert after["faithfulness"] == 1 and after["publishable"]
    assert (after["requests"], after["tokens"]) == (9, 188560)
    assert len(data["repairs"]) == 8
    assert sum(len(row["before"]["assertions"]) for row in data["repairs"]) == 9
    parser = PageLinks()
    parser.feed(page)
    assert max(parser.ids.values()) == 1
    assert {
        link[1:] for link in parser.links if link.startswith("#")
    } <= parser.ids.keys()
    assert "#before-claim_191" in parser.links
    assert "https://arxiv.org/abs/2105.00379" in parser.links
    for root in (PUBLIC_ROOT, PUBLIC_ROOT.parent / "flow-demo"):
        entry = (root / "index.html").read_text(encoding="utf-8")
        assert f'href="/static/presentation/{STEM}.html"' in entry


def test_frozen_case_regenerates_from_read_only_archives(tmp_path):
    """Use actual receipts and reject substituted report bytes or source cards."""
    archive = Path("artifacts/acceptance")
    before = archive / "native-repair-trial-20261110"
    after = archive / "native-repair-377-multi-20261110"
    if not all((root / JOURNAL).is_file() for root in (before, after)):
        pytest.skip("380 private archives are available only on the validation machine")
    expected = json.loads((PUBLIC_ROOT / f"{STEM}.json").read_bytes())
    assert publication(before, after) == expected
    for name in (
        "live-exit.json",
        "manifest.json",
        "ledger.json",
        "result.json",
        "rounds.json",
        "report.json",
        "publishable.md",
        JOURNAL,
    ):
        copyfile(after / name, tmp_path / name)
    report_path = tmp_path / "report.json"
    report = json.loads(report_path.read_bytes())
    card = next(iter(report["metadata"]["fact_cards"].values()))
    card["paper_id"] = "arxiv:9999.99999"
    report_path.write_text(json.dumps(report, ensure_ascii=False), encoding="utf-8")
    with pytest.raises(ValueError, match="repair_lineage_mismatch"):
        publication(before, tmp_path)
    copyfile(after / "report.json", report_path)
    (tmp_path / "publishable.md").write_text("substituted", encoding="utf-8")
    with pytest.raises(ValueError, match="repair_lineage_mismatch"):
        publication(before, tmp_path)
