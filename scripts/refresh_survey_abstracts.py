"""Refresh benchmark abstracts from the official arXiv Atom feed."""

from __future__ import annotations

import argparse
import hashlib
import re
import xml.etree.ElementTree as ET
from datetime import date
from difflib import SequenceMatcher
from pathlib import Path

import requests
import yaml

ATOM = "{http://www.w3.org/2005/Atom}"
MANIFESTS = (
    Path("benchmarks/survey/vit_nerf_manifest.yaml"),
    Path("benchmarks/survey/survey_additions_manifest.yaml"),
)


def refresh(*, write: bool = False) -> dict[str, object]:
    """Verify every source identity before updating local benchmark metadata."""
    manifests = {
        path: yaml.safe_load(path.read_text(encoding="utf-8")) for path in MANIFESTS
    }
    papers = [paper for manifest in manifests.values() for paper in manifest["papers"]]
    identifiers = [str(paper["arxiv_id"]) for paper in papers]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("duplicate arXiv id in survey manifests")
    response = requests.get(
        "https://export.arxiv.org/api/query",
        params={"id_list": ",".join(identifiers), "max_results": len(identifiers)},
        timeout=60,
    )
    response.raise_for_status()
    root = ET.fromstring(response.content)
    fetched: dict[str, tuple[str, str]] = {}
    for entry in root.findall(f"{ATOM}entry"):
        raw_id = entry.findtext(f"{ATOM}id", default="").rsplit("/", 1)[-1]
        paper_id = re.sub(r"v\d+$", "", raw_id)
        title = " ".join(entry.findtext(f"{ATOM}title", default="").split())
        abstract = " ".join(entry.findtext(f"{ATOM}summary", default="").split())
        if paper_id in fetched or not title or not abstract:
            raise ValueError(f"invalid or duplicate arXiv entry: {paper_id}")
        fetched[paper_id] = (title, abstract)
    if set(fetched) != set(identifiers):
        raise ValueError(
            f"arXiv response does not cover corpus: {sorted(set(identifiers) - set(fetched))}"
        )

    mismatched: list[str] = []
    for paper in papers:
        arxiv_id = str(paper["arxiv_id"])
        title, abstract = fetched[arxiv_id]
        ratio = SequenceMatcher(None, paper["title"].lower(), title.lower()).ratio()
        if ratio < 0.55:
            mismatched.append(f"{arxiv_id}: {paper['title']} != {title}")
            continue
        if paper.get("abstract_source") != "arxiv_api":
            paper["abstract_summary"] = paper.get("abstract", "")
        paper["abstract"] = abstract
        paper["abstract_source"] = "arxiv_api"
        paper["abstract_source_version"] = "arxiv_atom_v1"
        paper["abstract_url"] = f"https://arxiv.org/abs/{arxiv_id}"
        paper["abstract_accessed_at"] = date.today().isoformat()
        paper["abstract_sha256"] = hashlib.sha256(abstract.encode("utf-8")).hexdigest()
    if mismatched:
        raise ValueError(
            "arXiv title mismatch; no manifests written: " + "; ".join(mismatched)
        )
    if write:
        for path, manifest in manifests.items():
            path.write_text(
                yaml.safe_dump(
                    manifest, allow_unicode=True, sort_keys=False, width=100
                ),
                encoding="utf-8",
            )
    return {
        "papers": len(papers),
        "manifests": [str(path) for path in MANIFESTS],
        "written": write,
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--write", action="store_true", help="replace local manifest abstracts"
    )
    print(refresh(write=parser.parse_args().write))
