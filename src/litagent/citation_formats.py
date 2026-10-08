"""Format immutable bibliographic snapshots with a pinned, local CSL processor."""

from __future__ import annotations

import hashlib
import json
import re
import shutil
import subprocess
from collections.abc import Mapping
from functools import lru_cache
from html.parser import HTMLParser
from pathlib import Path

STYLES = (
    "evidence",
    "numbered",
    "ieee",
    "apa7",
    "gbt7714_2025_numeric",
    "gbt7714_2025_author_date",
)
ACADEMIC_STYLES = STYLES[2:]
BIBLIOGRAPHY_FORMATS = ("bibtex", "ris", "csl_json")
ASSET_ROOT = Path(__file__).with_name("csl")
REF_PATTERN = re.compile(r"\[E:([^\[\]\s;,]+)\]")


def format_capabilities() -> dict:
    """Expose installed capabilities without invoking models or external services."""
    return {
        "citation_styles": list(STYLES),
        "academic_engine_available": shutil.which("node") is not None,
        "report_formats": ["md", "pdf", "docx"],
        "bibliography_formats": list(BIBLIOGRAPHY_FORMATS),
    }


def build_bibliographic_snapshot(report, ledger: Mapping, catalog: Mapping) -> dict:
    """Collect only cited papers, preserving supplied names and field provenance."""
    items = {}
    warnings = []
    for match in REF_PATTERN.finditer(report.survey):
        eid = match.group(1)
        evidence = ledger.get(eid) or {}
        pid = evidence.get("paper_id")
        if not isinstance(pid, str) or not pid:
            warnings.append(f"unresolved_evidence:{eid}")
            continue
        if pid in items:
            continue
        source = dict(catalog.get(pid) or {})
        item = {
            "id": pid,
            "type": "article",
            "title": str(source.get("title") or evidence.get("paper_title") or ""),
        }
        if not item["title"]:
            warnings.append(f"missing_title:{pid}")
        authors = source.get("author_details")
        if (
            isinstance(authors, list)
            and authors
            and all(
                isinstance(x, Mapping) and (x.get("family") or x.get("literal"))
                for x in authors
            )
        ):
            item["author"] = [
                {
                    k: str(v)
                    for k, v in x.items()
                    if k
                    in {
                        "family",
                        "given",
                        "literal",
                        "suffix",
                        "non-dropping-particle",
                        "dropping-particle",
                    }
                    and v
                }
                for x in authors
            ]
        else:
            raw = source.get("authors") or []
            names = (
                [x.strip() for x in raw if isinstance(x, str) and x.strip()]
                if isinstance(raw, list)
                else []
            )
            if names:
                item["author"] = [{"literal": x} for x in names]
                warnings.append(f"unparsed_author_names:{pid}")
            else:
                warnings.append(f"missing_authors:{pid}")
        year = source.get("year")
        if isinstance(year, int) and not isinstance(year, bool) and 1 <= year <= 9999:
            item["issued"] = {"date-parts": [[year]]}
        else:
            warnings.append(f"missing_year:{pid}")
        for key, target in {
            "doi": "DOI",
            "url": "URL",
            "journal": "container-title",
            "venue": "container-title",
            "volume": "volume",
            "issue": "issue",
            "pages": "page",
        }.items():
            if source.get(key) and target not in item:
                item[target] = str(source[key])
        if source.get("journal"):
            item["type"] = "article-journal"
        if pid.startswith("doi:") and "DOI" not in item:
            item["DOI"] = pid[4:]
        if pid.startswith("arxiv:"):
            item.setdefault("URL", "https://arxiv.org/abs/" + pid[6:])
            if item["type"] != "article-journal":
                item["genre"] = "Preprint"
            item["archive"] = "arXiv"
            item["archive_location"] = pid[6:]
        for field in ("container-title", "volume", "issue", "page"):
            if not item.get(field):
                warnings.append(f"missing_{field}:{pid}")
        items[pid] = item
    values = list(items.values())
    payload = {
        "report_sha256": report.content_sha256,
        "items": values,
        "warnings": list(dict.fromkeys(warnings)),
    }
    payload["snapshot_sha256"] = hashlib.sha256(
        json.dumps(values, sort_keys=True, ensure_ascii=False).encode()
    ).hexdigest()
    return payload


@lru_cache(maxsize=1)
def _verify_assets() -> None:
    """Detect changed vendored assets before invoking the deterministic processor."""
    try:
        manifest = json.loads((ASSET_ROOT / "manifest.json").read_text())
        for name, entry in manifest["files"].items():
            if (
                hashlib.sha256((ASSET_ROOT / name).read_bytes()).hexdigest()
                != entry["sha256"]
            ):
                raise RuntimeError("citation_assets_invalid")
    except (OSError, ValueError, KeyError) as exc:
        raise RuntimeError("citation_assets_unavailable") from exc


def format_citations(items: list[dict], clusters: list[list[str]], style: str) -> dict:
    """Render CSL citation clusters and their sorted bibliography without fallback."""
    if style not in ACADEMIC_STYLES:
        raise ValueError("unsupported_citation_style")
    node = shutil.which("node")
    if node is None:
        raise RuntimeError("citation_engine_unavailable")
    _verify_assets()
    request = json.dumps(
        {"style": style, "items": items, "clusters": clusters}, ensure_ascii=False
    )
    try:
        result = subprocess.run(
            [node, str(ASSET_ROOT / "adapter.js")],
            input=request,
            capture_output=True,
            text=True,
            timeout=15,
        )
    except subprocess.TimeoutExpired as exc:
        raise RuntimeError("citation_format_timeout") from exc
    if result.returncode:
        raise RuntimeError("citation_format_failed")
    return json.loads(result.stdout)


class _MarkdownHTML(HTMLParser):
    """Convert processor emphasis into safe Markdown without allowing raw HTML."""

    def __init__(self):
        super().__init__(convert_charrefs=True)
        self.parts = []

    def handle_starttag(self, tag, attrs):
        if tag in {"i", "em"}:
            self.parts.append("*")
        elif tag in {"b", "strong"}:
            self.parts.append("**")

    def handle_endtag(self, tag):
        if tag in {"i", "em"}:
            self.parts.append("*")
        elif tag in {"b", "strong"}:
            self.parts.append("**")

    def handle_data(self, data):
        self.parts.append(re.sub(r"([\\`*\[\]<>])", r"\\\1", data))


def csl_markdown(value: str) -> str:
    """Preserve CSL emphasis while escaping source-controlled Markdown syntax."""
    parser = _MarkdownHTML()
    parser.feed(value)
    return "".join(parser.parts).strip()


def bibliography_key(item_id: str) -> str:
    """Return a stable syntax-safe entry key without guessing author surnames."""
    return "litagent_" + hashlib.sha256(item_id.encode()).hexdigest()[:16]


def export_bibliography(snapshot: Mapping, format: str) -> bytes:
    """Serialize real fields to interoperable bibliography exchange formats."""
    if format not in BIBLIOGRAPHY_FORMATS:
        raise ValueError("unsupported_bibliography_format")
    items = snapshot["items"]
    if format == "csl_json":
        return (json.dumps(items, ensure_ascii=False, indent=2) + "\n").encode()
    output = []
    for item in items:
        year = (item.get("issued", {}).get("date-parts") or [[None]])[0][0]
        names = item.get("author", [])
        if format == "ris":
            lines = ["TY  - " + ("JOUR" if item.get("container-title") else "GEN")]
            for name in names:
                lines.append(
                    "AU  - "
                    + str(
                        name.get("literal")
                        or (name.get("family", "") + ", " + name.get("given", ""))
                    )
                    .replace("\n", " ")
                    .replace("\r", " ")
                )
            for key, tag in {
                "title": "TI",
                "container-title": "JO",
                "volume": "VL",
                "issue": "IS",
                "DOI": "DO",
                "URL": "UR",
                "page": "SP",
            }.items():
                if item.get(key):
                    value = str(item[key]).replace("\n", " ").replace("\r", " ")
                    if key == "page" and re.fullmatch(r"\d+[-–]\d+", value):
                        first, last = re.split("[-–]", value)
                        lines.extend(["SP  - " + first, "EP  - " + last])
                    else:
                        lines.append(tag + "  - " + value)
            if year:
                lines.append("PY  - " + str(year))
            lines.extend(["ID  - " + bibliography_key(item["id"]), "ER  - "])
            output.append("\n".join(lines))
        else:

            def escape(value):
                return "".join(
                    {
                        "\\": r"\textbackslash{}",
                        "{": r"\{",
                        "}": r"\}",
                        "%": r"\%",
                        "&": r"\&",
                        "#": r"\#",
                        "_": r"\_",
                        "$": r"\$",
                        "~": r"\textasciitilde{}",
                        "^": r"\textasciicircum{}",
                    }.get(c, c)
                    for c in str(value).replace("\n", " ").replace("\r", " ")
                )

            fields = []
            for key, field in {
                "title": "title",
                "container-title": "journal",
                "volume": "volume",
                "issue": "number",
                "page": "pages",
                "DOI": "doi",
                "URL": "url",
            }.items():
                if item.get(key):
                    fields.append(f"  {field} = {{{escape(item[key])}}}")
            if names:
                authors = [
                    (
                        "{" + escape(x["literal"]) + "}"
                        if x.get("literal")
                        else escape(x.get("family", ""))
                        + ", "
                        + escape(x.get("given", ""))
                    )
                    for x in names
                ]
                fields.append("  author = {" + " and ".join(authors) + "}")
            if year:
                fields.append("  year = {" + str(year) + "}")
            if item.get("archive"):
                fields.extend(
                    [
                        "  archivePrefix = {arXiv}",
                        "  eprint = {" + escape(item["archive_location"]) + "}",
                    ]
                )
            kind = "article" if item.get("container-title") else "misc"
            output.append(
                "@"
                + kind
                + "{"
                + bibliography_key(item["id"])
                + ",\n"
                + ",\n".join(fields)
                + "\n}"
            )
    return ("\n\n".join(output) + "\n").encode()
