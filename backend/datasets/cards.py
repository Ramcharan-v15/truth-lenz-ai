"""V5 dataset cards.

generate_card(dataset_id) -> markdown string, saved to datasets/cards/{dataset_id}.md

Sections: purpose, source, license, provenance, collection process, class
definitions, sample counts, known biases, limitations, leakage risks,
preprocessing, intended use, prohibited use.

Everything is derived from the registry record + manifest + quality audit.
Unknown information is stated as such, never invented.
"""
from __future__ import annotations

import os
from collections import Counter
from typing import Any, Dict

from .quality import audit_dataset, load_manifest
from .registry import get_dataset

_CARDS_DIR = os.path.abspath(
    os.path.join(os.path.dirname(__file__), "..", "..", "datasets", "cards"))


def generate_card(dataset_id: str) -> str:
    rec: Dict[str, Any] = get_dataset(dataset_id) or {}
    try:
        manifest = load_manifest(dataset_id)
        records = manifest.get("records", []) or []
    except KeyError:
        manifest, records = {}, []
    audit = audit_dataset(dataset_id)

    dist = Counter(str(r.get("class")) for r in records)
    counts = "\n".join(f"- {c}: {n}" for c, n in sorted(dist.items())) or "- none"

    def val(key: str, default: str = "not_provided") -> Any:
        v = rec.get(key, default)
        return v if v not in (None, "") else default

    lic = val("license_status", "UNKNOWN")
    lines = [
        f"# Dataset Card — {dataset_id}",
        "",
        f"**Name:** {val('name')}",
        f"**Version:** {val('version', '1.0')}",
        f"**Registered:** {val('registered_at', 'not_provided')}",
        f"**Quality status:** {audit['status']}",
        "",
        "## Purpose",
        "",
        val("purpose", "Real-world validation of TRUTH LENZ AI forensic detection. "
                       "NOT_YET_VALIDATED until a blind evaluation is executed."),
        "",
        "## Source",
        "",
        f"- Source: {val('source')}",
        f"- Creator/organization: {val('creator')}",
        f"- Acquisition date: {val('acquisition_date')}",
        f"- URL/reference: {val('url')}",
        "",
        "## License",
        "",
        f"- License: {val('license')}",
        f"- License status: **{lic}**",
        f"- Usage restrictions: {val('usage_restrictions')}",
        f"- Attribution requirements: {val('attribution_requirements')}",
        f"- License notes: {val('license_notes', '') or 'none'}",
        "",
        "> Legal usability is claimed ONLY when license status is "
        "LICENSE_VERIFIED. All other states mean: do not redistribute, "
        "do not assume research use is permitted.",
        "",
        "## Provenance",
        "",
        f"- Provenance: {val('provenance')}",
        f"- Collection methodology: {val('collection_methodology')}",
        "",
        "## Collection process",
        "",
        val("collection_methodology"),
        "",
        "## Class definitions",
        "",
    ]
    class_defs = rec.get("class_definitions") or []
    if class_defs:
        for cd in class_defs:
            lines.append(f"- {cd}")
    else:
        lines.append("- Defined per record `class` field "
                     "(authentic / ai_generated / real_edited / ai_edited / "
                     "screenshot_reencoded / unknown).")
    lines += [
        "",
        "## Sample counts",
        "",
        f"Total images: {len(records)}",
        "",
        counts,
        "",
        "## Generator / camera info",
        "",
        f"- Generator info: {val('generator_info', [])}",
        f"- Camera info: {val('camera_info', [])}",
        "",
        "## Known biases",
        "",
        "- Biases are dataset-specific and must be assessed per dataset.",
        "- Until assessed: assume selection bias, geographic bias, and "
        "device bias are present.",
        "",
        "## Limitations",
        "",
        "- Sample sizes may be insufficient for strong conclusions "
        "(see class counts above).",
        "- Ground truth comes from dataset provenance, never from the detector.",
        "- Synthetic metrics must not be compared against these results.",
        "",
        "## Leakage risks",
        "",
    ]
    leak_checks = [c for c in audit["checks"]
                   if c["name"] in ("exact_duplicates", "perceptual_duplicates")]
    for c in leak_checks:
        lines.append(f"- {c['name']}: {c['detail']}")
    lines += [
        "",
        "## Preprocessing",
        "",
        "Images are analyzed as supplied. Any preprocessing applied during "
        "ingestion is recorded in the manifest; none is applied silently.",
        "",
        "## Intended use",
        "",
        "- Blind real-world validation of forensic detection.",
        "- False-positive / false-negative analysis.",
        "- Calibration and OOD evaluation on real data.",
        "",
        "## Prohibited use",
        "",
        "- Do not redistribute image content unless the license permits it.",
        "- Do not use as training data without a new experiment record.",
        "- Do not present results without sample sizes and confidence intervals.",
        "",
        "## Quality audit summary",
        "",
        f"Status: **{audit['status']}**",
        "",
    ]
    for c in audit["checks"]:
        mark = "PASS" if c["passed"] else ("WARN" if not c["blocking"] else "FAIL")
        lines.append(f"- [{mark}] {c['name']}: {c['detail']}")
    if audit["reasons"]:
        lines += ["", "### Blocking reasons", ""]
        lines += [f"- {r}" for r in audit["reasons"]]
    lines += ["",
              "---",
              "",
              "*Generated by TRUTH LENZ AI V5 dataset card generator. "
              "Unknown fields are stated as unknown, never invented.*",
              ""]
    card = "\n".join(lines)

    os.makedirs(_CARDS_DIR, exist_ok=True)
    with open(os.path.join(_CARDS_DIR, f"{dataset_id}.md"),
              "w", encoding="utf-8") as fh:
        fh.write(card)
    return card


def card_path(dataset_id: str) -> str:
    return os.path.join(_CARDS_DIR, f"{dataset_id}.md")
