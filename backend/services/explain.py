"""Deterministic explanation generator (Phase 22).

Three levels - BASIC, TECHNICAL, FORENSIC - generated from STRUCTURED
evidence only. This is not an LLM: it cannot invent findings, it only
rephrases computed values. An optional LLM layer (backend/llm) may later
translate these same structures into prose, but it may never compute scores.
"""
from __future__ import annotations

from typing import Any, Dict, List

VERDICT_BASIC = {
    "LIKELY_AI_GENERATED": "This image shows several signals associated with AI-generated imagery.",
    "LIKELY_AUTHENTIC": "This image looks consistent with an authentic camera photograph.",
    "LIKELY_MANIPULATED": "This image shows signs of having been edited after it was captured.",
    "INCONCLUSIVE": "There isn't enough clear evidence to classify this image confidently.",
}


def _top_evidence(verdict_block: Dict[str, Any], n: int = 3) -> List[Dict[str, Any]]:
    return verdict_block.get("strongest_evidence", [])[:n]


def build_basic(verdict_block: Dict[str, Any]) -> str:
    v = verdict_block["verdict"]
    lines = [VERDICT_BASIC.get(v, "")]
    top = _top_evidence(verdict_block)
    if top:
        lines.append("The main reasons: " + "; ".join(
            e["text"].rstrip(".") for e in top) + ".")
    lines.append(
        f"We estimate a {verdict_block['ai_probability']:.0f}% chance of AI generation "
        f"and hold that estimate with {verdict_block['confidence_label'].lower()} "
        f"confidence ({verdict_block['confidence']:.0f}%).")
    against = verdict_block.get("evidence_against", [])[:2]
    if against:
        lines.append("On the other hand: " + "; ".join(
            e["text"].rstrip(".") for e in against) + ".")
    lines.append("These are probability estimates, not proof.")
    return " ".join(lines)


def build_technical(verdict_block: Dict[str, Any],
                    signals: List[Dict[str, Any]]) -> str:
    v = verdict_block["verdict"]
    parts = [
        f"Verdict: {v} (AI {verdict_block['ai_probability']:.1f}%, "
        f"manipulation {verdict_block['manipulation_probability']:.1f}%, "
        f"confidence {verdict_block['confidence_label']} {verdict_block['confidence']:.0f}%).",
        verdict_block.get("verdict_reason", ""),
    ]
    scored = sorted(
        ((s["display_name"], s["ai_score"], s["manip_score"], s["confidence"])
         for s in signals if s.get("status") != "unable"),
        key=lambda x: abs(x[1] - 50), reverse=True)[:4]
    parts.append("Strongest signal contributions: " + "; ".join(
        f"{name} (AI {ai:.0f}, manip {mp:.0f}, conf {cf:.0f}%)"
        for name, ai, mp, cf in scored) + ".")
    hw = verdict_block.get("hybrid_weights", {})
    if hw:
        parts.append(
            f"Hybrid combination weights - forensics {hw.get('forensics')}, "
            f"ML {hw.get('ml')}, provenance {hw.get('provenance')}.")
    if verdict_block.get("ml_available"):
        parts.append(
            f"ML detector P(AI) {verdict_block.get('ml_ai_probability')}% vs "
            f"forensic P(AI) {verdict_block.get('forensic_ai_probability')}%; "
            f"disagreement {verdict_block.get('ml_disagreement')} points.")
    pens = verdict_block.get("confidence_penalties", [])
    if pens:
        parts.append("Confidence reduced by: " + "; ".join(pens) + ".")
    return " ".join(p for p in parts if p)


def build_forensic(verdict_block: Dict[str, Any],
                   signals: List[Dict[str, Any]]) -> str:
    parts = [build_technical(verdict_block, signals)]
    parts.append(
        "Per-signal detail (score | confidence | status): " + "; ".join(
            f"{s['display_name']} {s['ai_score']:.0f}/{s['manip_score']:.0f} | "
            f"{s['confidence']:.0f}% | {s.get('status')}"
            for s in signals) + ".")
    dets = verdict_block.get("detectors", [])
    if dets:
        for d in dets:
            if d.get("suppressed"):
                # Stage 2 (S2-1): never label a withheld output as
                # calibrated. The raw value stays visible as execution
                # provenance, explicitly marked withheld, never as a
                # calibrated probability.
                parts.append(
                    f"Detector '{d.get('detector_name')}' v{d.get('detector_version')} "
                    f"({d.get('model_type')}, status {d.get('model_status')}): "
                    f"calibrated P(AI)=withheld (raw model output "
                    f"{d.get('raw_score')} recorded as execution provenance only), "
                    f"calibration={d.get('calibration')}, "
                    f"inference {d.get('inference_time_ms')}ms.")
            else:
                parts.append(
                    f"Detector '{d.get('detector_name')}' v{d.get('detector_version')} "
                    f"({d.get('model_type')}, status {d.get('model_status')}): "
                    f"raw={d.get('raw_score')}, calibrated P(AI)={d.get('probability_ai')}%, "
                    f"calibration={d.get('calibration')}, "
                    f"inference {d.get('inference_time_ms')}ms.")
    else:
        parts.append("No ML detector configured; verdict from forensic signals only.")
    if verdict_block.get("ml_suppressed_reason"):
        # The detector output printed above is execution provenance only.
        # Say so explicitly so the percentage is never read as a usable
        # prediction when the verdict block withholds it.
        parts.append(
            f"ML probability withheld: {verdict_block['ml_suppressed_reason']} "
            "Any detector output above is execution provenance only, not a usable prediction.")
    parts.append(
        f"Thresholds: AI>={verdict_block.get('thresholds_used', {}).get('ai_likely')}, "
        f"manip>={verdict_block.get('thresholds_used', {}).get('manip_likely')}, "
        f"margin>={verdict_block.get('thresholds_used', {}).get('verdict_margin')}. "
        f"Signal spread (std) {verdict_block.get('signal_disagreement')}.")
    return " ".join(parts)


def build_explanations(verdict_block: Dict[str, Any],
                       signals: List[Any]) -> Dict[str, str]:
    sig_dicts = [s if isinstance(s, dict) else s.to_dict() for s in signals]
    return {
        "basic": build_basic(verdict_block),
        "technical": build_technical(verdict_block, sig_dicts),
        "forensic": build_forensic(verdict_block, sig_dicts),
        "note": ("Generated deterministically from computed evidence. "
                 "No language model was involved in scoring."),
    }
