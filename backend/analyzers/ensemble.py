"""Ensemble engine: combines independent forensic signals into a verdict.

Design principles (spec sections 3, 21, 35, 36):
  * Weighted evidence - never a simple average. Weights come from
    detector_config.json and are documented there.
  * Probability and confidence are SEPARATE concepts.
  * Signal disagreement -> INCONCLUSIVE, never forced.
  * Poor quality / heavy compression / screenshots reduce confidence,
    never flip the verdict by themselves.
  * Missing metadata is never evidence of AI generation.
  * Probabilities are clamped to [1, 99]: never 100% certainty.
"""
from __future__ import annotations

from typing import Any, Dict, List, Tuple

from .base import (AIDetectorProvider, AnalysisContext, AnalyzerResult,
                   EvidenceDirection, NullDetectorProvider, SignalStatus, clamp)


VERDICTS = ("LIKELY_AUTHENTIC", "LIKELY_AI_GENERATED", "LIKELY_MANIPULATED", "INCONCLUSIVE")

VERDICT_EXPLANATIONS = {
    "LIKELY_AUTHENTIC": (
        "Forensic signals are broadly consistent with an authentic camera capture. "
        "No strong AI-generation or manipulation indicators were found. This is a "
        "probabilistic assessment, not proof of authenticity."
    ),
    "LIKELY_AI_GENERATED": (
        "Multiple forensic signals are consistent with synthetic image generation. "
        "The assessment weighs metadata, noise, frequency, texture and artifact "
        "signals together. Evidence is probabilistic, not definitive."
    ),
    "LIKELY_MANIPULATED": (
        "Forensic signals suggest the image was edited after capture - for example "
        "recompression traces, inconsistent noise, or duplicated regions. Editing "
        "ranges from benign retouching to deceptive manipulation."
    ),
    "INCONCLUSIVE": (
        "The available evidence does not support a confident classification. Signals "
        "may disagree, image quality may be too poor, or key metadata may be missing. "
        "Treat the image with appropriate skepticism."
    ),
}


def _weighted_mean(items: List[Tuple[float, float, float]]) -> Tuple[float, float]:
    """items: (score, base_weight, confidence).

    Opinion-weighted: a signal sitting at 50 ('no opinion') is down-weighted,
    while a decisive signal (far from 50, high confidence) counts more. This
    is weighted EVIDENCE, not a flat average - one strong finding is not
    drowned out by nine shrugs.
    """
    num, den = 0.0, 0.0
    for s, w, c in items:
        opinion = abs(s - 50.0) / 50.0          # 0 = no opinion, 1 = maximal
        eff = w * (0.25 + 1.75 * opinion * (c / 100.0))
        num += s * eff
        den += eff
    if den <= 1e-9:
        return 50.0, 0.0
    return num / den, den


def _decisive_support(usable: List[AnalyzerResult], axis: str):
    """Strongest analyzer scoring >=75 with >=60 confidence on this axis."""
    best = None
    for r in usable:
        score = r.ai_score if axis == "ai" else r.manip_score
        if score >= 75 and r.confidence >= 60:
            key = score * r.confidence
            if best is None or key > best[0]:
                best = (key, score, r.confidence, r.display_name)
    return best


def _strong_contradiction(usable: List[AnalyzerResult], axis: str):
    """Any analyzer scoring <=35 with >=60 confidence against this axis."""
    for r in usable:
        score = r.ai_score if axis == "ai" else r.manip_score
        if score <= 35 and r.confidence >= 60:
            return (score, r.confidence, r.display_name)
    return None


class EnsembleEngine:
    """Hybrid ensemble (V2): forensic signals + ML detector signals + provenance.

    Config (detector_config.json -> "ensemble_v2"):
      forensics_weight, ml_weight, provenance_weight,
      allow_ml_dominance (default false: ML can never outweigh forensics alone),
      ml_disagreement_threshold, ml_disagreement_penalty.
    """

    def __init__(self, config: dict, detector: AIDetectorProvider | None = None,
                 detectors: List[AIDetectorProvider] | None = None):
        self.config = config
        # backward compat: single detector -> one-element list
        if detectors is not None:
            self.detectors: List[AIDetectorProvider] = list(detectors)
        else:
            self.detectors = [detector or NullDetectorProvider()]
        self.weights: Dict[str, Dict[str, float]] = config.get("weights", {})
        self.thresholds = config.get("verdict_thresholds", {})
        self.conf_cfg = config.get("confidence", {})
        self.v2 = config.get("ensemble_v2", {})

    # ------------------------------------------------------------------
    def _run_detectors(self, ctx: AnalysisContext) -> List[Dict[str, Any]]:
        """Run every registered detector; normalize legacy + new results."""
        out = []
        for det in self.detectors:
            try:
                res = det.detect(ctx)
            except Exception as exc:
                res = None
                err = f"{type(exc).__name__}: {exc}"
            else:
                err = ""
            if res is None:
                out.append({
                    "detector_name": det.get_model_name(),
                    "detector_version": det.get_version(),
                    "model_type": det.get_model_type(),
                    "probability_ai": None, "confidence": 0.0,
                    "model_status": "error", "detail": err,
                    "available": False,
                })
                continue
            # normalize old DetectorResult -> new shape
            d = res.to_dict() if hasattr(res, "to_dict") else {}
            prob = d.get("probability_ai", getattr(res, "ai_probability", None))
            if prob is None and isinstance(d.get("ai_probability"), (int, float)):
                prob = d["ai_probability"]
            out.append({
                "detector_name": d.get("detector_name", d.get("model_name", det.get_model_name())),
                "detector_version": d.get("detector_version", d.get("model_version", det.get_version())),
                "model_type": d.get("model_type", det.get_model_type()),
                "probability_ai": prob,
                "probability_authentic": d.get("probability_authentic"),
                "raw_score": d.get("raw_score"),
                "calibration": d.get("calibration", "unknown"),
                "confidence": float(d.get("confidence", 0.0) or 0.0),
                "preprocessing": d.get("preprocessing", {}),
                "evidence": d.get("evidence", []),
                "limitations": d.get("limitations", det.get_limitations()),
                "inference_time_ms": d.get("inference_time_ms", 0.0),
                "model_status": d.get("model_status", "ready"),
                "demonstration": bool(d.get("demonstration", False)),
                "generator_label": d.get("generator_label"),
                "has_attention_heatmap": d.get("has_attention_heatmap", False),
                "detail": d.get("detail", ""),
                "available": bool(getattr(res, "available", False)),
            })
        return out

    def _provenance_signal(self, by_name: Dict[str, AnalyzerResult]) -> Tuple[float, float, str]:
        """Provenance contribution: C2PA state + camera EXIF -> AI-leaning score.

        Returns (ai_score, confidence, note). 50 = neutral.
        """
        md = by_name.get("metadata")
        if not md:
            return 50.0, 0.0, "no metadata signal"
        tech = md.technical or {}
        c2pa = tech.get("c2pa_status", "missing")
        if c2pa == "verified":
            return 30.0, 75.0, "C2PA credentials verified"
        if c2pa == "invalid":
            return 60.0, 55.0, "C2PA manifest present but invalid"
        # camera EXIF present -> leans authentic; generator tag -> leans AI
        ai_hit = any("AI generator signature" in e.text or "generation parameters" in e.text
                     for e in md.evidence)
        cam_hit = any("Camera capture metadata present" in e.text for e in md.evidence)
        if ai_hit:
            return 85.0, 80.0, "provenance names an AI generator"
        if cam_hit:
            return 35.0, 65.0, "camera provenance present"
        return 50.0, 40.0, "no provenance signal"

    # ------------------------------------------------------------------
    # ------------------------------------------------------------------
    def _false_positive_risk(self, by_name: Dict[str, AnalyzerResult],
                             verdict: str, ai_prob: float,
                             decisive_ai) -> Dict[str, Any]:
        """False-positive risk indicator (Phase 18).

        Scores known hard-negative traits: screenshots, heavy compression,
        stripped metadata, poor quality, denoised look. Elevated when the
        verdict is AI-generated WITHOUT decisive supporting evidence.
        Advisory only — never alters the verdict.
        """
        score = 0
        reasons: List[str] = []
        quality = by_name.get("quality")
        compression = by_name.get("compression")
        metadata = by_name.get("metadata")
        noise = by_name.get("noise")

        if quality:
            if quality.technical.get("screenshot_likely"):
                score += 30
                reasons.append("Possible screenshot: UI/text elements distort forensic signals")
            tier = quality.technical.get("quality_tier")
            if tier == "Poor":
                score += 15
                reasons.append("Very low resolution thins the evidence base")
        if compression:
            q = compression.technical.get("estimated_jpeg_quality")
            if q is not None and q <= 60:
                score += 20
                reasons.append(f"Heavy JPEG compression (est. quality {q}): real photos often look 'synthetic' when crushed")
        if metadata and metadata.status == SignalStatus.WARNING:
            score += 15
            reasons.append("Metadata stripped: provenance context unavailable (common for innocent shares)")
        if noise:
            ns = noise.technical.get("global_noise_std", noise.technical.get("noise_std"))
            if ns is not None and ns < 0.004:
                score += 10
                reasons.append("Unusually clean image: heavy denoising mimics synthetic smoothness")
        if verdict == "LIKELY_AI_GENERATED" and decisive_ai is None:
            score += 25
            reasons.append("AI verdict without a single decisive evidence item — treat with extra caution")

        level = "LOW" if score < 25 else ("MODERATE" if score < 50 else "ELEVATED")
        return {
            "level": level,
            "score": min(score, 100),
            "reasons": reasons,
            "note": ("Advisory: known false-positive traits present. Edited, compressed or "
                     "screenshot real photographs can resemble synthetic imagery to detectors."),
        }

    def combine(self, ctx: AnalysisContext,
                results: List[AnalyzerResult]) -> Dict[str, Any]:
        by_name = {r.name: r for r in results}
        usable = [r for r in results if r.status != SignalStatus.UNABLE and r.confidence > 0]

        ai_items, manip_items = [], []
        for r in usable:
            w = self.weights.get(r.name, {"ai": 1.0, "manip": 1.0})
            ai_items.append((r.ai_score, w.get("ai", 1.0), r.confidence))
            manip_items.append((r.manip_score, w.get("manip", 1.0), r.confidence))

        # ---- hybrid combination: forensics + ML + provenance ----
        forensic_ai, _ = _weighted_mean(ai_items)

        det_results = self._run_detectors(ctx)
        ml_ready = [d for d in det_results if d["available"] and d["probability_ai"] is not None]

        ml_ai = None
        ml_conf = 0.0
        ml_ood_status = "UNKNOWN"
        if ml_ready:
            num = sum(d["probability_ai"] * (d["confidence"] / 100.0) for d in ml_ready)
            den = sum(d["confidence"] / 100.0 for d in ml_ready)
            ml_ai = num / den if den > 0 else 50.0
            ml_conf = sum(d["confidence"] for d in ml_ready) / len(ml_ready)
            # V4: OOD status from the primary ML detector
            ml_ood_status = (ml_ready[0].get("preprocessing", {}) or {}).get(
                "ood_status", "UNKNOWN")

        # V5.1.8.64: when no forensic signal is usable, the ML input is
        # default values only, not measurements. Withhold the ML
        # probability from the verdict (null + explicit reason) instead
        # of presenting the model's extrapolation as a usable prediction.
        # The raw detector entries in "detectors" are preserved as
        # execution provenance. This changes no weight, threshold,
        # calibration, or model artifact: it is a validity rule on the
        # verdict block, triggered only when usable is empty.
        # Stage 2 (S2-3): the same validity rule extends to thin
        # coverage. The stacking model was trained on complete feature
        # vectors from full analyses; when a majority of the analyzers
        # are unusable, most of its input is defaults, not measurements,
        # and its output is extrapolation, not a measurement-based
        # prediction. The majority line (fewer than half the analyzers
        # usable) is a validity convention, not a calibrated threshold:
        # it does not change any probability threshold, weight, or
        # model artifact. Limitation: signal count is a proxy for
        # feature coverage (technical features are not evenly spread
        # across analyzers); a future refinement could measure
        # feature-level coverage directly.
        # Stage 2 (S2-1): a withheld detector output must not be
        # presented as available anywhere. The withheld entries keep
        # their raw probability as execution provenance, but their
        # "available" flag is cleared and they carry an explicit
        # suppression marker, so every consumer (dashboard, exports,
        # explanations) treats them as withheld, not calibrated.
        n_usable = len(usable)
        n_total = len(results)
        ml_suppressed = bool(ml_ready) and (not usable or n_usable * 2 < n_total)
        ml_suppressed_reason = None
        if ml_suppressed:
            ml_ai = None
            if not usable:
                ml_suppressed_reason = (
                    "No usable forensic signals; ML probability withheld - "
                    "model input was default values only, not measurements.")
            else:
                ml_suppressed_reason = (
                    f"Only {n_usable} of {n_total} forensic signals usable; "
                    "ML probability withheld - model input was mostly "
                    "default values, not measurements.")
            for d in ml_ready:
                d["available"] = False
                d["suppressed"] = True
                d["suppressed_reason"] = ml_suppressed_reason

        prov_ai, prov_conf, prov_note = self._provenance_signal(by_name)

        fw = float(self.v2.get("forensics_weight", 1.0))
        mw = float(self.v2.get("ml_weight", 1.0))
        pw = float(self.v2.get("provenance_weight", 0.6))

        # RULE: ML must not automatically dominate forensics.
        if ml_ai is not None and not self.v2.get("allow_ml_dominance", False):
            mw = min(mw, fw)
        # RULE (V3): demonstration-grade models are advisory only — their
        # ensemble weight is scaled down to reflect unvalidated competence.
        if ml_ready and all(d.get("demonstration", False) for d in ml_ready):
            demo_scale = float(self.v2.get("demonstration_weight_scale", 0.5))
            mw = mw * demo_scale

        parts = [(forensic_ai, fw)]
        if ml_ai is not None:
            parts.append((ml_ai, mw * (ml_conf / 100.0)))
        if prov_conf > 0:
            parts.append((prov_ai, pw * (prov_conf / 100.0)))
        total_w = sum(w for _, w in parts) or 1.0
        ai_prob = sum(s * w for s, w in parts) / total_w

        # ML-forensics disagreement -> uncertainty, never silent override.
        # Demonstration-grade models contribute probability + a confidence
        # penalty, but do NOT get veto power over forensic verdicts: they are
        # not validated on real-world imagery (see MODEL_CARD.md).
        ml_disagreement = 0.0
        ml_disagree_flag = False
        ml_veto = False  # forced-INCONCLUSIVE veto (production models only)
        if ml_ai is not None:
            ml_disagreement = abs(ml_ai - forensic_ai)
            ml_thresh = float(self.v2.get("ml_disagreement_threshold", 25))
            if ml_disagreement > ml_thresh and ml_conf >= 60:
                ml_disagree_flag = True
                ml_demo = bool(ml_ready[0].get("demonstration", False)) if ml_ready else False
                ml_veto = not ml_demo

        manip_prob, manip_w = _weighted_mean(manip_items)

        # authenticity: inverse of the stronger suspicious hypothesis, tempered by
        # pro-authentic evidence count
        pro_authentic = sum(
            1 for r in usable for e in r.evidence
            if e.direction == EvidenceDirection.SUPPORTS_AUTHENTIC
        )
        authentic_prob = clamp(100 - max(ai_prob, manip_prob) + min(pro_authentic * 2, 12))

        # never claim certainty
        ai_prob = clamp(ai_prob, 1, 99)
        manip_prob = clamp(manip_prob, 1, 99)
        authentic_prob = clamp(authentic_prob, 1, 99)

        # --- signal agreement ---
        ai_scores = [s for s, _, _ in ai_items]
        weights_c = [w * (c / 100.0) for _, w, c in ai_items]
        mean_ai = sum(s * wc for s, wc in zip(ai_scores, weights_c)) / (sum(weights_c) or 1)
        var = sum(wc * (s - mean_ai) ** 2 for s, wc in zip(ai_scores, weights_c)) / (sum(weights_c) or 1)
        disagreement = var ** 0.5
        disagree_threshold = self.conf_cfg.get("disagreement_std_threshold", 18)

        # --- confidence (separate from probability) ---
        penalties_cfg = self.conf_cfg.get("penalties", {})
        base_conf = sum(r.confidence for r in usable) / max(len(usable), 1)
        penalty_reasons: List[str] = []
        quality = by_name.get("quality")
        compression = by_name.get("compression")
        metadata = by_name.get("metadata")

        def penalize(key: str, reason: str):
            nonlocal base_conf
            p = penalties_cfg.get(key, 0)
            if p:
                base_conf -= p
                penalty_reasons.append(reason)

        if quality:
            tier = quality.technical.get("quality_tier")
            if tier == "Poor":
                penalize("poor_quality", "Very low resolution thins the evidence")
            elif tier == "Fair":
                penalize("fair_quality", "Reduced resolution/clarity")
            if quality.technical.get("screenshot_likely"):
                penalize("screenshot_detected", "Possible screenshot alters forensic signals")
            if quality.technical.get("ai_common_size"):
                pass  # size note, not a confidence penalty
        if compression and compression.technical.get("estimated_jpeg_quality") is not None:
            if compression.technical["estimated_jpeg_quality"] <= 60:
                penalize("heavy_compression", "Heavy JPEG compression degrades fine signals")
        if metadata and metadata.status == SignalStatus.WARNING:
            penalize("missing_metadata", "No embedded metadata for provenance context")
        if disagreement > disagree_threshold:
            penalize("signal_disagreement",
                     f"Signals disagree (std dev {disagreement:.0f}) - classification uncertain")
        if ml_disagree_flag:
            ml_pen = float(self.v2.get("ml_disagreement_penalty", 15))
            base_conf -= ml_pen
            penalty_reasons.append(
                f"ML detector disagrees with forensics by {ml_disagreement:.0f} points - uncertainty increased")
        if ctx.extra.get("downscaled"):
            penalize("resized_likely", "Image was downscaled for analysis")
        confidence = clamp(base_conf, 5, 98)

        # --- verdict ---
        t = self.thresholds
        ai_t = t.get("ai_likely", 60)
        manip_t = t.get("manip_likely", 60)
        margin = t.get("verdict_margin", 12)
        auth_ai_max = t.get("authentic_max_ai", 40)
        auth_manip_max = t.get("authentic_max_manip", 40)

        # Stage 2 (S2-2): a resolution claim needs a real measurement.
        # A quality result in UNABLE state carries only an error in its
        # technical block — no width/height — so "pixels" below the
        # threshold previously meant "unmeasured", yet the verdict
        # reason still claimed "Resolution too low". Only measured
        # dimensions below the threshold support that reason. When no
        # usable signal exists at all, the honest reason is neutral
        # insufficiency, not resolution. Consequence: a quality-UNABLE
        # report WITH usable signals no longer forces INCONCLUSIVE on
        # a false resolution pretext; its verdict follows the actual
        # evidence chain below.
        quality_measured = (
            quality is not None
            and quality.status != SignalStatus.UNABLE
            and isinstance(quality.technical.get("width"), (int, float))
            and isinstance(quality.technical.get("height"), (int, float))
        )
        resolution_insufficient = False
        if quality_measured:
            pixels = quality.technical.get("width", 0) * quality.technical.get("height", 1)
            if pixels < self.config.get("quality", {}).get("min_analysis_pixels", 16384):
                resolution_insufficient = True
        insufficient = resolution_insufficient or not usable

        verdict = "INCONCLUSIVE"
        verdict_reason = ""
        decisive_ai = _decisive_support(usable, "ai")
        decisive_manip = _decisive_support(usable, "manip")
        contra_ai = _strong_contradiction(usable, "ai")
        contra_manip = _strong_contradiction(usable, "manip")

        if resolution_insufficient:
            verdict_reason = "Resolution too low for meaningful forensic analysis (insufficient evidence)."
        elif not usable:
            verdict_reason = "Insufficient usable forensic evidence for a supported verdict."
        elif ml_veto and abs(ai_prob - 50) < 20:
            # A production-validated ML model disagrees significantly with
            # forensics: increase uncertainty rather than forcing a verdict.
            verdict = "INCONCLUSIVE"
            verdict_reason = (f"ML detector ({ml_ai:.0f}%) disagrees with forensic signals "
                              f"({forensic_ai:.0f}%) by {ml_disagreement:.0f} points. "
                              "Conflicting evidence of this strength cannot support a verdict.")
        elif (decisive_ai and not contra_ai and ai_prob >= ai_t
              and (not decisive_manip or decisive_ai[1] >= decisive_manip[1])
              and not ml_veto):
            # Direct, uncontradicted evidence (e.g. an AI-generator tag in metadata):
            # one decisive signal plus neutral others is agreement, not disagreement.
            # Checked BEFORE the disagreement branch: a lone strong finding with no
            # contradicting signal is a finding, not a conflict.
            verdict = "LIKELY_AI_GENERATED"
            verdict_reason = (f"Decisive signal from {decisive_ai[3]} ({decisive_ai[1]:.0f}/100, "
                              f"{decisive_ai[2]:.0f}% confidence) with no strong contradiction; "
                              f"ensemble AI probability {ai_prob:.0f}%.")
        elif (decisive_manip and not contra_manip and manip_prob >= manip_t
              and (not decisive_ai or decisive_manip[1] > decisive_ai[1])
              and not ml_veto):
            verdict = "LIKELY_MANIPULATED"
            verdict_reason = (f"Decisive signal from {decisive_manip[3]} ({decisive_manip[1]:.0f}/100, "
                              f"{decisive_manip[2]:.0f}% confidence) with no strong contradiction; "
                              f"ensemble manipulation probability {manip_prob:.0f}%.")
        elif disagreement > disagree_threshold and abs(ai_prob - 50) < 25 and abs(manip_prob - 50) < 25:
            verdict = "INCONCLUSIVE"
            verdict_reason = (f"Forensic signals genuinely conflict (spread {disagreement:.0f} with "
                              "contradicting directions). Forcing a classification would be misleading.")
        elif ai_prob >= ai_t and (ai_prob - manip_prob) >= margin:
            verdict = "LIKELY_AI_GENERATED"
            verdict_reason = (f"AI-generation probability {ai_prob:.0f}% exceeds the {ai_t}% threshold "
                              f"with a {ai_prob - manip_prob:.0f}-point margin over manipulation.")
        elif manip_prob >= manip_t and (manip_prob - ai_prob) >= margin:
            verdict = "LIKELY_MANIPULATED"
            verdict_reason = (f"Manipulation probability {manip_prob:.0f}% exceeds the {manip_t}% threshold "
                              f"with a {manip_prob - ai_prob:.0f}-point margin over AI generation.")
        elif ai_prob <= auth_ai_max and manip_prob <= auth_manip_max:
            verdict = "LIKELY_AUTHENTIC"
            verdict_reason = (f"Both AI ({ai_prob:.0f}%) and manipulation ({manip_prob:.0f}%) probabilities "
                              "are below their suspicion thresholds.")
        else:
            verdict_reason = ("Probabilities sit in the uncertain band or the leading hypothesis "
                              f"lacks the required {margin}-point margin.")

        # --- explainability: strongest evidence for / against ---
        for_ev, against_ev, uncertain = [], [], []
        for r in usable:
            for e in r.evidence:
                item = {"analyzer": r.display_name, "text": e.text, "detail": e.detail}
                if e.direction == EvidenceDirection.SUPPORTS_AI and verdict == "LIKELY_AI_GENERATED":
                    for_ev.append(item)
                elif e.direction == EvidenceDirection.SUPPORTS_MANIP and verdict == "LIKELY_MANIPULATED":
                    for_ev.append(item)
                elif e.direction == EvidenceDirection.SUPPORTS_AUTHENTIC and verdict == "LIKELY_AUTHENTIC":
                    for_ev.append(item)
                elif e.direction in (EvidenceDirection.SUPPORTS_AUTHENTIC,):
                    against_ev.append(item)
                elif e.direction == EvidenceDirection.UNCERTAIN:
                    uncertain.append(item)
        # cap lists
        for_ev, against_ev, uncertain = for_ev[:8], against_ev[:8], uncertain[:6]

        # why-summary: top contributing analyzers
        contributors = sorted(
            [(r.display_name, r.ai_score if verdict == "LIKELY_AI_GENERATED" else r.manip_score,
              r.confidence) for r in usable],
            key=lambda x: abs(x[1] - 50) * (x[2] / 100.0), reverse=True,
        )[:3]
        why_lines = [
            f"{name}: scored {score:.0f}/100 with {conf:.0f}% signal confidence"
            for name, score, conf in contributors
        ]

        # --- explicit uncertainty breakdown (V3, Phase 17; V4: +OOD) ---
        uncertainty = None
        ood_concern = ml_ood_status in ("MILD_SHIFT", "OUT_OF_DISTRIBUTION")
        if verdict == "INCONCLUSIVE" and (ml_disagree_flag or disagreement > disagree_threshold or ood_concern):
            parties = []
            parties.append({
                "side": "Forensic signals",
                "ai_probability": round(forensic_ai, 1),
                "position": "AI-generated" if forensic_ai >= 60 else ("authentic" if forensic_ai <= 40 else "undecided"),
            })
            if ml_ai is not None:
                ml_demo = bool(ml_ready[0].get("demonstration", False)) if ml_ready else False
                parties.append({
                    "side": (f"ML detector ({ml_ready[0]['detector_name']})"
                             + (" [demonstration model]" if ml_demo else "")) if ml_ready else "ML detector",
                    "ai_probability": round(ml_ai, 1),
                    "position": "AI-generated" if ml_ai >= 60 else ("authentic" if ml_ai <= 40 else "undecided"),
                })
            parties.append({
                "side": "Provenance",
                "ai_probability": round(prov_ai, 1),
                "position": prov_note,
            })
            uncertainty = {
                "title": "WHY THE SYSTEM IS UNCERTAIN",
                "parties": parties,
                "gap": round(ml_disagreement if ml_disagree_flag else disagreement, 1),
                "ood_status": ml_ood_status,
                "explanation": ("Independent evidence sources disagree too strongly to support "
                                "a verdict. Forcing a classification would be misleading — "
                                "the honest answer is INCONCLUSIVE."
                                + (f" The ML input is {ml_ood_status.lower().replace('_', ' ')} "
                                   "relative to its training distribution." if ood_concern else "")),
            }

        conf_label = "HIGH" if confidence >= 70 else ("MEDIUM" if confidence >= 45 else "LOW")

        # --- false-positive risk indicator (V3, Phase 18) ---
        # Edited/compressed/screenshot real photos must not silently become "AI".
        # This flag does NOT change the verdict; it warns when the verdict rests
        # on shaky ground for known hard-negative traits.
        fp_risk = self._false_positive_risk(by_name, verdict, ai_prob, decisive_ai)

        # backward compat: "detector" = first detector summary (or null-detector)
        first_det = det_results[0] if det_results else {
            "detector_name": "No trained model configured", "detector_version": "n/a",
            "model_type": "none", "probability_ai": None, "confidence": 0.0,
            "model_status": "not_configured", "available": False,
            "limitations": "No trained AI-detection model is connected.",
        }

        return {
            "verdict": verdict,
            "verdict_explanation": VERDICT_EXPLANATIONS[verdict],
            "verdict_reason": verdict_reason,
            "ai_probability": round(ai_prob, 1),
            "manipulation_probability": round(manip_prob, 1),
            "authenticity_probability": round(authentic_prob, 1),
            "confidence": round(confidence, 1),
            "confidence_label": conf_label,
            "confidence_penalties": penalty_reasons,
            "signal_disagreement": round(disagreement, 1),
            "signals_agree": disagreement <= disagree_threshold and not ml_disagree_flag,
            "insufficient_evidence": insufficient,
            "probability_bands": self.config.get("bands", {}).get("ai_probability_bands", []),
            "why": why_lines,
            "strongest_evidence": for_ev,
            "evidence_against": against_ev,
            "uncertain_factors": uncertain,
            "detector": first_det,
            "detectors": det_results,
            "ml_available": len(ml_ready) > 0 and not ml_suppressed,
            "ml_ai_probability": round(ml_ai, 1) if ml_ai is not None else None,
            "ml_suppressed_reason": ml_suppressed_reason,
            "forensic_ai_probability": round(forensic_ai, 1),
            "provenance": {"ai_score": round(prov_ai, 1), "confidence": round(prov_conf, 1),
                           "note": prov_note},
            "ml_disagreement": round(ml_disagreement, 1),
            "ml_ood_status": ml_ood_status,
            "uncertainty": uncertainty,
            "false_positive_risk": fp_risk,
            "hybrid_weights": {"forensics": fw, "ml": mw if ml_ai is not None else 0.0,
                               "provenance": pw},
            "weights_used": {k: {"ai": v.get("ai"), "manip": v.get("manip")}
                             for k, v in self.weights.items() if isinstance(v, dict)},
            "thresholds_used": self.thresholds,
            "disclaimer": ("Probability estimates, not definitive proof. No detector can guarantee "
                           "whether an image is AI-generated or authentic."),
        }
