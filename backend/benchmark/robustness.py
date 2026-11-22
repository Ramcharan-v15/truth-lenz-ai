"""Robustness testing (Phase 12).

Answers: "How easily can ordinary image processing break our detector?"

Each transformation is applied to a test image, the pipeline is re-run, and
stability is measured as:
  - probability change (original vs transformed P(AI) / P(manipulation))
  - verdict flip (did the classification change?)
  - robustness score per transform and overall

Transformations: JPEG compression, resize, crop, blur, sharpen, screenshot
simulation, color adjustment, noise addition, metadata stripping.
"""
from __future__ import annotations

import io
from typing import Any, Callable, Dict, List

import numpy as np
from PIL import Image, ImageEnhance, ImageFilter, ImageOps

# (transform_id, display_name, function)
Transform = Callable[[Image.Image], Image.Image]


def t_jpeg_recompress(quality: int = 60) -> Transform:
    def f(img: Image.Image) -> Image.Image:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, "JPEG", quality=quality)
        buf.seek(0)
        return Image.open(buf).copy()
    return f


def t_resize(scale: float = 0.5) -> Transform:
    def f(img: Image.Image) -> Image.Image:
        w, h = img.size
        return img.resize((max(1, int(w * scale)), max(1, int(h * scale))), Image.BILINEAR)
    return f


def t_center_crop(frac: float = 0.8) -> Transform:
    def f(img: Image.Image) -> Image.Image:
        w, h = img.size
        nw, nh = int(w * frac), int(h * frac)
        x0, y0 = (w - nw) // 2, (h - nh) // 2
        return img.crop((x0, y0, x0 + nw, y0 + nh))
    return f


def t_gaussian_blur(radius: float = 1.5) -> Transform:
    def f(img: Image.Image) -> Image.Image:
        return img.filter(ImageFilter.GaussianBlur(radius))
    return f


def t_sharpen() -> Transform:
    def f(img: Image.Image) -> Image.Image:
        return img.filter(ImageFilter.UnsharpMask(radius=2, percent=150, threshold=3))
    return f


def t_screenshot_sim() -> Transform:
    """Simulate a screenshot: slight downscale + recompress + strip metadata."""
    def f(img: Image.Image) -> Image.Image:
        small = img.resize((int(img.width * 0.9), int(img.height * 0.9)), Image.BILINEAR)
        buf = io.BytesIO()
        small.convert("RGB").save(buf, "PNG")
        buf.seek(0)
        return Image.open(buf).copy()
    return f


def t_color_adjust(brightness: float = 1.15, contrast: float = 1.1) -> Transform:
    def f(img: Image.Image) -> Image.Image:
        out = ImageEnhance.Brightness(img).enhance(brightness)
        return ImageEnhance.Contrast(out).enhance(contrast)
    return f


def t_add_noise(sigma: float = 5.0, seed: int = 0) -> Transform:
    def f(img: Image.Image) -> Image.Image:
        rng = np.random.default_rng(seed)
        arr = np.asarray(img.convert("RGB"), dtype=np.float32)
        noisy = np.clip(arr + rng.normal(0, sigma, arr.shape), 0, 255).astype(np.uint8)
        return Image.fromarray(noisy)
    return f


def t_strip_metadata() -> Transform:
    """Re-encode without any metadata (what messaging apps do)."""
    def f(img: Image.Image) -> Image.Image:
        buf = io.BytesIO()
        img.convert("RGB").save(buf, "JPEG", quality=95)
        buf.seek(0)
        return Image.open(buf).copy()
    return f


TRANSFORMS: List[Dict[str, Any]] = [
    {"id": "jpeg_q60", "name": "JPEG recompress (q=60)", "fn": t_jpeg_recompress(60)},
    {"id": "jpeg_q30", "name": "JPEG recompress (q=30)", "fn": t_jpeg_recompress(30)},
    {"id": "resize_half", "name": "Resize 50%", "fn": t_resize(0.5)},
    {"id": "crop_80", "name": "Center crop 80%", "fn": t_center_crop(0.8)},
    {"id": "blur", "name": "Gaussian blur (r=1.5)", "fn": t_gaussian_blur(1.5)},
    {"id": "sharpen", "name": "Sharpen", "fn": t_sharpen()},
    {"id": "screenshot", "name": "Screenshot simulation", "fn": t_screenshot_sim()},
    {"id": "color", "name": "Brightness/contrast adjust", "fn": t_color_adjust()},
    {"id": "noise", "name": "Gaussian noise (sigma=5)", "fn": t_add_noise()},
    {"id": "strip_meta", "name": "Metadata strip", "fn": t_strip_metadata()},
]


def robustness_score(delta_ai: float, delta_manip: float, verdict_flipped: bool) -> float:
    """Deterministic robustness score (V4, spec §20).

    Formula:
        score = max(0, 100 − mean(|Δai|, |Δmanip|) − 25·[verdict flipped])

    Inputs: probability deltas in percentage points (0-100), boolean flip.
    Range: 0 (brittle) to 100 (perfectly stable).
    Interpretation: >90 stable, 70-90 moderate, <70 brittle.
    Limitations: linear penalty; treats all transforms equally; does not
    measure whether the ORIGINAL verdict was correct.
    """
    instability = (abs(delta_ai) + abs(delta_manip)) / 2.0
    if verdict_flipped:
        instability += 25.0
    return round(max(0.0, 100.0 - instability), 1)


def _encode_for_pipeline(img: Image.Image) -> bytes:
    buf = io.BytesIO()
    img.convert("RGB").save(buf, "JPEG", quality=95)
    return buf.getvalue()


def evaluate_robustness(run_fn, image_bytes: bytes, filename: str,
                        transforms: List[Dict[str, Any]] | None = None) -> Dict[str, Any]:
    """run_fn(filename, mime, data) -> report dict. Returns per-transform
    stability: original vs transformed prediction."""
    transforms = transforms or TRANSFORMS
    base = run_fn(filename, "image/jpeg", image_bytes)
    base_ens = base["ensemble"]
    base_ai = base_ens["ai_probability"]
    base_manip = base_ens["manipulation_probability"]
    base_verdict = base_ens["verdict"]

    # V5.1.8.64: if the original analysis withheld its probabilities
    # (numerical fallback), no delta is computable. Return an explicit
    # unavailable result instead of letting every transform row fail
    # into a misleading 0.0 "brittle" score or a generic client error.
    if base_ai is None or base_manip is None:
        return {
            "filename": filename,
            "original": {"ai_probability": base_ai,
                         "manipulation_probability": base_manip,
                         "verdict": base_verdict},
            "transforms": [],
            "overall_robustness_score": None,
            "verdict_flips": 0,
            "transforms_measured": 0,
            "transforms_unavailable": 0,
            "numerical_evidence": "unavailable",
            "unavailable_reason": (
                "Original analysis withheld verdict probabilities "
                "(invalid numeric values); robustness deltas cannot be computed."),
            "interpretation": (
                "Robustness could not be measured: the original analysis "
                "withheld verdict probabilities because numerical validation "
                "found non-finite values. This is unavailable evidence, "
                "not detector brittleness."),
        }

    rows: List[Dict[str, Any]] = []
    for t in transforms:
        try:
            img = Image.open(io.BytesIO(image_bytes)).convert("RGB")
            out = t["fn"](img)
            data = _encode_for_pipeline(out)
            rep = run_fn(f"{filename} [{t['id']}]", "image/jpeg", data)
            ens = rep["ensemble"]
            # V5.1.8.64: a transformed analysis that withheld its
            # probabilities is unavailable evidence, not a failed
            # transform and not brittleness. Record it with null deltas
            # and exclude it from the aggregate (never a 0.0 score).
            t_ai = ens["ai_probability"]
            t_manip = ens["manipulation_probability"]
            if t_ai is None or t_manip is None:
                ood_base = (base_ens.get("ml_ood_status") or "UNKNOWN")
                ood_new = (ens.get("ml_ood_status") or "UNKNOWN")
                rows.append({
                    "transform_id": t["id"],
                    "transform_name": t["name"],
                    "original_probability": base_ai,
                    "transformed_probability": None,
                    "probability_delta": None,
                    "ai_probability": None,
                    "manipulation_probability": None,
                    "verdict": ens["verdict"],
                    "original_verdict": base_verdict,
                    "delta_ai": None,
                    "delta_manipulation": None,
                    "verdict_changed": ens["verdict"] != base_verdict,
                    "verdict_flipped": ens["verdict"] != base_verdict,
                    "confidence_change": None,
                    "ood_change": f"{ood_base} -> {ood_new}" if ood_base != ood_new else "unchanged",
                    "robustness_score": None,
                    "numerical_evidence": "unavailable",
                    "unavailable_reason": (
                        "Transformed analysis withheld verdict probabilities "
                        "(invalid numeric values); deltas unavailable."),
                })
                continue
            d_ai = abs(ens["ai_probability"] - base_ai)
            d_manip = abs(ens["manipulation_probability"] - base_manip)
            flipped = ens["verdict"] != base_verdict
            d_conf = abs(ens["confidence"] - base_ens["confidence"])
            ood_base = (base_ens.get("ml_ood_status") or "UNKNOWN")
            ood_new = (ens.get("ml_ood_status") or "UNKNOWN")
            score = robustness_score(d_ai, d_manip, flipped)
            rows.append({
                "transform_id": t["id"],
                "transform_name": t["name"],
                "original_probability": base_ai,
                "transformed_probability": ens["ai_probability"],
                "probability_delta": round(d_ai, 1),
                "ai_probability": ens["ai_probability"],
                "manipulation_probability": ens["manipulation_probability"],
                "verdict": ens["verdict"],
                "original_verdict": base_verdict,
                "delta_ai": round(d_ai, 1),
                "delta_manipulation": round(d_manip, 1),
                "verdict_changed": flipped,
                "verdict_flipped": flipped,
                "confidence_change": round(d_conf, 1),
                "ood_change": f"{ood_base} -> {ood_new}" if ood_base != ood_new else "unchanged",
                "robustness_score": score,
            })
        except Exception as exc:
            rows.append({"transform_id": t["id"], "transform_name": t["name"],
                         "error": f"{type(exc).__name__}: {str(exc)[:120]}",
                         "robustness_score": 0.0})

    # Measured rows only: unavailable rows are excluded (never 0.0).
    # Genuine error rows keep the pre-existing 0.0 convention, and if
    # every row errored the overall stays 0.0 as before. Overall is
    # null only when nothing was measured and nothing errored.
    measured = [r for r in rows
                if "error" not in r and r.get("robustness_score") is not None]
    errored = [r for r in rows if "error" in r]
    unavailable = [r for r in rows if r.get("numerical_evidence") == "unavailable"]
    if measured:
        overall = round(sum(r["robustness_score"] for r in measured) / len(measured), 1)
    elif errored:
        overall = 0.0
    else:
        overall = None
    return {
        "filename": filename,
        "original": {"ai_probability": base_ai,
                     "manipulation_probability": base_manip,
                     "verdict": base_verdict},
        "transforms": rows,
        "overall_robustness_score": overall,
        "verdict_flips": sum(1 for r in measured if r["verdict_flipped"]),
        "transforms_measured": len(measured),
        "transforms_unavailable": len(unavailable),
        "numerical_evidence": "partial" if unavailable else "available",
        "interpretation": (
            "High score = predictions survive ordinary processing. "
            "Low score = the detector is brittle to everyday edits - "
            "treat its verdicts with extra caution on processed images."
        ),
    }
