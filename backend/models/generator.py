"""Generator-family classification interface (Phase 7).

Optional: a detector may ALSO report which generator family an image most
resembles. Attribution is only claimed when a model was trained AND evaluated
for that task; otherwise: "Generator attribution unavailable."

Categories:
  camera_photograph | diffusion_generated | gan_generated | ai_edited |
  unknown_synthetic | unknown
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, List, Optional

GENERATOR_LABELS = {
    "camera_photograph": "Camera photograph",
    "diffusion_generated": "Diffusion-generated",
    "gan_generated": "GAN-generated",
    "ai_edited": "AI-edited",
    "stable_diffusion_family": "Stable Diffusion family",
    "midjourney_family": "Midjourney family",
    "dalle_family": "DALL-E family",
    "firefly_family": "Adobe Firefly family",
    "other_diffusion": "Other diffusion",
    "unknown_synthetic": "Unknown synthetic source",
    "unknown": "Unknown",
}


@dataclass
class GeneratorPrediction:
    label: str                          # key from GENERATOR_LABELS
    confidence: float                   # 0-100
    all_scores: Dict[str, float] = field(default_factory=dict)
    evaluated: bool = False             # was this model evaluated for attribution?
    detail: str = ""

    def to_dict(self) -> Dict:
        return {
            "label": self.label,
            "label_display": GENERATOR_LABELS.get(self.label, self.label),
            "confidence": round(self.confidence, 1),
            "all_scores": {k: round(v, 1) for k, v in self.all_scores.items()},
            "evaluated": self.evaluated,
            "detail": self.detail,
        }


class GeneratorClassifier(ABC):
    """Optional interface for generator-family attribution."""

    @abstractmethod
    def classify_generator(self, ctx) -> GeneratorPrediction: ...

    @abstractmethod
    def get_attribution_limitations(self) -> str: ...


class UnavailableGeneratorClassifier(GeneratorClassifier):
    """Honest fallback: attribution not supported."""

    def classify_generator(self, ctx) -> GeneratorPrediction:
        return GeneratorPrediction(
            label="unknown", confidence=0.0, evaluated=False,
            detail="Generator attribution unavailable: no attribution model is configured.")

    def get_attribution_limitations(self) -> str:
        return ("No generator-attribution model is connected. Family-level labels "
                "(Stable Diffusion, Midjourney, DALL-E, ...) require a model trained "
                "and evaluated specifically for attribution.")
