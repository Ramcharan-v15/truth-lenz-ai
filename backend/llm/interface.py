"""LLM explanation layer interface (Phase 21).

HARD RULES (enforced by architecture, not just documentation):

1. The LLM NEVER calculates forensic scores. The deterministic engine computes
   scores, probabilities, confidence and evidence. The LLM only translates
   STRUCTURED evidence into prose.
2. The LLM NEVER invents evidence. Input is the serialized evidence packet
   (backend/utils/security.py::serialize_evidence_for_llm); output is validated
   with validate_llm_output, which flags any number not present in the packet.
3. If the LLM layer is not configured, /api/explain returns 501 with an
   explicit "not configured" message. Nothing is faked.

No LLM is bundled with Truth Lenz AI. To enable, implement LLMExplainer and
register it in backend/llm/__init__.py.
"""
from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any, Dict, List

from ..utils.security import serialize_evidence_for_llm, validate_llm_output


class LLMExplainer(ABC):
    """Translate structured forensic evidence into natural language."""

    @abstractmethod
    def get_provider_name(self) -> str: ...

    @abstractmethod
    def explain(self, structured_report: Dict[str, Any],
                level: str = "basic") -> Dict[str, Any]:
        """Return {"text": ..., "warnings": [...]}. Must not alter numbers."""
        ...


class DisabledLLMExplainer(LLMExplainer):
    """Default: the LLM layer is not configured."""

    def get_provider_name(self) -> str:
        return "not configured"

    def explain(self, structured_report: Dict[str, Any],
                level: str = "basic") -> Dict[str, Any]:
        raise RuntimeError(
            "LLM explanation layer is not configured. Deterministic explanations "
            "are available in the report's 'explanations' field (basic/technical/forensic).")


def get_explainer() -> LLMExplainer:
    # Future: read config/env to select a real provider. Today: disabled.
    return DisabledLLMExplainer()


def build_evidence_packet(structured_report: Dict[str, Any]) -> str:
    """Injection-resistant serialization for a future LLM provider."""
    return serialize_evidence_for_llm(structured_report)


def check_explanation(text: str, structured_report: Dict[str, Any]) -> List[str]:
    """Validate an LLM explanation against computed evidence."""
    return validate_llm_output(text, structured_report)
