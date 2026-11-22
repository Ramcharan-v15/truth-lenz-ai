export type Verdict = 'LIKELY_AUTHENTIC' | 'LIKELY_AI_GENERATED' | 'LIKELY_MANIPULATED' | 'INCONCLUSIVE';

export interface EvidenceItem {
  text: string;
  direction: 'supports_ai' | 'supports_authentic' | 'supports_manip' | 'neutral' | 'uncertain';
  detail: string;
}

export interface Signal {
  name: string;
  display_name: string;
  ai_score: number;
  manip_score: number;
  confidence: number;
  status: 'ok' | 'warning' | 'unable';
  evidence: EvidenceItem[];
  warnings: string[];
  technical: Record<string, any>;
  duration_ms: number;
  visuals: Record<string, any>;
}

export interface MLDetector {
  detector_name: string;
  detector_version: string;
  model_type: string;
  probability_ai: number | null;
  probability_authentic: number | null;
  raw_score: number | null;
  calibration: string;
  confidence: number;
  supported_formats?: string[];
  preprocessing: Record<string, any>;
  evidence: string[];
  limitations: string;
  inference_time_ms: number;
  model_status: string;
  generator_label: string | null;
  has_attention_heatmap: boolean;
  attention_heatmap?: string;
  detail: string;
  available: boolean;
  /** Stage 2 (S2-1): set when the ensemble withholds this detector's
      output. The raw probability stays recorded as provenance, but the
      entry is not available and must never be shown as calibrated. */
  suppressed?: boolean;
  suppressed_reason?: string;
}

export interface Reproducibility {
  analysis_id: string;
  app_version: string;
  backend_version: string;
  config_version: string;
  config_sha256: string;
  detectors: { name: string; version: string; model_type: string; status: string; calibration: string }[];
  analyzed_at: string;
  image_sha256: string;
  image_dimensions: number[];
  pipeline_ms: number;
  note: string;
}

export interface Explanations {
  basic: string;
  technical: string;
  forensic: string;
  note: string;
}

export interface Ensemble {
  verdict: Verdict;
  verdict_explanation: string;
  verdict_reason: string;
  ai_probability: number | null;
  manipulation_probability: number | null;
  authenticity_probability: number | null;
  confidence: number;
  confidence_label: 'HIGH' | 'MEDIUM' | 'LOW';
  confidence_penalties: string[];
  signal_disagreement: number | null;
  signals_agree: boolean;
  insufficient_evidence: boolean;
  probability_bands: { label: string; max: number }[];
  why: string[];
  strongest_evidence: { analyzer: string; text: string; detail: string }[];
  evidence_against: { analyzer: string; text: string; detail: string }[];
  uncertain_factors: { analyzer: string; text: string; detail: string }[];
  detector: {
    ai_probability: number | null;
    confidence: number;
    model_name: string;
    /** Fallback verdict blocks carry detector_name instead of model_name. */
    detector_name?: string;
    model_version: string;
    limitations: string;
    available: boolean;
    detail: string;
  };
  detectors: MLDetector[];
  ml_available: boolean;
  ml_ai_probability: number | null;
  /** Stage 2 (S2-1): non-null exactly when the ML probability is
      withheld (no usable signals, or thin coverage). Absent on
      reports predating Stage 2, which behave as not suppressed. */
  ml_suppressed_reason?: string | null;
  forensic_ai_probability: number | null;
  provenance: { ai_score: number; confidence: number; note: string };
  ml_disagreement: number;
  ml_ood_status?: string;
  uncertainty: {
    title: string;
    parties: { side: string; ai_probability: number; position: string }[];
    gap: number;
    explanation: string;
  } | null;
  false_positive_risk: {
    level: string;
    score: number;
    reasons: string[];
    note: string;
  };
  hybrid_weights: { forensics: number; ml: number; provenance: number };
  weights_used: Record<string, { ai: number; manip: number }>;
  thresholds_used: Record<string, number>;
  disclaimer: string;
}

export interface AnalysisReport {
  case_id?: string;
  analysis_id: string;
  generator_attribution?: {
    status: string;
    label: string | null;
    confidence: number;
    detail: string;
  };
  filename: string;
  mime_type: string;
  file_size: number;
  width: number;
  height: number;
  preview: string;
  analyzed_at: string;
  pipeline_ms: number;
  stages: { stage: string; display: string; status: string; duration_ms: number }[];
  signals: Signal[];
  ensemble: Ensemble;
  educational: Record<string, { what: string; why: string; limits: string }>;
  explanations: Explanations;
  reproducibility: Reproducibility;
  report_hash: string;
  limitations: string[];
  saved_to_history?: boolean;
  demo?: boolean;
}

export const VERDICT_META: Record<Verdict, { label: string; color: string; icon: string; blurb: string }> = {
  LIKELY_AUTHENTIC: {
    label: 'Likely Authentic',
    color: '#22c55e',
    icon: '◈',
    blurb: 'Signals are consistent with an authentic camera capture.',
  },
  LIKELY_AI_GENERATED: {
    label: 'Likely AI-Generated',
    color: '#f43f5e',
    icon: '⬢',
    blurb: 'Multiple signals are consistent with synthetic image generation.',
  },
  LIKELY_MANIPULATED: {
    label: 'Likely Manipulated',
    color: '#f59e0b',
    icon: '⬣',
    blurb: 'Signals suggest the image was edited after capture.',
  },
  INCONCLUSIVE: {
    label: 'Inconclusive',
    color: '#94a3b8',
    icon: '◌',
    blurb: 'Evidence does not support a confident classification.',
  },
};

export function fmtBytes(n: number): string {
  if (n < 1024) return `${n} B`;
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`;
  return `${(n / 1024 / 1024).toFixed(2)} MB`;
}

/**
 * Stage 1 null-safety helpers (V5.1.8.67): a measurement is only rendered as a
 * number when it is a finite number. Anything else — null, undefined, NaN,
 * Infinity — renders as 'unavailable'. Missing values are NEVER coerced to
 * zero: 0 is a real measurement, null is the absence of one.
 */
export function isFiniteNumber(v: unknown): v is number {
  return typeof v === 'number' && Number.isFinite(v);
}

/** Format a 0–100 measurement as a percentage, or 'unavailable'. */
export function fmtPct(v: unknown, digits = 0): string {
  return isFiniteNumber(v) ? `${v.toFixed(digits)}%` : 'unavailable';
}

/** Format a 0–1 fraction as a percentage, or 'unavailable'. */
export function fmtFractionPct(v: unknown, digits = 2): string {
  return isFiniteNumber(v) ? `${(v * 100).toFixed(digits)}%` : 'unavailable';
}

/** Format a plain number, or 'unavailable'. */
export function fmtNum(v: unknown, digits = 0): string {
  return isFiniteNumber(v) ? v.toFixed(digits) : 'unavailable';
}
