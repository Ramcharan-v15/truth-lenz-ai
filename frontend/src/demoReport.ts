import type { AnalysisReport, Signal } from './types';

/**
 * DEMO RESULT - NOT REAL ANALYSIS.
 * Hand-crafted sample output for UI testing without running detection.
 * Never mixed with real analysis results.
 */

function sig(name: string, display: string, ai: number, manip: number, conf: number, evidence: [string, string][]): Signal {
  return {
    name, display_name: display, ai_score: ai, manip_score: manip, confidence: conf,
    status: 'ok', duration_ms: 120,
    evidence: evidence.map(([text, direction]) => ({ text, direction: direction as any, detail: '' })),
    warnings: [], technical: { note: 'demo data' }, visuals: {},
  };
}

const signals: Signal[] = [
  sig('metadata', 'Metadata Forensics', 82, 50, 88, [
    ["AI generator signature found in metadata: 'stable diffusion'", 'supports_ai'],
    ['No editing-software tag detected', 'neutral'],
  ]),
  sig('statistics', 'Image Statistics', 52, 50, 85, [
    ['Luminance entropy 6.91 bits (max 8.0)', 'neutral'],
  ]),
  sig('compression', 'Compression Analysis', 56, 50, 80, [
    ['Lossy WEBP encoding is common in AI-generator and web-pipeline outputs', 'supports_ai'],
  ]),
  sig('noise', 'Noise Analysis', 74, 50, 75, [
    ['Implausibly low noise in flat regions', 'supports_ai'],
    ['Noise inconsistent across flat regions', 'supports_manip'],
  ]),
  sig('frequency', 'Frequency Analysis', 76, 50, 70, [
    ['9 abnormal spectral peaks detected', 'supports_ai'],
    ['Abnormally steep spectral falloff (slope -7.42)', 'supports_ai'],
  ]),
  sig('pixel', 'Pixel Forensics', 66, 45, 70, [
    ['Periodic micro-texture detected', 'supports_ai'],
    ['ELA response is uniform across the frame', 'supports_authentic'],
  ]),
  sig('ai_artifacts', 'AI Artifact Analysis', 72, 50, 60, [
    ['High micro-pattern repetition (31.2% duplicate blocks)', 'supports_ai'],
  ]),
  sig('manipulation', 'Manipulation Detection', 50, 44, 65, [
    ['No copy-move duplication pattern found', 'supports_authentic'],
  ]),
  sig('quality', 'Image Quality Assessment', 58, 50, 70, [
    ['Dimensions 1024x1024 match common AI-generator output sizes', 'supports_ai'],
    ['Image quality: Good (1024x1024)', 'neutral'],
  ]),
  sig('consistency', 'Consistency Check', 50, 50, 70, [
    ['No metadata to check for consistency', 'neutral'],
  ]),
];
// attach demo visuals
signals[4].visuals = { radial_profile: { frequencies: [0, 0.5, 1], power: [0.9, 0.08, 0.02] } };
signals[0].technical = { format: 'WEBP', exif_tag_count: 0, info_keys: ['Software'], size: [1024, 1024] };

export const DEMO_REPORT: AnalysisReport = {
  case_id: 'TL-2026-DEMO01',
  analysis_id: 'demo00000001',
  generator_attribution: {
    status: 'ATTRIBUTION_UNAVAILABLE',
    label: null,
    confidence: 0,
    detail: 'Demo data: no attribution model.',
  },
  filename: 'demo_portrait.webp',
  mime_type: 'image/webp',
  file_size: 184320,
  width: 1024,
  height: 1024,
  preview:
    'data:image/svg+xml,' +
    encodeURIComponent(
      `<svg xmlns="http://www.w3.org/2000/svg" width="512" height="512"><rect width="512" height="512" fill="#101623"/><circle cx="256" cy="200" r="90" fill="#1e293b"/><rect x="156" y="310" width="200" height="140" rx="20" fill="#1e293b"/><text x="256" y="480" text-anchor="middle" fill="#38bdf8" font-size="22" font-family="sans-serif">DEMO IMAGE PLACEHOLDER</text></svg>`,
    ),
  analyzed_at: '2026-10-06T12:00:00',
  pipeline_ms: 1840,
  stages: signals.map((s) => ({ stage: s.name, display: s.display_name, status: 'ok', duration_ms: s.duration_ms })),
  signals,
  ensemble: {
    verdict: 'LIKELY_AI_GENERATED',
    verdict_explanation: 'Multiple forensic signals are consistent with synthetic image generation. The assessment weighs metadata, noise, frequency, texture and artifact signals together. Evidence is probabilistic, not definitive.',
    verdict_reason: 'Decisive signal from Metadata Forensics (82/100, 88% confidence) with no strong contradiction; ensemble AI probability 78%.',
    ai_probability: 78.4,
    manipulation_probability: 32.1,
    authenticity_probability: 28.6,
    confidence: 68,
    confidence_label: 'MEDIUM',
    confidence_penalties: ['No embedded metadata for provenance context'],
    signal_disagreement: 9.2,
    signals_agree: true,
    insufficient_evidence: false,
    probability_bands: [
      { label: 'Strongly likely authentic', max: 20 },
      { label: 'Probably authentic', max: 40 },
      { label: 'Inconclusive', max: 60 },
      { label: 'Probably AI-generated', max: 80 },
      { label: 'Strongly likely AI-generated', max: 100 },
    ],
    why: [
      'Metadata Forensics: scored 82/100 with 88% signal confidence',
      'Frequency Analysis: scored 76/100 with 70% signal confidence',
      'Noise Analysis: scored 74/100 with 75% signal confidence',
    ],
    strongest_evidence: [
      { analyzer: 'Metadata Forensics', text: "AI generator signature found in metadata: 'stable diffusion'", detail: '' },
      { analyzer: 'Frequency Analysis', text: '9 abnormal spectral peaks detected', detail: '' },
      { analyzer: 'Noise Analysis', text: 'Implausibly low noise in flat regions', detail: '' },
    ],
    evidence_against: [
      { analyzer: 'Pixel Forensics', text: 'ELA response is uniform across the frame', detail: '' },
      { analyzer: 'Manipulation Detection', text: 'No copy-move duplication pattern found', detail: '' },
    ],
    uncertain_factors: [
      { analyzer: 'Compression', text: 'Lossy WEBP encoding is common in AI-generator and web-pipeline outputs', detail: '' },
    ],
    detector: {
      ai_probability: null, confidence: 0, model_name: 'No trained model configured',
      model_version: 'n/a', limitations: 'Demo data.', available: false, detail: 'demo',
    },
    detectors: [{
      detector_name: 'No trained model configured', detector_version: 'n/a', model_type: 'none',
      probability_ai: null, probability_authentic: null, raw_score: null, calibration: 'none (uncalibrated)',
      confidence: 0, preprocessing: {}, evidence: [], limitations: 'Demo data.',
      inference_time_ms: 0, model_status: 'not_configured', generator_label: null,
      has_attention_heatmap: false, detail: 'demo', available: false,
    }],
    ml_available: false,
    ml_ai_probability: null,
    forensic_ai_probability: 78.4,
    provenance: { ai_score: 85, confidence: 80, note: 'provenance names an AI generator (demo)' },
    ml_disagreement: 0,
    uncertainty: null,
    false_positive_risk: { level: 'LOW', score: 0, reasons: [], note: 'Demo data.' },
    hybrid_weights: { forensics: 1.0, ml: 0, provenance: 0.6 },
    weights_used: {},
    thresholds_used: { ai_likely: 60, manip_likely: 60 },
    disclaimer: 'DEMO DATA. Probability estimates, not definitive proof.',
  },
  educational: {
    metadata: { what: 'Demo explainer.', why: 'Demo explainer.', limits: 'Demo explainer.' },
  },
  explanations: {
    basic: 'DEMO: This image shows several signals associated with AI-generated imagery.',
    technical: 'DEMO: Verdict LIKELY_AI_GENERATED (AI 78.4%, confidence MEDIUM 68%).',
    forensic: 'DEMO: hand-crafted sample output for UI testing.',
    note: 'DEMO DATA - not a real analysis.',
  },
  reproducibility: {
    analysis_id: 'demo00000001', app_version: '2.0.0-demo', backend_version: '2.0.0-demo',
    config_version: 'demo', config_sha256: 'demo', detectors: [],
    analyzed_at: '2026-10-06T12:00:00', image_sha256: 'demo',
    image_dimensions: [1024, 1024], pipeline_ms: 1840,
    note: 'DEMO DATA - not reproducible.',
  },
  report_hash: 'demo',
  limitations: [
    'DEMO RESULT — this output was hand-crafted for UI testing and is not a real analysis.',
    'No detector can guarantee whether an image is AI-generated or authentic.',
  ],
  demo: true,
};
