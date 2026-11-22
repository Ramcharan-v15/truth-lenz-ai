/**
 * Teacher Demo helpers — TRUTH LENZ AI.
 *
 * Purpose: a polished, teacher-ready upload + preview demonstration that
 * is scientifically honest about what it does NOT do.
 *
 * Hard rules enforced by this module:
 * - No prediction, probability, confidence score, accuracy, or benchmark
 *   is invented here, and none is returned by the teacher demo.
 * - The teacher demo NEVER calls POST /api/analyze, /api/compare,
 *   /api/batch, /api/explain, or /api/robustness. Those endpoints run
 *   the forensic pipeline. The demo only reads GET /api/models, a
 *   metadata endpoint that reports registry status and runs no
 *   inference on any image.
 * - AI-generation detection and manipulation detection are separate
 *   tasks. Evidence of manipulation is NOT evidence of AI generation.
 * - Result labels LIKELY_AI_GENERATED / LIKELY_AUTHENTIC /
 *   LIKELY_MANIPULATED / INCONCLUSIVE exist in the existing API
 *   contract (see types.ts Verdict), but they are RESERVED here:
 *   the teacher demo never emits them, because no authorized,
 *   independently validated, two-class real-world classification path
 *   is available in this task. The only statuses the demo emits are
 *   MODEL_NOT_CONFIGURED (model status read successfully, but no
 *   authorized classification model is available) and
 *   ANALYSIS_UNAVAILABLE (model status could not be read, e.g.
 *   network error or timeout).
 * - A model is authorized for real-world classification in this demo
 *   only if ALL of the following hold: it is ready, it is not a
 *   demonstration/synthetic-only model, and it carries an explicit
 *   authorized-for-real-world-classification flag. No current model
 *   meets those conditions (tl-forensic-stacking is DEMONSTRATION /
 *   SYNTHETIC_ONLY, trained on synthetic procedural renders only),
 *   and Gate 10 is not approved, so the function below correctly
 *   returns false for the current configuration. Changing a lifecycle
 *   label alone must never be treated as authorization.
 */

export type TeacherDemoResultStatus = 'MODEL_NOT_CONFIGURED' | 'ANALYSIS_UNAVAILABLE';

/** Verdict labels from the existing API contract, for legend only. */
export type TeacherDemoVerdictLabel =
  | 'LIKELY_AI_GENERATED'
  | 'LIKELY_AUTHENTIC'
  | 'LIKELY_MANIPULATED'
  | 'INCONCLUSIVE'
  | 'MODEL_NOT_CONFIGURED'
  | 'ANALYSIS_UNAVAILABLE';

export const TEACHER_DEMO_RESULT_LABELS: Record<
  TeacherDemoVerdictLabel,
  { label: string; meaning: string; emittedByDemo: boolean }
> = {
  LIKELY_AI_GENERATED: {
    label: 'Likely AI-Generated',
    meaning: 'Signals are consistent with synthetic image generation. Reserved — not emitted by this demo.',
    emittedByDemo: false,
  },
  LIKELY_AUTHENTIC: {
    label: 'Likely Authentic',
    meaning: 'Signals are consistent with an authentic camera capture. Reserved — not emitted by this demo.',
    emittedByDemo: false,
  },
  LIKELY_MANIPULATED: {
    label: 'Likely Manipulated',
    meaning: 'Signals suggest the image was edited after capture. Manipulation alone is not proof of AI generation. Reserved — not emitted by this demo.',
    emittedByDemo: false,
  },
  INCONCLUSIVE: {
    label: 'Inconclusive',
    meaning: 'Evidence does not support a confident classification. A legitimate abstention, not proof either way. Reserved — not emitted by this demo.',
    emittedByDemo: false,
  },
  MODEL_NOT_CONFIGURED: {
    label: 'Model Not Configured',
    meaning: 'Classification is not currently available: no authorized, validated classification model is configured for arbitrary uploads in this demo.',
    emittedByDemo: true,
  },
  ANALYSIS_UNAVAILABLE: {
    label: 'Analysis Unavailable',
    meaning: 'Classification is not currently available: the model-status service could not be reached or timed out. No classification was made.',
    emittedByDemo: true,
  },
};

import { fmtBytes } from './types';

export const TEACHER_MAX_MB = 25;
export const TEACHER_MAX_BYTES = TEACHER_MAX_MB * 1024 * 1024;
export const TEACHER_ALLOWED_EXTENSIONS = ['jpg', 'jpeg', 'png', 'webp'] as const;
export const TEACHER_ALLOWED_MIME_TYPES = ['image/jpeg', 'image/png', 'image/webp'] as const;

/**
 * Detect an image format from magic bytes only. Never trusts the
 * extension or the claimed MIME type alone. Mirrors the backend
 * magic-byte logic in backend/api/routes.py::_detect_mime, restricted
 * to the teacher demo's JPG / PNG / WebP set (no BMP, no HEIC).
 * Returns null for unrecognized or malformed data.
 */
export function detectTeacherFormat(data: Uint8Array): 'image/jpeg' | 'image/png' | 'image/webp' | null {
  if (data.length >= 3 && data[0] === 0xff && data[1] === 0xd8 && data[2] === 0xff) {
    return 'image/jpeg';
  }
  if (
    data.length >= 8 &&
    data[0] === 0x89 && data[1] === 0x50 && data[2] === 0x4e && data[3] === 0x47 &&
    data[4] === 0x0d && data[5] === 0x0a && data[6] === 0x1a && data[7] === 0x0a
  ) {
    return 'image/png';
  }
  if (
    data.length >= 12 &&
    data[0] === 0x52 && data[1] === 0x49 && data[2] === 0x46 && data[3] === 0x46 &&
    data[8] === 0x57 && data[9] === 0x45 && data[10] === 0x42 && data[11] === 0x50
  ) {
    return 'image/webp';
  }
  return null;
}

/**
 * Validate file metadata only (name, claimed type, size). Returns an
 * error string, or null if metadata passes. Magic-byte and decodability
 * checks run separately in TeacherUpload and must also pass.
 */
export function validateTeacherFileMeta(file: { name: string; type: string; size: number }): string | null {
  if (file.size === 0) return 'This file is empty. Please choose a non-empty image.';
  if (file.size > TEACHER_MAX_BYTES) {
    return `File is ${fmtBytes(file.size)} — the limit is ${TEACHER_MAX_MB} MB. Please choose a smaller file or re-export it at a lower size.`;
  }
  const ext = file.name.split('.').pop()?.toLowerCase() ?? '';
  const extOk = (TEACHER_ALLOWED_EXTENSIONS as readonly string[]).includes(ext);
  const mimeOk = (TEACHER_ALLOWED_MIME_TYPES as readonly string[]).includes(file.type);
  if (!extOk && !mimeOk) {
    return `Unsupported format "${ext || '?'}". Please use JPG, JPEG, PNG, or WebP.`;
  }
  return null;
}

function gcd(a: number, b: number): number {
  return b === 0 ? a : gcd(b, a % b);
}

/**
 * Aspect ratio for the image-information panel only. This is file
 * geometry, not detection evidence. Examples: 64×48 → "4:3",
 * 100×100 → "1:1". Very uneven ratios fall back to a decimal form.
 */
export function formatAspectRatio(width: number, height: number): string {
  if (!width || !height) return 'unavailable';
  const d = gcd(width, height);
  const w = width / d;
  const h = height / d;
  if (w <= 32 && h <= 32) return `${w}:${h}`;
  return `${(width / height).toFixed(2)}:1`;
}

/** Megapixels, to two decimals, for the image-information panel only. */
export function formatMegapixels(width: number, height: number): string {
  if (!width || !height) return 'unavailable';
  return `${((width * height) / 1_000_000).toFixed(2)} MP`;
}

export interface TeacherDemoModelInfo {
  name: string;
  version: string;
  status: string;
  lifecycle_status?: string;
  validation_status?: string;
  limitations?: string;
  demonstration?: boolean;
  authorized_for_real_world_classification?: boolean;
}

/**
 * Fail-safe authorization check. Returns true only if a model is ready,
 * is not demonstration / synthetic-only, and carries an explicit
 * authorized flag. Unknown or missing fields default to NOT authorized.
 * No current registry entry satisfies this, by design.
 */
export function isAuthorizedClassificationModel(model: TeacherDemoModelInfo | null | undefined): boolean {
  if (!model) return false;
  if (model.status !== 'ready') return false;
  if (model.demonstration === true) return false;
  if ((model.lifecycle_status ?? '').toUpperCase() === 'DEMONSTRATION') return false;
  if ((model.validation_status ?? '').toUpperCase() === 'SYNTHETIC_ONLY') return false;
  if (/demonstration|synthetic/i.test(model.limitations ?? '')) return false;
  if (model.authorized_for_real_world_classification !== true) return false;
  return true;
}

/**
 * Teacher demo classification status for a registry response.
 * Always MODEL_NOT_CONFIGURED for the current configuration, because
 * no model passes isAuthorizedClassificationModel. This function never
 * returns a LIKELY_* verdict and never returns a probability.
 */
export function teacherDemoResultStatus(models: TeacherDemoModelInfo[] | null | undefined): TeacherDemoResultStatus {
  const list = models ?? [];
  if (list.some((m) => isAuthorizedClassificationModel(m))) {
    // Even a future authorized model must not be silently used here:
    // activating an inference path requires a separate authorization
    // and a separately approved Gate 10 protocol. Until that exists,
    // the honest demo status remains MODEL_NOT_CONFIGURED.
    return 'MODEL_NOT_CONFIGURED';
  }
  return 'MODEL_NOT_CONFIGURED';
}
