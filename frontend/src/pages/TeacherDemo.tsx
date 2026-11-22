import { useRef, useState } from 'react';
import TeacherUpload, { type TeacherPickedFile } from '../components/TeacherUpload';
import { getModelStatus } from '../api';
import { fmtBytes } from '../types';
import {
  formatAspectRatio,
  formatMegapixels,
  TEACHER_DEMO_RESULT_LABELS,
  teacherDemoResultStatus,
  type TeacherDemoModelInfo,
  type TeacherDemoResultStatus,
} from '../teacherDemo';

/**
 * Teacher Demo page — TRUTH LENZ AI.
 *
 * What this page DOES: upload + local preview, strict validation
 * (JPG/JPEG/PNG/WebP, 25 MB, magic bytes, decodability), a clear
 * status section, and an honest result panel.
 *
 * What this page does NOT do, deliberately:
 * - It never sends the selected image to the server. Preview is
 *   local (object URL) only. No upload, no storage, no third party.
 * - It never calls POST /api/analyze, /api/compare, /api/batch,
 *   /api/explain, or /api/robustness. It only reads GET /api/models,
 *   a metadata endpoint that runs no inference on any image.
 * - It never emits a LIKELY_* verdict, a probability, or a confidence
 *   score. The only results it emits are MODEL_NOT_CONFIGURED
 *   (model status read, but no authorized classification model) and
 *   ANALYSIS_UNAVAILABLE (model status could not be read).
 * - It never presents a successful UI test as proof of scientific
 *   accuracy. Validation status is shown as incomplete, always.
 */
export default function TeacherDemo() {
  const [picked, setPicked] = useState<TeacherPickedFile | null>(null);
  const [resetKey, setResetKey] = useState(0);
  const [phase, setPhase] = useState<'idle' | 'checking' | 'done' | 'error'>('idle');
  const [resultStatus, setResultStatus] = useState<TeacherDemoResultStatus | null>(null);
  const [modelLine, setModelLine] = useState<string | null>(null);
  const [apiError, setApiError] = useState<string | null>(null);
  const abortRef = useRef(false);
  const controllerRef = useRef<AbortController | null>(null);

  const reset = () => {
    abortRef.current = true;
    if (controllerRef.current) controllerRef.current.abort();
    if (picked) URL.revokeObjectURL(picked.previewUrl);
    setPicked(null);
    setResultStatus(null);
    setModelLine(null);
    setApiError(null);
    setPhase('idle');
    setResetKey((k) => k + 1);
  };

  const runAnalyze = async () => {
    if (!picked) {
      setApiError('Select an image first.');
      setPhase('error');
      return;
    }
    abortRef.current = false;
    setApiError(null);
    setResultStatus(null);
    setModelLine(null);
    setPhase('checking');
    const controller = new AbortController();
    controllerRef.current = controller;
    const timer = window.setTimeout(() => controller.abort(), 10000);
    try {
      const reg = await getModelStatus(controller.signal);
      window.clearTimeout(timer);
      if (abortRef.current) return;
      const models: TeacherDemoModelInfo[] = (reg.models ?? []).map((m) => ({
        name: m.name,
        version: m.version,
        status: m.status,
        lifecycle_status: m.lifecycle_status,
        validation_status: m.validation_status,
        limitations: m.limitations,
        demonstration: m.demonstration ?? m.config?.demonstration,
        authorized_for_real_world_classification: m.config?.authorized_for_real_world_classification,
      }));
      const status = teacherDemoResultStatus(models);
      setResultStatus(status);
      if (models.length === 0) {
        setModelLine('No model configured.');
      } else {
        setModelLine(
          models
            .map((m) => `${m.name} v${m.version} — ${(m.lifecycle_status ?? m.status ?? 'unknown').toUpperCase()}${m.validation_status ? ` / ${String(m.validation_status).toUpperCase()}` : ''}`)
            .join('; '),
        );
      }
      setPhase('done');
    } catch {
      window.clearTimeout(timer);
      if (abortRef.current) return;
      setResultStatus('ANALYSIS_UNAVAILABLE');
      setApiError('Could not reach the model-status service. No classification was made.');
      setModelLine('Model status unavailable.');
      setPhase('done');
    }
  };

  return (
    <div className="shell" style={{ paddingTop: 30, paddingBottom: 20 }} data-testid="teacher-demo-page">
      <div className="eyebrow">TEACHER DEMONSTRATION</div>
      <h1 style={{ margin: '0 0 8px', fontSize: 30 }}>TRUTH LENZ AI</h1>
      <p className="section-sub">
        Upload an image, preview it, and see an honest, evidence-based status.
        This demonstration shows upload and preview working — it does not claim
        that the image has been classified, because no authorized, validated
        classification model is currently available for arbitrary uploads.
      </p>

      <div className="panel panel-pad" data-testid="teacher-status" style={{ marginBottom: 20 }}>
        <div className="eyebrow">Current status — read this first</div>
        <div className="grid grid-2" style={{ marginTop: 12 }}>
          <div>
            <div style={{ fontWeight: 700, fontSize: 14 }}>Image upload and preview</div>
            <div className="dim" style={{ fontSize: 13, marginTop: 4 }}>
              Available in this demonstration. Preview happens in your browser only; the image is not sent to the server, not stored, and not sent to any third party.
            </div>
          </div>
          <div>
            <div style={{ fontWeight: 700, fontSize: 14 }}>AI-generation detection</div>
            <div className="dim" style={{ fontSize: 13, marginTop: 4 }}>
              Not yet validated on representative real-world data. Do not treat any output from this demo as a real-world AI-detection result.
            </div>
          </div>
          <div>
            <div style={{ fontWeight: 700, fontSize: 14 }}>Manipulation detection</div>
            <div className="dim" style={{ fontSize: 13, marginTop: 4 }}>
              A separate task from AI-generation detection. Evidence that an image was edited is not, by itself, evidence that it was AI-generated — and vice versa.
            </div>
          </div>
          <div>
            <div style={{ fontWeight: 700, fontSize: 14 }}>Validation status</div>
            <div className="dim" style={{ fontSize: 13, marginTop: 4 }}>
              <strong>VALIDATION INCOMPLETE — REAL-WORLD PERFORMANCE NOT MEASURED.</strong> A successful upload or preview test is not proof of scientific accuracy.
            </div>
          </div>
        </div>
        <div className="dim" style={{ fontSize: 13, marginTop: 14 }}>
          <strong>Current model status:</strong> {modelLine ?? 'Not checked yet — click Analyze Image after choosing a file to read the current model registry status. The only model currently registered is a synthetic demonstration model, not a validated real-world classifier.'}
        </div>
      </div>

      <TeacherUpload
        key={resetKey}
        onPick={(p) => {
          setPicked(p);
          if (!p) {
            setResultStatus(null);
            setApiError(null);
            setPhase('idle');
          } else {
            setResultStatus(null);
            setPhase('idle');
          }
        }}
      />

      {picked && phase !== 'done' && (
        <div style={{ display: 'flex', gap: 12, marginTop: 14, alignItems: 'center', flexWrap: 'wrap' }}>
          <button
            className="btn btn-primary"
            onClick={runAnalyze}
            disabled={phase === 'checking'}
            aria-label="Analyze image"
            data-testid="teacher-analyze-button"
          >
            {phase === 'checking' ? 'Checking model availability…' : '🔍 Analyze Image'}
          </button>
          <button className="btn btn-ghost" onClick={reset} data-testid="teacher-select-another">
            Select another image
          </button>
          <span className="dim" style={{ fontSize: 12 }}>
            Analyze Image checks model availability only. It does not run a detector on your image in this demo.
          </span>
        </div>
      )}

      {picked && (
        <div className="panel panel-pad" style={{ marginTop: 14 }} data-testid="teacher-image-info">
          <div className="eyebrow">Image information — not an AI verdict</div>
          <div className="grid grid-2" style={{ marginTop: 12, fontSize: 13 }}>
            <div><strong>Filename:</strong> {picked.file.name}</div>
            <div><strong>Dimensions:</strong> {picked.dims}</div>
            <div><strong>Aspect ratio:</strong> {formatAspectRatio(picked.width, picked.height)}</div>
            <div><strong>Size:</strong> {formatMegapixels(picked.width, picked.height)} · {fmtBytes(picked.file.size)}</div>
            <div><strong>Format (detected):</strong> {picked.detectedFormat}</div>
            <div><strong>Sent to server:</strong> No — preview only</div>
          </div>
          <p className="dim" style={{ fontSize: 12.5, margin: '12px 0 0' }}>
            This panel describes the file itself — its size, shape, and format. It does not say whether the image is AI-generated or manipulated, and it must not be read as a verdict. Embedded metadata such as EXIF, GPS, camera, or author information is not read or displayed by this demo, to protect privacy.
          </p>
        </div>
      )}

      {phase === 'checking' && (
        <div className="panel panel-pad" style={{ marginTop: 14 }} role="status" aria-live="polite">
          <div className="eyebrow">Checking model availability…</div>
          <div className="faint" style={{ fontSize: 13, marginTop: 6 }}>
            Reading the model registry status. No image is being uploaded and no classification is being run.
          </div>
        </div>
      )}

      {phase === 'error' && apiError && (
        <div className="alert error" role="alert" style={{ marginTop: 14 }}>
          {apiError}
        </div>
      )}

      {phase === 'done' && resultStatus && (
        <div data-testid="teacher-result" style={{ marginTop: 14 }}>
          <div className="panel panel-pad">
            <div className="eyebrow">Result</div>
            <div style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap', marginTop: 8 }}>
              <span className="tag unknown" data-testid="teacher-result-status">{resultStatus}</span>
              <strong style={{ fontSize: 18 }}>{TEACHER_DEMO_RESULT_LABELS[resultStatus].label}</strong>
            </div>
            <p style={{ fontSize: 14, margin: '12px 0 0' }}>
              {resultStatus === 'MODEL_NOT_CONFIGURED'
                ? 'Classification is not currently available for this image. No authorized, validated classification model is configured for arbitrary uploads, so no AI-generation or manipulation verdict can be given here. This is an honest unavailable state — not a guess, not a default authentic result, and not a hidden prediction.'
                : 'Classification is not currently available because the model-status service could not be reached or timed out. No classification was made, and no result should be inferred from this failure.'}
            </p>
            <p className="dim" style={{ fontSize: 13, margin: '12px 0 0' }}>
              <strong>Model status:</strong> {modelLine ?? 'Unavailable.'}
            </p>
            <p className="dim" style={{ fontSize: 13, margin: '8px 0 0' }}>
              <strong>No probability or confidence score is shown</strong>, because no classification was made. A number shown here would be invented, and invented numbers are not evidence.
            </p>
            <p className="dim" style={{ fontSize: 13, margin: '8px 0 0' }}>
              <strong>Disclaimer:</strong> Do not use this demonstration as the sole basis for legal, academic, employment, financial, or other high-stakes decisions. AI image detection is probabilistic even when validated, and this demo is not validated for real-world use.
            </p>
            <div style={{ display: 'flex', gap: 10, marginTop: 16, flexWrap: 'wrap' }}>
              <button className="btn btn-ghost" onClick={reset} data-testid="teacher-select-another-result">
                Select another image
              </button>
            </div>
          </div>

          <div className="panel panel-pad" style={{ marginTop: 16 }}>
            <div className="eyebrow">Result states — what each one means</div>
            <div style={{ fontSize: 13, marginTop: 10 }}>
              {(Object.keys(TEACHER_DEMO_RESULT_LABELS) as Array<keyof typeof TEACHER_DEMO_RESULT_LABELS>).map((k) => (
                <div key={k} style={{ marginBottom: 8 }}>
                  <code>{k}</code> — {TEACHER_DEMO_RESULT_LABELS[k].meaning}{' '}
                  {TEACHER_DEMO_RESULT_LABELS[k].emittedByDemo ? (
                    <span className="tag present">Emitted by this demo</span>
                  ) : (
                    <span className="tag absent">Reserved — not emitted by this demo</span>
                  )}
                </div>
              ))}
            </div>
          </div>
        </div>
      )}

      <div className="panel panel-pad" style={{ marginTop: 20 }} data-testid="teacher-ai-vs-manipulation">
        <div className="eyebrow">Two different questions</div>
        <h2 style={{ fontSize: 18, margin: '8px 0' }}>AI generation vs manipulation — two different questions</h2>
        <div className="grid grid-2" style={{ fontSize: 13.5 }}>
          <div>
            <strong>AI-generation detection</strong>
            <p style={{ margin: '6px 0 0' }}>
              Asks: was this whole image created by an AI generator, rather than captured by a camera? Example: a portrait that never existed as a photograph. A positive answer needs evidence about generation, not just about editing.
            </p>
          </div>
          <div>
            <strong>Manipulation detection</strong>
            <p style={{ margin: '6px 0 0' }}>
              Asks: was a real image edited after capture — for example, spliced, retouched, or recombined? Example: a real photo with a person moved or removed. Manipulation evidence is not, by itself, evidence of AI generation, and AI-generation evidence is not, by itself, proof of editing.
            </p>
          </div>
        </div>
        <p className="dim" style={{ fontSize: 12.5, margin: '12px 0 0' }}>
          This demo gives no verdict on either question. Keeping them separate is what stops a teacher, a reviewer, or a user from turning one kind of finding into the other kind of claim.
        </p>
      </div>

      <div className="panel panel-pad" style={{ marginTop: 20 }} data-testid="teacher-how-to-interpret">
        <div className="eyebrow">How to Interpret Results</div>
        <h2 style={{ fontSize: 18, margin: '8px 0' }}>How to Interpret Results</h2>
        <p style={{ fontSize: 13.5, margin: '0 0 10px' }}>
          Even a validated detector can be wrong. These three ideas apply to every image result, in plain language:
        </p>
        <div style={{ fontSize: 13.5 }}>
          <p style={{ margin: '0 0 10px' }}>
            <strong>False positive — a real image flagged as AI-generated.</strong> This can happen when a real photo is heavily compressed, resized, screenshotted, denoised, sharpened, or colour-graded. Those changes alter the fine signals detectors look at, so a real photo can start to look synthetic. A flag is a reason to check further, not proof.
          </p>
          <p style={{ margin: '0 0 10px' }}>
            <strong>False negative — an AI-generated image that is missed.</strong> Generators keep improving, and new generators can produce signals a detector has never seen. A clean result does not prove an image is authentic. Absence of a detection is not evidence of authenticity.
          </p>
          <p style={{ margin: '0 0 10px' }}>
            <strong>Inconclusive — the evidence is not enough to decide.</strong> When signals disagree, are weak, or are missing (for example, metadata was stripped by a messaging app), a careful system abstains instead of guessing. Inconclusive means “we cannot tell from this evidence”, not “probably real” and not “probably AI”.
          </p>
          <p style={{ margin: 0 }}>
            <strong>Manipulation is not the same as AI generation.</strong> An edited real photo (for example, spliced or retouched) and a fully AI-generated photo are different things. One kind of evidence must not be relabelled as the other.
          </p>
        </div>
      </div>

      <div className="panel panel-pad" style={{ marginTop: 16 }}>
        <div className="eyebrow">What would be needed for genuine classification</div>
        <p className="dim" style={{ fontSize: 13, margin: 0 }}>
          Gate 8 human review completed and verified, an independent eligibility approver assigned (A7 resolved), registration completed under an approved Gate 9 protocol, and a two-class evaluation completed under an approved Gate 10 protocol with independently sourced real and AI-generated classes, frozen configuration, predeclared metrics, uncertainty reporting, and independent review. None of those conditions are satisfied in this demo, which is why classification is shown as unavailable rather than faked.
        </p>
      </div>
    </div>
  );
}
