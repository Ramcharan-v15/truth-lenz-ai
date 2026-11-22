import { useState } from 'react';
import UploadZone, { type PickedFile } from '../components/UploadZone';
import { compareImages } from '../api';
import { VERDICT_META, fmtBytes } from '../types';

interface LabResult {
  image_a: any; // REAL
  image_b: any; // AI candidate
  comparison: any;
}

function sig(report: any, name: string) {
  return report?.signals?.find((s: any) => s.name === name);
}

function verdictChip(v: string) {
  const m = VERDICT_META[v as keyof typeof VERDICT_META];
  if (!m) return <span className="tag">{v}</span>;
  return <span className="tag" style={{ color: m.color, borderColor: m.color + '66' }}>{m.label}</span>;
}

function Slot({
  title, hint, picked, onPick, onClear,
}: {
  title: string; hint: string; picked: PickedFile | null;
  onPick: (f: PickedFile) => void; onClear: () => void;
}) {
  return (
    <div>
      <div className="eyebrow">{title}</div>
      <div className="faint" style={{ fontSize: 12, marginBottom: 8 }}>{hint}</div>
      {!picked ? <UploadZone compact onPick={(f) => onPick(f[0])} /> : (
        <div className="file-row">
          <img className="thumb" src={picked.previewUrl} alt={title} />
          <div style={{ flex: 1 }}>
            <div style={{ fontWeight: 600, fontSize: 13.5 }}>{picked.file.name}</div>
            <div className="faint mono" style={{ fontSize: 11.5 }}>{fmtBytes(picked.file.size)} · {picked.dims}</div>
          </div>
          <button className="btn btn-ghost btn-sm" onClick={onClear} aria-label={`Remove ${title}`}>✕</button>
        </div>
      )}
    </div>
  );
}

function VerdictCard({ report, label }: { report: any; label: string }) {
  const e = report.ensemble;
  return (
    <div className="card" style={{ textAlign: 'center' }}>
      <div className="eyebrow">{label}</div>
      <img src={report.preview} alt={label} style={{ maxHeight: 220, borderRadius: 8, marginTop: 8 }} />
      <div style={{ marginTop: 8, fontWeight: 600, fontSize: 13 }}>{report.filename}</div>
      <div style={{ marginTop: 6 }}>{verdictChip(e.verdict)}</div>
      <div className="mono faint" style={{ fontSize: 12, marginTop: 4 }}>
        AI {Number(e.ai_probability).toFixed(1)}% · Manip {Number(e.manipulation_probability).toFixed(1)}%
      </div>
      <div className="faint" style={{ fontSize: 12, marginTop: 2 }}>
        Confidence: <b style={{ color: 'var(--text)' }}>{e.confidence_label}</b> ({Number(e.confidence).toFixed(0)}%)
      </div>
    </div>
  );
}

function Spotlight({ report, label }: { report: any; label: string }) {
  const noise = sig(report, 'noise')?.technical ?? {};
  const freq = sig(report, 'frequency')?.technical ?? {};
  const comp = sig(report, 'compression')?.technical ?? {};
  const md = sig(report, 'metadata') ?? {};
  const mdTech = md.technical ?? {};
  const aiSig = (md.evidence ?? []).some((ev: any) => /ai generator signature/i.test(ev.text));
  const camMake = mdTech.exif_summary?.Make;
  const camModel = mdTech.exif_summary?.Model;
  const c2pa = mdTech.c2pa_status ?? 'missing';
  const noiseStd = noise.global_noise_std ?? noise.noise_std;
  const rows: [string, string][] = [
    ['Noise (flat-region σ)', noiseStd !== undefined ? Number(noiseStd).toFixed(4) : 'n/a'],
    ['Spectral slope', freq.spectral_slope !== undefined ? String(freq.spectral_slope) : 'n/a'],
    ['Spectral peaks', freq.spectral_peak_count !== undefined ? String(freq.spectral_peak_count) : 'n/a'],
    ['Est. JPEG quality', comp.estimated_jpeg_quality !== undefined ? String(comp.estimated_jpeg_quality) : 'n/a'],
    ['AI signature in metadata', aiSig ? 'YES — generator tag found' : 'none found'],
    ['Camera EXIF', camMake || camModel ? `${camMake ?? ''} ${camModel ?? ''}`.trim() : 'none'],
    ['C2PA status', String(c2pa).toUpperCase()],
  ];
  return (
    <div className="panel panel-pad">
      <div className="eyebrow">{label} — forensic spotlight</div>
      <dl className="kv" style={{ marginTop: 8 }}>
        {rows.map(([k, v]) => (
          <div style={{ display: 'contents' }} key={k}><dt>{k}</dt><dd className="mono" style={{ fontSize: 12.5 }}>{v}</dd></div>
        ))}
      </dl>
    </div>
  );
}

export default function AiRealLab() {
  const [real, setReal] = useState<PickedFile | null>(null);
  const [ai, setAi] = useState<PickedFile | null>(null);
  const [result, setResult] = useState<LabResult | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    if (!real || !ai) return;
    setLoading(true); setError(null);
    try {
      // A = real photo, B = AI candidate. Real analysis, no staged scores.
      const r = await compareImages(real.file, ai.file);
      setResult(r);
    } catch (e: any) {
      setError(e.message);
    } finally {
      setLoading(false);
    }
  };

  const clear = () => { setReal(null); setAi(null); setResult(null); setError(null); };

  const isDemo = !!(result && (result.image_a?.demo || result.image_b?.demo));

  // delta: AI candidate minus real photo, on the AI-probability axis
  const aiDelta = result
    ? Number(result.image_b.ensemble.ai_probability) - Number(result.image_a.ensemble.ai_probability)
    : 0;
  const manipDelta = result
    ? Number(result.image_b.ensemble.manipulation_probability) - Number(result.image_a.ensemble.manipulation_probability)
    : 0;

  const deltaReading = !result ? '' : aiDelta >= 10
    ? `The AI candidate scores ${aiDelta.toFixed(1)} points higher on AI probability — the expected direction for this pair.`
    : aiDelta <= -10
      ? `Surprisingly, the real photo scored ${Math.abs(aiDelta).toFixed(1)} points higher on AI probability. Estimates are fallible — editing, compression or unusual content can push a genuine photo toward the synthetic side.`
      : `Only ${Math.abs(aiDelta).toFixed(1)} points separate the two images on AI probability — too close to read much into.`;

  return (
    <div className="shell" style={{ paddingTop: 30, paddingBottom: 20 }}>
      <div className="eyebrow">CONTROLLED COMPARISON</div>
      <h1 style={{ margin: '0 0 8px', fontSize: 30 }}>AI vs REAL Lab</h1>
      <p className="section-sub">
        Upload a photo you trust and a synthetic image. Both run through the full Truth Lenz
        pipeline — real analysis, no staged scores.
      </p>

      {isDemo && (
        <div className="demo-banner" role="alert" style={{ marginBottom: 16 }}>
          ⚠ DEMO RESULT — NOT REAL ANALYSIS. This output is sample data for UI testing only.
        </div>
      )}

      <div className="grid grid-2">
        <Slot
          title="📷 REAL — a genuine camera photograph"
          hint="Something you shot yourself, or a photo whose origin you trust."
          picked={real}
          onPick={(f) => { setReal(f); setResult(null); }}
          onClear={() => { setReal(null); setResult(null); }}
        />
        <Slot
          title="🤖 AI — a synthetic / AI-generated image"
          hint="An image you believe was generated or heavily synthesized by AI."
          picked={ai}
          onPick={(f) => { setAi(f); setResult(null); }}
          onClear={() => { setAi(null); setResult(null); }}
        />
      </div>

      <div style={{ marginTop: 16, display: 'flex', gap: 10 }}>
        <button className="btn btn-primary" disabled={!real || !ai || loading} onClick={run}>
          {loading ? 'Analyzing both images…' : 'Run both through Truth Lenz'}
        </button>
        {(real || ai) && (
          <button className="btn btn-ghost" onClick={clear} disabled={loading}>Clear</button>
        )}
      </div>
      {error && <div className="alert error" style={{ marginTop: 12 }}>{error}</div>}
      {loading && <div className="skeleton" style={{ height: 220, marginTop: 16 }} />}

      {result && (
        <div style={{ marginTop: 20 }}>
          <div className="grid grid-2">
            <VerdictCard report={result.image_a} label="📷 REAL candidate" />
            <VerdictCard report={result.image_b} label="🤖 AI candidate" />
          </div>

          <div className="panel panel-pad" style={{ marginTop: 16 }}>
            <div className="eyebrow">Which looks more synthetic?</div>
            <div className="mono" style={{
              fontSize: 22, fontWeight: 800, marginTop: 6,
              color: aiDelta >= 10 ? '#f43f5e' : aiDelta <= -10 ? '#f59e0b' : 'var(--text)',
            }}>
              {aiDelta > 0 ? '+' : ''}{aiDelta.toFixed(1)} pts <span className="faint" style={{ fontSize: 14, fontWeight: 400 }}>AI-probability (AI − REAL)</span>
            </div>
            <p className="dim" style={{ fontSize: 13.5, marginTop: 8, marginBottom: 0 }}>{deltaReading}</p>
            <p className="faint mono" style={{ fontSize: 12, marginTop: 6, marginBottom: 0 }}>
              Manipulation delta (AI − REAL): {manipDelta > 0 ? '+' : ''}{manipDelta.toFixed(1)} pts
            </p>
          </div>

          <h2 className="section-title">Signal differences <span className="faint" style={{ fontSize: 13 }}>(AI − REAL)</span></h2>
          <div className="panel panel-pad" style={{ overflowX: 'auto' }}>
            <table className="data">
              <thead><tr><th>Signal</th><th>AI (REAL)</th><th>AI (AI img)</th><th>Δ AI</th><th>Manip (REAL)</th><th>Manip (AI img)</th><th>Δ Manip</th></tr></thead>
              <tbody>
                {(result.comparison.signal_deltas ?? []).map((d: any) => (
                  <tr key={d.signal}>
                    <td>{d.signal}</td>
                    <td className="mono">{Number(d.ai_a).toFixed(0)}</td>
                    <td className="mono">{Number(d.ai_b).toFixed(0)}</td>
                    <td className="mono" style={{ color: Math.abs(d.ai_b - d.ai_a) >= 10 ? (d.ai_b > d.ai_a ? '#f43f5e' : '#22c55e') : 'var(--text-faint)', fontWeight: 700 }}>
                      {(d.ai_b - d.ai_a) > 0 ? '+' : ''}{(d.ai_b - d.ai_a).toFixed(0)}
                    </td>
                    <td className="mono">{Number(d.manip_a).toFixed(0)}</td>
                    <td className="mono">{Number(d.manip_b).toFixed(0)}</td>
                    <td className="mono" style={{ color: Math.abs(d.manip_b - d.manip_a) >= 10 ? (d.manip_b > d.manip_a ? '#f59e0b' : '#22c55e') : 'var(--text-faint)', fontWeight: 700 }}>
                      {(d.manip_b - d.manip_a) > 0 ? '+' : ''}{(d.manip_b - d.manip_a).toFixed(0)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <h2 className="section-title">Forensic spotlight</h2>
          <div className="grid grid-2">
            <Spotlight report={result.image_a} label="📷 REAL candidate" />
            <Spotlight report={result.image_b} label="🤖 AI candidate" />
          </div>

          <h2 className="section-title">ML detector</h2>
          <div className="grid grid-2">
            {(['image_a', 'image_b'] as const).map((k, i) => {
              const ens = result[k].ensemble;
              const mlProb = ens.ml_ai_probability;
              return (
                <div className="panel panel-pad" key={k}>
                  <div className="eyebrow">{i === 0 ? '📷 REAL candidate' : '🤖 AI candidate'}</div>
                  {ens.ml_available && mlProb !== null && mlProb !== undefined ? (
                    <dl className="kv" style={{ marginTop: 8 }}>
                      <div style={{ display: 'contents' }}><dt>Calibrated P(AI)</dt><dd className="mono">{Number(mlProb).toFixed(1)}%</dd></div>
                      <div style={{ display: 'contents' }}><dt>Detectors</dt><dd className="mono" style={{ fontSize: 12 }}>
                        {(ens.detectors ?? []).filter((d: any) => d.available).map((d: any) => d.detector_name).join(', ') || '—'}
                      </dd></div>
                    </dl>
                  ) : (
                    <p className="faint" style={{ fontSize: 13, marginTop: 8, marginBottom: 0 }}>
                      {ens.ml_suppressed_reason
                        ? `ML output withheld — this verdict used forensic signals only. ${ens.ml_suppressed_reason}`
                        : 'ML detector not configured — this verdict used forensic signals only.'}
                    </p>
                  )}
                </div>
              );
            })}
          </div>

          <h2 className="section-title">Read this carefully</h2>
          <div className="panel panel-pad">
            <div className="evidence-item"><span className="ev-dot" style={{ color: 'var(--warn)' }}>•</span><span className="dim">Lighting, heavy editing, screenshots and social-media compression can push a genuine photo toward the synthetic side — and can hide synthetic traces.</span></div>
            <div className="evidence-item"><span className="ev-dot" style={{ color: 'var(--warn)' }}>•</span><span className="dim">One pair proves nothing about the detector. A controlled comparison is a demonstration, not an evaluation — real benchmarking needs hundreds of labelled images.</span></div>
            <div className="evidence-item"><span className="ev-dot" style={{ color: 'var(--warn)' }}>•</span><span className="dim">Try swapping in your own images: a heavily edited photo, a screenshot, a compressed download. Watch which signals move — that's where the learning happens.</span></div>
          </div>
        </div>
      )}
    </div>
  );
}
