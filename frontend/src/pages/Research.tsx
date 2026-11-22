import { useEffect, useMemo, useRef, useState } from 'react';
import { useNavigate, useParams } from 'react-router-dom';
import { getResearchView } from '../api';
import type { Ensemble, MLDetector, Reproducibility, Signal } from '../types';

interface StageInfo {
  stage: string;
  display: string;
  status: string;
  duration_ms: number;
}

interface ResearchView {
  analysis_id: string;
  report_hash: string;
  reproducibility: Reproducibility;
  stages: StageInfo[];
  signals: Signal[];
  ensemble: Ensemble;
  limitations: string[];
  note: string;
}

type SortKey = 'deviation' | 'name' | 'ai_score' | 'manip_score' | 'confidence' | 'status' | 'duration_ms';
interface SortCfg { key: SortKey; dir: 'asc' | 'desc' }

const clamp01 = (v: number) => Math.max(0, Math.min(1, v));

/** Simple blue -> violet -> red colormap for noise grids. */
function blueRed(v: number): [number, number, number] {
  const stops: [number, [number, number, number]][] = [
    [0, [30, 58, 138]],
    [0.5, [139, 92, 246]],
    [1, [239, 68, 68]],
  ];
  for (let i = 1; i < stops.length; i++) {
    if (v <= stops[i][0]) {
      const [v0, c0] = stops[i - 1];
      const [v1, c1] = stops[i];
      const t = (v - v0) / (v1 - v0);
      return [
        Math.round(c0[0] + t * (c1[0] - c0[0])),
        Math.round(c0[1] + t * (c1[1] - c0[1])),
        Math.round(c0[2] + t * (c1[2] - c0[2])),
      ];
    }
  }
  return [239, 68, 68];
}

function scoreColor(score: number): string {
  if (score >= 60) return 'var(--bad)';
  if (score >= 40) return 'var(--warn)';
  return 'var(--good)';
}

function statusColor(s: string): string {
  if (s === 'ok') return 'var(--good)';
  if (s === 'warning') return 'var(--warn)';
  return 'var(--text-faint)';
}

function truncHash(h: string | null | undefined, n = 16): string {
  if (!h) return '—';
  return h.length > n + 3 ? `${h.slice(0, n)}…` : h;
}

function fmtVal(v: unknown, max = 120): string {
  const s = typeof v === 'object' ? JSON.stringify(v) : String(v);
  return s.length > max ? `${s.slice(0, max)}…` : s;
}

/* ---------- canvas grid heatmap ---------- */

function GridCanvas({ grid, label }: { grid: number[][]; label: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current;
    if (!c || !grid.length) return;
    const h = grid.length;
    const w = grid[0]?.length ?? 1;
    c.width = w;
    c.height = h;
    const ctx = c.getContext('2d');
    if (!ctx) return;
    const img = ctx.createImageData(w, h);
    for (let y = 0; y < h; y++) {
      for (let x = 0; x < w; x++) {
        const [r, g, b] = blueRed(clamp01(Number(grid[y][x]) || 0));
        const i = (y * w + x) * 4;
        img.data[i] = r;
        img.data[i + 1] = g;
        img.data[i + 2] = b;
        img.data[i + 3] = 255;
      }
    }
    ctx.putImageData(img, 0, 0);
  }, [grid]);
  return (
    <canvas
      ref={ref}
      aria-label={label}
      style={{
        width: '100%', maxWidth: 520, imageRendering: 'pixelated',
        borderRadius: 8, border: '1px solid var(--border)', background: '#000',
      }}
    />
  );
}

/* ---------- detector card ---------- */

function DetectorCard({ d }: { d: MLDetector }) {
  const preEntries = Object.entries(d.preprocessing ?? {});
  return (
    <div className="card" style={{ height: '100%' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', gap: 8, flexWrap: 'wrap' }}>
        <h3 style={{ margin: 0 }}>{d.detector_name} <span className="dim">v{d.detector_version}</span></h3>
        <span className="tag" style={d.available ? { color: 'var(--good)' } : { color: 'var(--warn)' }}>
          {d.model_status}
        </span>
      </div>
      <div className="faint" style={{ fontSize: 12.5, marginBottom: 10 }}>model type: {d.model_type}</div>
      <div className="grid grid-2" style={{ gap: 10 }}>
        <div>
          <div className="eyebrow" style={{ marginBottom: 4 }}>Raw model score</div>
          <div className="mono" style={{ fontSize: 22, fontWeight: 700 }}>
            {d.raw_score == null ? '—' : Number(d.raw_score).toFixed(4)}
          </div>
          <div className="faint" style={{ fontSize: 11.5 }}>uncalibrated output — diagnostic only</div>
        </div>
        <div>
          <div className="eyebrow" style={{ marginBottom: 4 }}>Calibrated P(AI)</div>
          <div className="mono" style={{ fontSize: 22, fontWeight: 700, color: scoreColor((d.probability_ai ?? 50)) }}>
            {d.probability_ai == null ? '—' : `${Number(d.probability_ai).toFixed(1)}%`}
          </div>
          <div className="faint" style={{ fontSize: 11.5 }}>via {d.calibration}</div>
        </div>
        <div>
          <div className="eyebrow" style={{ marginBottom: 4 }}>Confidence</div>
          <div className="mono" style={{ fontSize: 15, fontWeight: 700 }}>{d.confidence.toFixed(0)}%</div>
        </div>
        <div>
          <div className="eyebrow" style={{ marginBottom: 4 }}>Inference time</div>
          <div className="mono" style={{ fontSize: 15, fontWeight: 700 }}>{d.inference_time_ms.toFixed(1)} ms</div>
        </div>
      </div>
      {preEntries.length > 0 && (
        <details className="expandable">
          <summary>Preprocessing ({preEntries.length} steps recorded)</summary>
          <div className="body">
            <dl className="kv">
              {preEntries.map(([k, v]) => (
                <div key={k} style={{ display: 'contents' }}>
                  <dt>{k}</dt>
                  <dd>{fmtVal(v)}</dd>
                </div>
              ))}
            </dl>
          </div>
        </details>
      )}
      {d.limitations && <div className="faint" style={{ fontSize: 12, marginTop: 10 }}>Limitations: {d.limitations}</div>}
      {d.detail && <div className="faint" style={{ fontSize: 12, marginTop: 4 }}>{d.detail}</div>}
    </div>
  );
}

/* ---------- main page ---------- */

export default function Research() {
  const { id } = useParams<{ id?: string }>();
  const navigate = useNavigate();
  const [input, setInput] = useState(id ?? '');
  const [view, setView] = useState<ResearchView | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [sort, setSort] = useState<SortCfg>({ key: 'deviation', dir: 'desc' });
  const [expanded, setExpanded] = useState<string | null>(null);

  useEffect(() => {
    setInput(id ?? '');
    if (!id) {
      setView(null);
      setError(null);
      setLoading(false);
      return;
    }
    let cancelled = false;
    setLoading(true);
    setError(null);
    setView(null);
    getResearchView(id)
      .then((v) => { if (!cancelled) setView(v as ResearchView); })
      .catch((e: unknown) => { if (!cancelled) setError(e instanceof Error ? e.message : String(e)); })
      .finally(() => { if (!cancelled) setLoading(false); });
    return () => { cancelled = true; };
  }, [id]);

  const sortedSignals = useMemo(() => {
    const arr = [...(view?.signals ?? [])];
    const dir = sort.dir === 'asc' ? 1 : -1;
    const val = (s: Signal): number | string => {
      switch (sort.key) {
        case 'name': return s.display_name || s.name;
        case 'ai_score': return s.ai_score;
        case 'manip_score': return s.manip_score;
        case 'confidence': return s.confidence;
        case 'status': return s.status;
        case 'duration_ms': return s.duration_ms;
        case 'deviation': return Math.abs(s.ai_score - 50);
      }
    };
    arr.sort((a, b) => {
      const va = val(a);
      const vb = val(b);
      if (typeof va === 'string') return va.localeCompare(vb as string) * dir;
      return (va - (vb as number)) * dir;
    });
    return arr;
  }, [view, sort]);

  const onSort = (key: SortKey) => {
    setSort((prev) => {
      if (prev.key === key) return { key, dir: prev.dir === 'asc' ? 'desc' : 'asc' };
      return { key, dir: key === 'name' || key === 'status' ? 'asc' : 'desc' };
    });
  };

  const exportJson = () => {
    if (!view) return;
    const blob = new Blob([JSON.stringify(view, null, 2)], { type: 'application/json' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url;
    a.download = `truthlenz-research-${view.analysis_id}.json`;
    document.body.appendChild(a);
    a.click();
    a.remove();
    URL.revokeObjectURL(url);
  };

  const ens = view?.ensemble;
  const detectors = ens?.detectors ?? [];
  const activeDetectors = detectors.filter((d) => d.available);
  const repro = view?.reproducibility;
  const stages = view?.stages ?? [];
  const maxStageMs = Math.max(1, ...stages.map((s) => s.duration_ms ?? 0));

  const sigByName = (n: string): Signal | undefined => view?.signals.find((s) => s.name === n);
  const noiseGrid: number[][] | null = (sigByName('noise')?.visuals?.noise_map?.grid as number[][] | undefined) ?? null;
  const heatmaps: { label: string; kind: 'image' | 'grid'; src?: string; grid?: number[][]; note: string }[] = [];
  const elaSrc = sigByName('pixel')?.visuals?.ela as string | undefined;
  if (elaSrc) heatmaps.push({ label: 'ELA', kind: 'image', src: elaSrc, note: 'Error-level response. Bright areas differ in compression history — not proof of editing.' });
  if (noiseGrid) heatmaps.push({ label: 'Noise map', kind: 'grid', grid: noiseGrid, note: 'Estimated noise per block (blue = low, red = high). Inconsistency can suggest compositing; not proof.' });
  const specSrc = sigByName('frequency')?.visuals?.spectrum as string | undefined;
  if (specSrc) heatmaps.push({ label: 'Frequency spectrum', kind: 'image', src: specSrc, note: 'Log-magnitude Fourier spectrum of luminance (DC centered).' });
  const manipSrc = sigByName('manipulation')?.visuals?.manipulation_overlay as string | undefined;
  if (manipSrc) heatmaps.push({ label: 'Manipulation overlay', kind: 'image', src: manipSrc, note: 'Combined anomaly overlay from copy-move, texture and edge analysis.' });
  detectors.forEach((d) => {
    if (d.attention_heatmap) {
      heatmaps.push({ label: `ML attention — ${d.detector_name}`, kind: 'image', src: d.attention_heatmap, note: 'Model attention visualization — shows where the model looked, NOT proof of AI generation.' });
    }
  });

  const cols: { key: SortKey; label: string }[] = [
    { key: 'name', label: 'Signal' },
    { key: 'ai_score', label: 'AI score' },
    { key: 'manip_score', label: 'Manip score' },
    { key: 'confidence', label: 'Confidence' },
    { key: 'status', label: 'Status' },
    { key: 'duration_ms', label: 'Duration (ms)' },
  ];

  return (
    <div className="shell" style={{ paddingTop: 34, paddingBottom: 60 }}>
      {/* header */}
      <div className="eyebrow">Research mode</div>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'flex-start', gap: 16, flexWrap: 'wrap' }}>
        <div>
          <h1 style={{ margin: '0 0 6px', fontSize: 32, letterSpacing: '-0.02em' }}>Raw forensic output</h1>
          <div className="dim mono" style={{ fontSize: 13, wordBreak: 'break-all' }}>
            analysis&nbsp;<span style={{ color: 'var(--accent)' }}>{view?.analysis_id ?? id ?? '—'}</span>
          </div>
          {view && (
            <div className="faint mono" style={{ fontSize: 12, marginTop: 4, wordBreak: 'break-all' }}>
              report hash&nbsp;{truncHash(view.report_hash, 24)}
            </div>
          )}
        </div>
        <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap' }}>
          <button className="btn btn-ghost btn-sm" onClick={() => navigate('/history')}>← History</button>
          <button className="btn btn-ghost btn-sm" onClick={exportJson} disabled={!view}>Export raw JSON</button>
        </div>
      </div>

      <div className="alert warn" style={{ marginTop: 16 }}>
        <b>Raw outputs.</b> Scores below are model and signal outputs, not ground truth. A high AI score is
        a statistical signal, not a determination. See Limitations for what each number cannot say.
      </div>

      {/* manual id entry */}
      <div className="card" style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap', marginBottom: 8 }}>
        <input
          type="text"
          value={input}
          onChange={(e) => setInput(e.target.value)}
          onKeyDown={(e) => { if (e.key === 'Enter' && input.trim()) navigate(`/research/${input.trim()}`); }}
          placeholder="Enter analysis ID (from a saved analysis)…"
          aria-label="Analysis ID"
          style={{ maxWidth: 420 }}
        />
        <button
          className="btn btn-primary btn-sm"
          onClick={() => { if (input.trim()) navigate(`/research/${input.trim()}`); }}
          disabled={!input.trim()}
        >
          Load
        </button>
        {!id && <span className="faint" style={{ fontSize: 12.5 }}>Tip: run an analysis with “save to history” enabled, then paste its ID here.</span>}
      </div>

      {loading && (
        <div className="grid" style={{ marginTop: 18 }}>
          <div className="skeleton" style={{ height: 260 }} />
          <div className="skeleton" style={{ height: 180 }} />
        </div>
      )}

      {error && (
        <div className="alert error" style={{ marginTop: 18 }}>
          <b>Could not load research view.</b> {error}
          <div className="dim" style={{ marginTop: 6, fontSize: 13 }}>
            Only saved analyses have a research view — re-run the analysis with “save to history” enabled first.
          </div>
        </div>
      )}

      {view && ens && (
        <>
          {/* ---------- raw analyzer scores ---------- */}
          <h2 className="section-title">Raw analyzer scores</h2>
          <p className="section-sub">
            Per-signal evidence before ensemble fusion. Click a column header to sort (default: distance of the
            AI score from the neutral 50 mark). Click a row to inspect technical detail.
          </p>
          <div className="panel" style={{ overflowX: 'auto' }}>
            <table className="data">
              <thead>
                <tr>
                  <th className="sortable" onClick={() => onSort('deviation')}>
                    |AI − 50| {sort.key === 'deviation' ? (sort.dir === 'asc' ? '↑' : '↓') : ''}
                  </th>
                  {cols.map((c) => (
                    <th key={c.key} className="sortable" onClick={() => onSort(c.key)}>
                      {c.label} {sort.key === c.key ? (sort.dir === 'asc' ? '↑' : '↓') : ''}
                    </th>
                  ))}
                </tr>
              </thead>
              <tbody>
                {sortedSignals.map((s) => {
                  const open = expanded === s.name;
                  return [
                    <tr key={s.name} onClick={() => setExpanded(open ? null : s.name)} style={{ cursor: 'pointer' }}>
                      <td className="mono" style={{ color: scoreColor(s.ai_score) }}>
                        {Math.abs(s.ai_score - 50).toFixed(1)}
                      </td>
                      <td><b>{s.display_name || s.name}</b></td>
                      <td className="mono" style={{ color: scoreColor(s.ai_score) }}>{s.ai_score.toFixed(1)}</td>
                      <td className="mono" style={{ color: scoreColor(s.manip_score) }}>{s.manip_score.toFixed(1)}</td>
                      <td className="mono">{s.confidence.toFixed(0)}%</td>
                      <td><span className="tag" style={{ color: statusColor(s.status) }}>{s.status}</span></td>
                      <td className="mono">{s.duration_ms.toFixed(1)}</td>
                    </tr>,
                    open && (
                      <tr key={`${s.name}-detail`}>
                        <td colSpan={7} style={{ background: 'var(--panel-solid)' }}>
                          <div className="grid grid-2" style={{ gap: 16 }}>
                            <div>
                              <div className="eyebrow">Technical detail</div>
                              <pre
                                className="mono"
                                style={{
                                  fontSize: 11.5, background: '#000', color: '#7dd3fc',
                                  padding: 12, borderRadius: 8, overflow: 'auto', maxHeight: 320,
                                  border: '1px solid var(--border)', whiteSpace: 'pre-wrap',
                                }}
                              >
                                {fmtVal(s.technical, 2000)}
                              </pre>
                            </div>
                            <div>
                              <div className="eyebrow">Evidence ({s.evidence.length})</div>
                              {s.evidence.length === 0 && <div className="faint" style={{ fontSize: 13 }}>No evidence items.</div>}
                              {s.evidence.map((e, i) => (
                                <div key={i} className="evidence-item">
                                  <span className="ev-dot">{e.direction === 'supports_ai' || e.direction === 'supports_manip' ? '▲' : e.direction === 'supports_authentic' ? '▽' : '•'}</span>
                                  <div>
                                    <div>{e.text}</div>
                                    <div className="faint" style={{ fontSize: 12 }}>{e.detail}</div>
                                  </div>
                                </div>
                              ))}
                              {s.warnings.length > 0 && (
                                <>
                                  <div className="eyebrow" style={{ marginTop: 12 }}>Warnings</div>
                                  <ul style={{ fontSize: 13, color: 'var(--warn)', margin: '4px 0', paddingLeft: 18 }}>
                                    {s.warnings.map((w, i) => <li key={i}>{w}</li>)}
                                  </ul>
                                </>
                              )}
                            </div>
                          </div>
                        </td>
                      </tr>
                    ),
                  ];
                })}
              </tbody>
            </table>
          </div>

          {/* ---------- score chart ---------- */}
          <h2 className="section-title">Signal AI scores</h2>
          <p className="section-sub">
            0 = authentic-leaning, 50 = neutral, 100 = AI-leaning. Positions are raw signal outputs, not verdicts.
          </p>
          <div className="card" style={{ display: 'flex', flexDirection: 'column', gap: 9 }}>
            {sortedSignals.map((s) => (
              <div key={s.name} style={{ display: 'grid', gridTemplateColumns: '200px 1fr 56px', gap: 12, alignItems: 'center' }}>
                <div style={{ fontSize: 12.5, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={s.display_name || s.name}>
                  {s.display_name || s.name}
                </div>
                <div style={{ position: 'relative', height: 14, background: 'var(--border)', borderRadius: 7 }}>
                  <div
                    style={{
                      position: 'absolute', left: '50%', top: -2, bottom: -2, width: 2,
                      background: 'var(--text-faint)', opacity: 0.7,
                    }}
                    title="neutral (50)"
                  />
                  <div
                    style={{
                      position: 'absolute', left: `${clamp01(s.ai_score / 100) * 100}%`, top: -2,
                      width: 10, height: 18, borderRadius: 5,
                      background: scoreColor(s.ai_score), transform: 'translateX(-50%)',
                    }}
                    title={`AI score ${s.ai_score.toFixed(1)}`}
                  />
                </div>
                <div className="mono" style={{ fontSize: 12.5, color: scoreColor(s.ai_score) }}>{s.ai_score.toFixed(1)}</div>
              </div>
            ))}
          </div>

          {/* ---------- detectors ---------- */}
          <h2 className="section-title">ML detector outputs</h2>
          {!ens.ml_available || activeDetectors.length === 0 ? (
            <div className="panel panel-pad">
              {ens.ml_suppressed_reason ? (
                <>
                  <b>ML probability withheld.</b>
                  <div className="dim" style={{ fontSize: 13.5, marginTop: 6 }}>
                    {ens.ml_suppressed_reason} The trained model is configured, but its output is not
                    presented as a calibrated probability. This analysis ran on the forensic engine only.
                  </div>
                </>
              ) : (
                <>
                  <b>No ML detector configured.</b>
                  <div className="dim" style={{ fontSize: 13.5, marginTop: 6 }}>
                    This analysis ran on the forensic engine only. No trained classification model contributed to the
                    scores above. Connect a model (see Models in Settings) to enable ML detection — absence of a model
                    is not a signal of anything about this image.
                  </div>
                </>
              )}
              {detectors.map((d) => (
                <div key={d.detector_name} className="faint" style={{ fontSize: 12.5, marginTop: 8 }}>
                  {d.detector_name} v{d.detector_version} — {d.model_status}. {d.limitations}
                </div>
              ))}
            </div>
          ) : (
            <div className="grid grid-2">
              {activeDetectors.map((d) => <DetectorCard key={d.detector_name} d={d} />)}
            </div>
          )}

          {/* ---------- calibration ---------- */}
          <h2 className="section-title">Calibration</h2>
          <div className="panel panel-pad">
            <div className="alert" style={{ marginTop: 0 }}>
              Raw model scores are <b>not</b> probabilities. They are uncalibrated outputs (logits, distances, or
              softmax values) whose scale depends on the model. A probability is only shown once a documented
              calibration step has been applied. Raw scores here are for diagnostics — do not quote them as
              “probability of AI generation”.
            </div>
            <table className="data">
              <thead>
                <tr><th>Detector</th><th>Calibration method</th><th>Raw score</th><th>Calibrated P(AI)</th></tr>
              </thead>
              <tbody>
                {detectors.length === 0 && (
                  <tr><td colSpan={4} className="faint">No detectors evaluated — nothing to calibrate.</td></tr>
                )}
                {detectors.map((d) => {
                  // Historical reports (pre-Stage-2 suppression flags)
                  // carry ml_suppressed_reason while their detector
                  // entries still say available=true and lack the
                  // suppressed flag. A valued entry under a suppression
                  // reason is a withheld output, not a calibrated
                  // probability — never place its raw value in this
                  // column. Null entries still render '—'.
                  const withheld = Boolean(
                    d.suppressed || (ens?.ml_suppressed_reason && d.probability_ai != null));
                  return (
                    <tr key={d.detector_name}>
                      <td>{d.detector_name} <span className="faint">v{d.detector_version}</span></td>
                      <td className="mono" style={{ fontSize: 12.5 }}>{d.calibration}</td>
                      <td className="mono">{d.raw_score == null ? '—' : Number(d.raw_score).toFixed(4)}</td>
                      <td className="mono" style={{ color: withheld ? undefined : scoreColor(d.probability_ai ?? 50) }}>
                        {withheld ? 'withheld' : (d.probability_ai == null ? '—' : `${Number(d.probability_ai).toFixed(1)}%`)}
                      </td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>

          {/* ---------- configuration ---------- */}
          <h2 className="section-title">Configuration</h2>
          <div className="grid grid-2">
            <div className="card">
              <h3>Thresholds used</h3>
              <table className="data">
                <thead><tr><th>Key</th><th>Value</th></tr></thead>
                <tbody>
                  {Object.entries(ens.thresholds_used ?? {}).map(([k, v]) => (
                    <tr key={k}><td className="mono" style={{ fontSize: 12.5 }}>{k}</td><td className="mono">{fmtVal(v)}</td></tr>
                  ))}
                  {Object.keys(ens.thresholds_used ?? {}).length === 0 && (
                    <tr><td colSpan={2} className="faint">No thresholds recorded.</td></tr>
                  )}
                </tbody>
              </table>
            </div>
            <div className="card">
              <h3>Hybrid ensemble weights</h3>
              <table className="data">
                <thead><tr><th>Component</th><th>Weight</th></tr></thead>
                <tbody>
                  {Object.entries(ens.hybrid_weights ?? {}).map(([k, v]) => (
                    <tr key={k}><td>{k}</td><td className="mono">{fmtVal(v)}</td></tr>
                  ))}
                </tbody>
              </table>
              <div className="faint" style={{ fontSize: 12.5, marginTop: 10 }}>
                The ML detector does not automatically dominate the forensic engine; weights are configurable and
                disagreement between high-quality detectors increases uncertainty rather than forcing a verdict.
              </div>
            </div>
          </div>
          {repro && (
            <div className="card" style={{ marginTop: 16 }}>
              <div className="kv">
                <dt>Config version</dt><dd>{repro.config_version}</dd>
                <dt>Config SHA-256</dt><dd>{repro.config_sha256}</dd>
              </div>
            </div>
          )}

          {/* ---------- execution times ---------- */}
          <h2 className="section-title">Execution times</h2>
          <div className="card">
            <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'baseline', flexWrap: 'wrap', gap: 8 }}>
              <span className="dim" style={{ fontSize: 13.5 }}>Per-stage pipeline timing</span>
              <span className="mono" style={{ fontSize: 13 }}>total {repro ? repro.pipeline_ms.toFixed(1) : '—'} ms</span>
            </div>
            <div style={{ display: 'flex', flexDirection: 'column', gap: 8, marginTop: 12 }}>
              {stages.map((st) => (
                <div key={st.stage} style={{ display: 'grid', gridTemplateColumns: '220px 1fr 84px', gap: 12, alignItems: 'center' }}>
                  <div style={{ fontSize: 12.5, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }} title={st.display}>
                    {st.display}
                  </div>
                  <div className="progress">
                    <div style={{ width: `${(st.duration_ms / maxStageMs) * 100}%` }} />
                  </div>
                  <div className="mono" style={{ fontSize: 12.5, textAlign: 'right' }}>{st.duration_ms.toFixed(1)} ms</div>
                </div>
              ))}
              {stages.length === 0 && <div className="faint" style={{ fontSize: 13 }}>No stage timing recorded.</div>}
            </div>
          </div>

          {/* ---------- heatmaps ---------- */}
          <h2 className="section-title">Forensic heatmaps</h2>
          <p className="section-sub">
            Visualizations highlight regions of interest. Brighter response means a stronger local signal —
            they do not imply certainty and are never proof of AI generation or manipulation.
          </p>
          {heatmaps.length === 0 ? (
            <div className="empty">No heatmap layers available for this analysis.</div>
          ) : (
            <div className="grid grid-2">
              {heatmaps.map((h) => (
                <div className="card" key={h.label}>
                  <h3>{h.label}</h3>
                  {h.kind === 'image' && h.src
                    ? <img src={h.src} alt={`${h.label} visualization`} style={{ width: '100%', borderRadius: 8, border: '1px solid var(--border)' }} />
                    : h.grid ? <GridCanvas grid={h.grid} label={h.label} /> : null}
                  <div style={{ fontSize: 12, color: 'var(--text-faint)', marginTop: 8 }}>{h.note}</div>
                </div>
              ))}
            </div>
          )}

          {/* ---------- reproducibility ---------- */}
          <h2 className="section-title">Reproducibility</h2>
          {repro ? (
            <div className="card">
              <dl className="kv">
                <dt>Analysis ID</dt><dd>{repro.analysis_id}</dd>
                <dt>App version</dt><dd>{repro.app_version}</dd>
                <dt>Backend version</dt><dd>{repro.backend_version}</dd>
                <dt>Config version</dt><dd>{repro.config_version}</dd>
                <dt>Config SHA-256</dt><dd>{repro.config_sha256}</dd>
                <dt>Image SHA-256</dt><dd title={repro.image_sha256}>{truncHash(repro.image_sha256, 24)}</dd>
                <dt>Image dimensions</dt><dd>{(repro.image_dimensions ?? []).join(' × ')} px</dd>
                <dt>Analyzed at</dt><dd>{repro.analyzed_at}</dd>
                <dt>Pipeline time</dt><dd>{repro.pipeline_ms.toFixed(1)} ms</dd>
              </dl>
              <div className="eyebrow" style={{ marginTop: 16 }}>Detectors</div>
              {(repro.detectors ?? []).map((d) => (
                <div key={d.name} className="faint mono" style={{ fontSize: 12.5 }}>
                  {d.name} v{d.version} · {d.model_type} · {d.status} · calibration: {d.calibration}
                </div>
              ))}
              {(repro.detectors ?? []).length === 0 && (
                <div className="faint" style={{ fontSize: 12.5 }}>No ML detectors — forensic engine only.</div>
              )}
              {repro.note && <div className="faint" style={{ fontSize: 12.5, marginTop: 8 }}>{repro.note}</div>}
            </div>
          ) : (
            <div className="empty">No reproducibility record.</div>
          )}

          {/* ---------- limitations ---------- */}
          <h2 className="section-title">Limitations</h2>
          <div className="panel panel-pad">
            <ul style={{ margin: 0, paddingLeft: 18, fontSize: 13.5, color: 'var(--text-dim)' }}>
              {(view.limitations ?? []).map((l, i) => <li key={i} style={{ marginBottom: 6 }}>{l}</li>)}
            </ul>
            {view.note && <div className="faint" style={{ fontSize: 12.5, marginTop: 10 }}>{view.note}</div>}
          </div>
        </>
      )}
    </div>
  );
}
