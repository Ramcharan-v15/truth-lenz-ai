import { useState } from 'react';
import type { AnalysisReport } from '../types';
import { VERDICT_META, fmtBytes, fmtPct, fmtNum, fmtFractionPct, isFiniteNumber } from '../types';
import Gauge, { ProbBar } from './Gauge';
import Histogram, { RadialChart } from './Histogram';
import EvidenceCard from './EvidenceCard';
import MetadataTable from './MetadataTable';
import ForensicsViewer from './ForensicsViewer';
import Scorecard from './Scorecard';
import ModelStatus from './ModelStatus';
import EvidenceGraph from './EvidenceGraph';
import HeatmapLayer from './HeatmapLayer';
import { reportUrl } from '../api';

function EvList({ items, empty }: { items: { analyzer: string; text: string; detail: string }[]; empty: string }) {
  if (!items.length) return <p className="faint" style={{ fontSize: 13 }}>{empty}</p>;
  return (
    <div>
      {items.map((e, i) => (
        <div key={i} className="evidence-item">
          <span className="ev-dot" style={{ color: 'var(--accent)' }}>▸</span>
          <span><strong style={{ fontSize: 12 }}>{e.analyzer}:</strong> {e.text}</span>
        </div>
      ))}
    </div>
  );
}

export default function ResultsDashboard({ report }: { report: AnalysisReport }) {
  const e = report.ensemble;
  const meta = VERDICT_META[e.verdict];
  const [showReports, setShowReports] = useState(false);
  const [explLevel, setExplLevel] = useState<'basic' | 'technical' | 'forensic'>('basic');
  const stats = report.signals.find((s) => s.name === 'statistics');
  const freq = report.signals.find((s) => s.name === 'frequency');
  const metadata = report.signals.find((s) => s.name === 'metadata');
  const quality = report.signals.find((s) => s.name === 'quality');
  const c2paStatus = (metadata?.technical?.c2pa_status as string) ?? 'missing';
  const c2paInfo = (metadata?.technical?.c2pa_info as Record<string, string>) ?? {};
  // Stage 1: the probability-band lookup runs ONLY on a finite number. A
  // withheld (null) probability gets no band label at all — never an
  // authentic-leaning label via the null <= max quirk.
  const aiProb = e.ai_probability;
  const aiBandLabel = isFiniteNumber(aiProb) && Array.isArray(e.probability_bands)
    ? e.probability_bands.find(b => aiProb <= b.max)?.label
    : undefined;

  return (
    <div>
      {report.demo && (
        <div className="demo-banner" role="alert">
          ⚠ DEMO RESULT — NOT REAL ANALYSIS. This is sample output for UI testing only.
        </div>
      )}

      <ModelStatus report={report} />
      <div style={{ height: 16 }} />

      {/* ------- verdict ------- */}
      <div className="eyebrow">Authenticity Verdict {report.case_id && <span className="mono" style={{ marginLeft: 8 }}>· CASE {report.case_id}</span>}</div>
      <div className="verdict-banner" style={{ borderColor: meta.color + '55' }}>
        <div className="verdict-icon" style={{ color: meta.color }}>{meta.icon}</div>
        <div style={{ flex: '1 1 220px' }}>
          <div style={{ fontSize: 26, fontWeight: 800, color: meta.color, letterSpacing: '-0.01em' }}>
            {meta.label.toUpperCase()}
          </div>
          <div className="dim" style={{ fontSize: 13.5, marginTop: 4 }}>{e.verdict_explanation}</div>
          <div className="faint mono" style={{ fontSize: 11.5, marginTop: 6 }}>
            {e.verdict_reason}
          </div>
        </div>
        <div className="metric" style={{ textAlign: 'right' }}>
          <div className="lbl">Confidence</div>
          <div className="val" style={{ color: e.confidence_label === 'HIGH' ? '#22c55e' : e.confidence_label === 'MEDIUM' ? '#f59e0b' : '#f43f5e' }}>
            {e.confidence_label}
          </div>
          <div className="faint mono" style={{ fontSize: 12 }}>{fmtPct(e.confidence, 0)}</div>
        </div>
      </div>

      <div className="grid grid-3">
        <div className="panel panel-pad"><Gauge value={e.ai_probability} label="AI Generation" color="#f43f5e" sub={aiBandLabel} /></div>
        <div className="panel panel-pad"><Gauge value={e.manipulation_probability} label="Manipulation" color="#f59e0b" /></div>
        <div className="panel panel-pad"><Gauge value={e.authenticity_probability} label="Authentic Capture" color="#22c55e" /></div>
      </div>
      <div className="disclaimer-strip">Probability estimates — not definitive proof. Confidence ({e.confidence_label}, {fmtPct(e.confidence, 0)}) measures how much to trust these numbers, and is separate from the probabilities themselves.</div>

      {/* ------- false-positive risk ------- */}
      {e.false_positive_risk && e.false_positive_risk.level !== 'LOW' && (() => {
        const fp = e.false_positive_risk;
        const isUnknown = fp.level === 'UNKNOWN';
        const accent = isUnknown ? '#94a3b8' : '#f59e0b';
        return (
          <div className="alert" style={{ marginTop: 12, borderColor: `${accent}66`, background: `rgba(${isUnknown ? '148,163,184' : '245,158,11'},0.07)` }}>
            <strong>{isUnknown ? '○' : '⚠'} False-positive risk: {fp.level}</strong>
            {isUnknown && (
              <div className="faint" style={{ fontSize: 12, marginTop: 4 }}>
                Could not be assessed — insufficient evidence to judge false-positive risk.
              </div>
            )}
            {fp.reasons.length > 0 && (
              <div style={{ marginTop: 6, fontSize: 13 }}>
                {fp.reasons.map((r: string, i: number) => <div key={i}>• {r}</div>)}
              </div>
            )}
            {fp.note && <div className="faint" style={{ fontSize: 12, marginTop: 6 }}>{fp.note}</div>}
          </div>
        );
      })()}

      {/* ------- why uncertain ------- */}
      {e.uncertainty && (
        <div className="panel panel-pad" style={{ marginTop: 12, borderColor: '#94a3b866' }}>
          <div className="eyebrow" style={{ color: '#94a3b8' }}>{e.uncertainty.title}</div>
          <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginTop: 8 }}>
            {e.uncertainty.parties.map((p: any, i: number) => (
              <div key={i} className="card" style={{ flex: '1 1 160px' }}>
                <div style={{ fontSize: 12, fontWeight: 700 }}>{p.side}</div>
                <div className="mono" style={{ fontSize: 20, fontWeight: 800, marginTop: 4 }}>
                  {typeof p.ai_probability === 'number' ? `${p.ai_probability}% AI` : p.position}
                </div>
                <div className="faint" style={{ fontSize: 12 }}>{p.position}</div>
              </div>
            ))}
          </div>
          <p className="dim" style={{ fontSize: 13, marginTop: 10 }}>{e.uncertainty.explanation}</p>
        </div>
      )}

      {/* ------- why ------- */}
      <h2 className="section-title">Why this verdict?</h2>
      <p className="section-sub">The ensemble's reasoning, in plain language. Only evidence produced by actual analyzers is shown.</p>
      <div className="grid grid-3">
        <div className="card">
          <h3 style={{ color: meta.color }}>Strongest evidence</h3>
          <EvList items={e.strongest_evidence} empty="No single dominant finding — the verdict rests on the balance of signals." />
        </div>
        <div className="card">
          <h3 style={{ color: '#22c55e' }}>Evidence against</h3>
          <EvList items={e.evidence_against} empty="No strong counter-evidence recorded." />
        </div>
        <div className="card">
          <h3 style={{ color: '#f59e0b' }}>Uncertain factors</h3>
          <EvList items={e.uncertain_factors} empty="No major uncertainties flagged." />
          {e.confidence_penalties.length > 0 && (
            <div style={{ marginTop: 10 }}>
              <div className="eyebrow">Confidence reduced by</div>
              {e.confidence_penalties.map((p, i) => <div key={i} className="faint" style={{ fontSize: 12.5 }}>• {p}</div>)}
            </div>
          )}
        </div>
      </div>
      {e.why.length > 0 && (
        <div className="panel panel-pad" style={{ marginTop: 16 }}>
          <div className="eyebrow">Top contributing signals</div>
          {e.why.map((w, i) => <div key={i} className="mono" style={{ fontSize: 12.5, padding: '4px 0' }}>{i + 1}. {w}</div>)}
        </div>
      )}

      {/* ------- explanation levels ------- */}
      <h2 className="section-title">Explanation</h2>
      <p className="section-sub">Same findings, three depths. Generated deterministically from computed evidence — no language model scored this image.</p>
      <div className="panel panel-pad">
        <div style={{ display: 'flex', gap: 6, marginBottom: 10 }}>
          {(['basic', 'technical', 'forensic'] as const).map((lv) => (
            <button
              key={lv}
              onClick={() => setExplLevel(lv)}
              style={{
                padding: '6px 14px', borderRadius: 8, fontSize: 12, fontWeight: 700,
                textTransform: 'uppercase', letterSpacing: '0.05em',
                border: `1px solid ${explLevel === lv ? '#38bdf8' : 'var(--border)'}`,
                background: explLevel === lv ? 'rgba(56,189,248,0.12)' : 'transparent',
                color: 'var(--text)', cursor: 'pointer',
              }}
            >
              {lv}
            </button>
          ))}
        </div>
        <p style={{ fontSize: 14, lineHeight: 1.6, margin: 0 }}>{report.explanations?.[explLevel]}</p>
        <p className="faint" style={{ fontSize: 11.5, marginTop: 8 }}>{report.explanations?.note}</p>
      </div>

      {/* ------- evidence graph ------- */}
      <h2 className="section-title">Evidence Graph</h2>
      <p className="section-sub">How the hybrid ensemble assembled the verdict from forensics, ML models and provenance.</p>
      <EvidenceGraph report={report} />

      {/* ------- forensics viewer ------- */}
      <h2 className="section-title">Image Forensics Viewer</h2>
      <p className="section-sub">Original, ELA visualization, frequency spectrum and manipulation heatmap. Hover the original for pixel inspection.</p>
      <div className="panel panel-pad">
        <ForensicsViewer signals={report.signals} preview={report.preview} />
      </div>

      {/* ------- unified heatmaps ------- */}
      <h2 className="section-title">Forensic Visualizations</h2>
      <p className="section-sub">All heatmap layers in one place — switch, overlay, zoom, compare side-by-side.</p>
      <HeatmapLayer report={report} />

      {/* ------- evidence breakdown ------- */}
      <h2 className="section-title">Evidence Breakdown</h2>
      <p className="section-sub">Every analyzer's score, confidence, findings and technical details. Scores are 0–100; 50 is neutral.</p>
      <div className="grid grid-2">
        {report.signals.map((s) => (
          <EvidenceCard key={s.name} signal={s} learn={report.educational[s.name]} />
        ))}
      </div>

      {/* ------- metadata ------- */}
      <h2 className="section-title">Metadata Forensics</h2>
      <div className="panel panel-pad">
        <MetadataTable signal={metadata} filename={report.filename} fileSize={report.file_size} mimeType={report.mime_type} />
      </div>

      {/* ------- C2PA / provenance ------- */}
      <h2 className="section-title">Content Credentials (C2PA)</h2>
      <div className="panel panel-pad">
        <dl className="kv">
          <div style={{ display: 'contents' }}><dt>Status</dt><dd>
            <span style={{
              padding: '3px 10px', borderRadius: 20, fontSize: 12, fontWeight: 700,
              background: c2paStatus === 'unvalidated' ? 'rgba(167,139,250,0.15)' : c2paStatus === 'invalid' ? 'rgba(244,63,94,0.15)' : 'rgba(100,116,139,0.15)',
              color: c2paStatus === 'unvalidated' ? '#a78bfa' : c2paStatus === 'invalid' ? '#f43f5e' : 'var(--muted)',
            }}>
              {c2paStatus.toUpperCase()}
            </span>
          </dd></div>
          {Object.entries(c2paInfo).map(([k, v]) => (
            <div style={{ display: 'contents' }} key={k}><dt>{k.replace(/_/g, ' ')}</dt><dd>{String(v)}</dd></div>
          ))}
          {c2paStatus === 'missing' && (
            <div style={{ display: 'contents' }}><dt>Note</dt><dd style={{ fontFamily: 'var(--font)' }}>
              No Content Credentials found. Missing provenance is never treated as evidence of AI generation.
            </dd></div>
          )}
          {c2paStatus === 'unvalidated' && (
            <div style={{ display: 'contents' }}><dt>Note</dt><dd style={{ fontFamily: 'var(--font)' }}>
              Manifest detected and parsed heuristically, but the cryptographic signature was not validated here.
              Provenance claims are context, not pixel evidence.
            </dd></div>
          )}
        </dl>
      </div>

      {/* ------- statistics & frequency charts ------- */}
      <h2 className="section-title">Statistical & Frequency Analysis</h2>
      <div className="grid grid-2">
        <div className="panel panel-pad">
          <div className="eyebrow">Histograms</div>
          {stats?.visuals.histograms
            ? <Histogram histograms={stats.visuals.histograms} />
            : <p className="faint">Histogram data unavailable.</p>}
          {stats && (
            <dl className="kv" style={{ marginTop: 12 }}>
              <div style={{ display: 'contents' }}><dt>Mean RGB</dt><dd>{Array.isArray(stats.technical.mean_rgb) ? stats.technical.mean_rgb.join(', ') : 'unavailable'}</dd></div>
              <div style={{ display: 'contents' }}><dt>Luminance entropy</dt><dd>{isFiniteNumber(stats.technical.luminance_entropy_bits) ? `${stats.technical.luminance_entropy_bits} bits` : 'unavailable'}</dd></div>
              <div style={{ display: 'contents' }}><dt>Contrast (RMS)</dt><dd>{fmtNum(stats.technical.contrast_rms)}</dd></div>
              <div style={{ display: 'contents' }}><dt>Edge density</dt><dd>{fmtFractionPct(stats.technical.edge_density, 2)}</dd></div>
            </dl>
          )}
        </div>
        <div className="panel panel-pad">
          <div className="eyebrow">Frequency spectrum</div>
          {freq?.visuals.radial_profile
            ? <RadialChart profile={freq.visuals.radial_profile} />
            : <p className="faint">Spectrum data unavailable.</p>}
          {freq && (
            <dl className="kv" style={{ marginTop: 12 }}>
              <div style={{ display: 'contents' }}><dt>Spectral slope</dt><dd>{fmtNum(freq.technical.spectral_slope)}</dd></div>
              <div style={{ display: 'contents' }}><dt>Spectral peaks</dt><dd>{fmtNum(freq.technical.spectral_peak_count, 0)}</dd></div>
              <div style={{ display: 'contents' }}><dt>High-freq energy</dt><dd>{fmtFractionPct(freq.technical.high_freq_energy, 2)}</dd></div>
            </dl>
          )}
        </div>
      </div>

      {/* ------- ML detector info ------- */}
      <h2 className="section-title">ML Detector Information</h2>
      <p className="section-sub">Trained models are one signal among many. Raw scores are never shown as probabilities without documented calibration.</p>
      {e.ml_suppressed_reason ? (
        <div className="panel panel-pad">
          <dl className="kv">
            <div style={{ display: 'contents' }}><dt>Trained model</dt><dd>{e.detector.model_name ?? e.detector.detector_name ?? 'unavailable'}</dd></div>
            <div style={{ display: 'contents' }}><dt>Status</dt><dd>ML probability withheld</dd></div>
            <div style={{ display: 'contents' }}><dt>Reason</dt><dd style={{ fontFamily: 'var(--font)' }}>{e.ml_suppressed_reason}</dd></div>
          </dl>
          <p className="faint" style={{ fontSize: 12.5, marginTop: 8 }}>
            The trained model is configured, but its output is not presented as a calibrated probability.
            Verdict from local forensic signals only.
          </p>
        </div>
      ) : (e.detectors?.filter((d) => d.available).length ?? 0) > 0 ? (
        <div className="grid grid-2">
          {e.detectors.filter((d) => d.available).map((d, i) => (
            <div key={i} className="panel panel-pad">
              <div className="eyebrow">{d.detector_name} <span className="faint">v{d.detector_version} · {d.model_type} · {d.model_status}</span></div>
              <dl className="kv" style={{ marginTop: 8 }}>
                <div style={{ display: 'contents' }}><dt>Raw model score</dt><dd className="mono">{d.raw_score}</dd></div>
                <div style={{ display: 'contents' }}><dt>Calibrated P(AI)</dt><dd><strong>{fmtPct(d.probability_ai, 1)}</strong></dd></div>
                <div style={{ display: 'contents' }}><dt>Calibration</dt><dd className="mono" style={{ fontSize: 12 }}>{d.calibration}</dd></div>
                <div style={{ display: 'contents' }}><dt>Confidence</dt><dd>{fmtPct(d.confidence, 0)}</dd></div>
                <div style={{ display: 'contents' }}><dt>Inference time</dt><dd>{d.inference_time_ms} ms</dd></div>
                {d.generator_label && <div style={{ display: 'contents' }}><dt>Generator label</dt><dd>{d.generator_label}</dd></div>}
                <div style={{ display: 'contents' }}><dt>Preprocessing</dt><dd className="mono" style={{ fontSize: 11.5 }}>
                  {(d.preprocessing?.steps ?? []).map((s: any) => s.step).join(' → ') || 'n/a'}
                  {d.preprocessing?.model_input_dimensions ? ` → input ${d.preprocessing.model_input_dimensions.join('×')}` : ''}
                </dd></div>
                <div style={{ display: 'contents' }}><dt>Limitations</dt><dd style={{ fontFamily: 'var(--font)', fontSize: 12.5 }}>{d.limitations}</dd></div>
              </dl>
            </div>
          ))}
        </div>
      ) : (
        <div className="panel panel-pad">
          <dl className="kv">
            <div style={{ display: 'contents' }}><dt>Trained model</dt><dd>{e.detector.model_name ?? e.detector.detector_name ?? 'unavailable'}</dd></div>
            <div style={{ display: 'contents' }}><dt>Status</dt><dd>MODEL_NOT_CONFIGURED — verdict from local forensic signals only</dd></div>
            <div style={{ display: 'contents' }}><dt>Limitations</dt><dd style={{ fontFamily: 'var(--font)' }}>{e.detector.limitations}</dd></div>
          </dl>
          {e.ml_disagreement > 0 && (
            <p className="faint" style={{ fontSize: 12.5, marginTop: 8 }}>
              ML/forensics disagreement: {e.ml_disagreement} points (penalized in confidence).
            </p>
          )}
        </div>
      )}
      {e.hybrid_weights && (
        <div className="panel panel-pad" style={{ marginTop: 12 }}>
          <div className="eyebrow">Hybrid ensemble weights</div>
          <div style={{ display: 'flex', gap: 20, marginTop: 8, flexWrap: 'wrap' }}>
            <span>Forensics <strong>{e.hybrid_weights.forensics}</strong> → P(AI) {fmtPct(e.forensic_ai_probability, 1)}</span>
            <span>ML <strong>{e.hybrid_weights.ml}</strong>{e.ml_suppressed_reason ? ' → withheld' : (e.ml_ai_probability !== null ? ` → P(AI) ${e.ml_ai_probability}%` : ' → not active')}</span>
            <span>Provenance <strong>{e.hybrid_weights.provenance}</strong> → {e.provenance?.note}</span>
          </div>
        </div>
      )}

      {/* ------- scorecard + file info ------- */}
      <h2 className="section-title">Forensic Scorecard</h2>
      <div className="grid grid-2">
        <Scorecard report={report} />
        <div className="panel panel-pad">
          <div className="eyebrow">Technical image information</div>
          <dl className="kv">
            <div style={{ display: 'contents' }}><dt>Filename</dt><dd>{report.filename}</dd></div>
            <div style={{ display: 'contents' }}><dt>File size</dt><dd>{fmtBytes(report.file_size)}</dd></div>
            <div style={{ display: 'contents' }}><dt>Dimensions</dt><dd>{report.width} × {report.height}</dd></div>
            <div style={{ display: 'contents' }}><dt>Image quality</dt><dd>{quality?.technical.quality_tier ? `${quality.technical.quality_tier} (${isFiniteNumber(quality.technical.megapixels) ? `${quality.technical.megapixels} MP` : 'unavailable'})` : 'unavailable'}</dd></div>
            <div style={{ display: 'contents' }}><dt>Analysis time</dt><dd>{fmtNum(report.pipeline_ms, 0)} ms</dd></div>
            <div style={{ display: 'contents' }}><dt>Signals agree</dt><dd>{e.signals_agree ? 'Yes' : (isFiniteNumber(e.signal_disagreement) ? `No (spread ${e.signal_disagreement.toFixed(0)})` : 'No (spread unavailable)')}</dd></div>
          </dl>
          <div style={{ marginTop: 16 }}>
            <button className="btn btn-sm" onClick={() => setShowReports(!showReports)}>
              {showReports ? 'Hide export options' : 'Export forensic report'}
            </button>
            {showReports && !report.demo && (
              <div style={{ display: 'flex', gap: 8, marginTop: 10, flexWrap: 'wrap' }}>
                <a className="btn btn-sm" href={reportUrl(report.analysis_id, 'pdf')} download>PDF</a>
                <a className="btn btn-sm" href={reportUrl(report.analysis_id, 'json')} download>JSON</a>
                <a className="btn btn-sm" href={reportUrl(report.analysis_id, 'csv')} download>CSV</a>
              </div>
            )}
            {showReports && report.demo && (
              <p className="faint" style={{ fontSize: 12.5, marginTop: 8 }}>Reports are disabled for demo results.</p>
            )}
            {!report.demo && (
              <p className="faint" style={{ fontSize: 12, marginTop: 8 }}>
                Reports require the analysis to be saved to history (enable “Save to history” before analyzing).
              </p>
            )}
          </div>
        </div>
      </div>

      {/* ------- reproducibility ------- */}
      <h2 className="section-title">Reproducibility</h2>
      <p className="section-sub">Everything needed to reproduce this exact report. The original image is not stored.</p>
      <div className="panel panel-pad">
        <dl className="kv">
          <div style={{ display: 'contents' }}><dt>Analysis ID</dt><dd className="mono">{report.reproducibility?.analysis_id}</dd></div>
          <div style={{ display: 'contents' }}><dt>Report hash</dt><dd className="mono" style={{ fontSize: 11.5, wordBreak: 'break-all' }}>{report.report_hash}</dd></div>
          <div style={{ display: 'contents' }}><dt>Image SHA-256</dt><dd className="mono" style={{ fontSize: 11.5, wordBreak: 'break-all' }}>{report.reproducibility?.image_sha256}</dd></div>
          <div style={{ display: 'contents' }}><dt>App / backend</dt><dd className="mono">v{report.reproducibility?.app_version} / v{report.reproducibility?.backend_version}</dd></div>
          <div style={{ display: 'contents' }}><dt>Config</dt><dd className="mono">v{report.reproducibility?.config_version} · sha256 {report.reproducibility?.config_sha256}</dd></div>
          <div style={{ display: 'contents' }}><dt>Detectors</dt><dd style={{ fontSize: 12.5 }}>
            {(report.reproducibility?.detectors ?? []).map((d) => `${d.name} v${d.version} (${d.status})`).join('; ') || 'none configured'}
          </dd></div>
          <div style={{ display: 'contents' }}><dt>Analyzed at</dt><dd>{report.reproducibility?.analyzed_at}</dd></div>
        </dl>
        {!report.demo && (
          <a className="btn btn-sm" href={`/research/${report.analysis_id}`} style={{ marginTop: 12, display: 'inline-block' }}>
            Open in research mode
          </a>
        )}
      </div>

      {/* ------- limitations ------- */}
      <h2 className="section-title">Limitations</h2>
      <div className="panel panel-pad">
        {report.limitations.map((l, i) => (
          <div key={i} className="evidence-item">
            <span className="ev-dot" style={{ color: 'var(--warn)' }}>!</span><span className="dim">{l}</span>
          </div>
        ))}
      </div>
    </div>
  );
}
