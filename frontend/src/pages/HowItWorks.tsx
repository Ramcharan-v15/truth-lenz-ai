import { useEffect, useState } from 'react';
import { getConfig } from '../api';

const STEPS = [
  { t: 'Upload', d: 'The image is validated by its actual file bytes — never by filename or claimed type — then decoded safely in memory.' },
  { t: 'Inspect', d: 'Dimensions, format, color model and basic statistics are extracted; histograms and quality metrics are computed.' },
  { t: 'Extract signals', d: 'Ten independent analyzers run: metadata, statistics, compression, noise, frequency, pixel forensics, AI artifacts, manipulation, quality and consistency.' },
  { t: 'Analyze', d: 'Each analyzer returns structured evidence — scores, confidence, findings, warnings and technical details. One analyzer can never veto the others.' },
  { t: 'Cross-check', d: 'Signals are compared for agreement. A lone strong finding with no contradiction is a finding; genuinely conflicting signals raise uncertainty.' },
  { t: 'Ensemble scoring', d: 'Opinion-weighted evidence combines the signals using configurable weights. Neutral signals are down-weighted; decisive ones count more.' },
  { t: 'Explain', d: 'The verdict ships with why-it-won reasoning: strongest evidence, evidence against, uncertain factors and confidence penalties.' },
  { t: 'Report', d: 'Export a forensic report (PDF/JSON/CSV) with branding, evidence trail, limitations, timestamp and analysis ID.' },
];

export default function HowItWorks() {
  const [config, setConfig] = useState<any>(null);
  useEffect(() => { getConfig().then(setConfig).catch(() => {}); }, []);

  return (
    <div className="shell" style={{ paddingTop: 30, paddingBottom: 20 }}>
      <div className="eyebrow">UNDER THE HOOD</div>
      <h1 style={{ margin: '0 0 8px', fontSize: 30 }}>How It Works</h1>
      <p className="section-sub">
        Why multiple signals beat one detector: every detection method has blind spots. Metadata can be
        stripped, frequency fingerprints fade as generators improve, and heuristics misfire on unusual
        photos. Independent signals with different blind spots cross-check each other — and the system
        says <em>inconclusive</em> when they genuinely disagree instead of guessing.
      </p>

      <div className="grid grid-2" style={{ marginTop: 8 }}>
        {STEPS.map((s, i) => (
          <div key={s.t} className="card">
            <div style={{ display: 'flex', gap: 12, alignItems: 'flex-start' }}>
              <span className="tag" style={{ color: 'var(--accent)', borderColor: 'var(--accent)', flexShrink: 0 }}>{i + 1}</span>
              <div><h3 style={{ margin: '0 0 4px' }}>{s.t}</h3><p className="dim" style={{ margin: 0, fontSize: 13.5 }}>{s.d}</p></div>
            </div>
          </div>
        ))}
      </div>

      <h2 className="section-title">Probability vs confidence</h2>
      <div className="panel panel-pad">
        <p style={{ margin: 0, fontSize: 14 }}>
          <strong>Probability</strong> answers “how AI-like does this look?” <strong>Confidence</strong> answers
          “how much should you trust that number?” An 84% AI probability with <em>low</em> confidence
          (poor quality, heavy compression, missing metadata) means far less than 68% with <em>high</em> confidence.
          Never read “84% confidence” as “84% AI” — they are different concepts, shown separately everywhere.
        </p>
      </div>

      {config && (
        <>
          <h2 className="section-title">Live detector configuration</h2>
          <p className="section-sub">Weights, thresholds and penalties currently driving the ensemble — editable in <span className="mono">backend/detector_config.json</span> without code changes.</p>
          <div className="panel panel-pad" style={{ overflowX: 'auto' }}>
            <table className="data">
              <thead><tr><th>Analyzer</th><th>AI weight</th><th>Manipulation weight</th><th>Rationale</th></tr></thead>
              <tbody>
                {Object.entries(config.weights || {}).filter(([, v]: any) => typeof v === 'object').map(([k, v]: any) => (
                  <tr key={k}><td className="mono">{k}</td><td className="mono">{v.ai}</td><td className="mono">{v.manip}</td><td className="dim" style={{ fontSize: 12.5 }}>{v.note}</td></tr>
                ))}
              </tbody>
            </table>
            <div className="grid grid-2" style={{ marginTop: 16 }}>
              <div>
                <div className="eyebrow">Verdict thresholds</div>
                <dl className="kv">
                  {Object.entries(config.verdict_thresholds || {}).filter(([, v]) => typeof v === 'number').map(([k, v]) => (
                    <div key={k} style={{ display: 'contents' }}><dt>{k.replace(/_/g, ' ')}</dt><dd>{String(v)}</dd></div>
                  ))}
                </dl>
              </div>
              <div>
                <div className="eyebrow">Confidence penalties</div>
                <dl className="kv">
                  {Object.entries(config.confidence?.penalties || {}).map(([k, v]) => (
                    <div key={k} style={{ display: 'contents' }}><dt>{k.replace(/_/g, ' ')}</dt><dd>−{String(v)}</dd></div>
                  ))}
                </dl>
              </div>
            </div>
            {config.ensemble_v2 && (
              <div style={{ marginTop: 16 }}>
                <div className="eyebrow">Hybrid ensemble (V2)</div>
                <dl className="kv" style={{ marginTop: 8 }}>
                  {Object.entries(config.ensemble_v2).filter(([, v]) => typeof v !== 'object').map(([k, v]) => (
                    <div key={k} style={{ display: 'contents' }}><dt>{k.replace(/_/g, ' ')}</dt><dd>{String(v)}</dd></div>
                  ))}
                </dl>
                <p className="faint" style={{ fontSize: 12.5, marginTop: 8 }}>
                  P(AI) blends forensic signals, ML detector outputs and provenance with these weights.
                  ML can never outweigh forensics unless <span className="mono">allow_ml_dominance</span> is
                  explicitly enabled. Raw model scores pass through a documented calibration layer —
                  uncalibrated outputs are labelled as such and their influence is capped.
                </p>
              </div>
            )}
          </div>
        </>
      )}
    </div>
  );
}
