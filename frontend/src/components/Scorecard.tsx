import type { AnalysisReport } from '../types';
import { VERDICT_META, fmtPct, isFiniteNumber } from '../types';

/** Final forensic scorecard - visual professional summary.
 * Stage 1: every row renders 'unavailable' for a missing/non-finite value and
 * draws no bar for it. Missing values are never coerced to zero. */
export default function Scorecard({ report }: { report: AnalysisReport }) {
  const e = report.ensemble;
  const meta = VERDICT_META[e.verdict];
  const quality = report.signals.find((s) => s.name === 'quality');
  const metadata = report.signals.find((s) => s.name === 'metadata');

  const rows: { label: string; value: string; bar?: number; color?: string }[] = [
    { label: 'AI Generation', value: fmtPct(e.ai_probability, 1), bar: isFiniteNumber(e.ai_probability) ? e.ai_probability : undefined, color: '#f43f5e' },
    { label: 'Manipulation', value: fmtPct(e.manipulation_probability, 1), bar: isFiniteNumber(e.manipulation_probability) ? e.manipulation_probability : undefined, color: '#f59e0b' },
    { label: 'Camera-like capture', value: fmtPct(e.authenticity_probability, 1), bar: isFiniteNumber(e.authenticity_probability) ? e.authenticity_probability : undefined, color: '#22c55e' },
    { label: 'Metadata quality', value: metadata ? fmtPct(metadata.confidence, 0) : '—', bar: metadata && isFiniteNumber(metadata.confidence) ? metadata.confidence : undefined, color: '#38bdf8' },
    { label: 'Image quality', value: quality?.technical.quality_tier?.toUpperCase() || '—' },
    { label: 'Confidence', value: `${fmtPct(e.confidence, 0)} (${e.confidence_label})`, bar: isFiniteNumber(e.confidence) ? e.confidence : undefined, color: '#a78bfa' },
  ];

  return (
    <div className="panel panel-pad" style={{ borderTop: `3px solid ${meta.color}` }}>
      <div className="eyebrow">Truth Lenz Forensic Score</div>
      <div style={{ display: 'flex', alignItems: 'center', gap: 12, marginBottom: 14 }}>
        <span style={{ fontSize: 26 }}>{meta.icon}</span>
        <div>
          <div style={{ fontWeight: 800, fontSize: 17, color: meta.color }}>{meta.label}</div>
          <div className="faint mono" style={{ fontSize: 11.5 }}>ID {report.analysis_id} · {report.analyzed_at}</div>
        </div>
      </div>
      {rows.map((r) => (
        <div key={r.label} style={{ display: 'flex', alignItems: 'center', gap: 12, padding: '7px 0', borderBottom: '1px solid var(--border)' }}>
          <span className="dim" style={{ width: 150, fontSize: 13 }}>{r.label}</span>
          {r.bar !== undefined && (
            <div className="progress" style={{ flex: 1 }}>
              <div style={{ width: `${r.bar}%`, background: r.color }} />
            </div>
          )}
          <span className="mono" style={{ width: r.bar !== undefined ? 110 : undefined, textAlign: 'right', fontSize: 13, fontWeight: 700 }}>{r.value}</span>
        </div>
      ))}
      <div className="faint" style={{ fontSize: 11.5, marginTop: 10 }}>
        Probability estimates — not definitive proof.
      </div>
    </div>
  );
}
