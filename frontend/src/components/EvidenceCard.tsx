import { useState } from 'react';
import type { Signal } from '../types';

const DIR_ICON: Record<string, { icon: string; color: string }> = {
  supports_ai: { icon: '▲', color: '#f43f5e' },
  supports_authentic: { icon: '●', color: '#22c55e' },
  supports_manip: { icon: '◆', color: '#f59e0b' },
  neutral: { icon: '○', color: '#94a3b8' },
  uncertain: { icon: '?', color: '#f59e0b' },
};

const DIR_LABEL: Record<string, string> = {
  supports_ai: 'Consistent with AI generation',
  supports_authentic: 'Consistent with authentic capture',
  supports_manip: 'Consistent with manipulation',
  neutral: 'Context',
  uncertain: 'Uncertain',
};

export default function EvidenceCard({ signal, learn }: { signal: Signal; learn?: { what: string; why: string; limits: string } }) {
  const [open, setOpen] = useState(false);
  const statusColor = signal.status === 'ok' ? '#22c55e' : signal.status === 'warning' ? '#f59e0b' : '#f43f5e';

  return (
    <div className="card" style={{ display: 'flex', flexDirection: 'column' }}>
      <div style={{ display: 'flex', alignItems: 'flex-start', justifyContent: 'space-between', gap: 8 }}>
        <h3 style={{ margin: 0 }}>{signal.display_name}</h3>
        <span className="tag" style={{ color: statusColor, borderColor: statusColor + '66' }}>
          {signal.status === 'ok' ? '✓' : signal.status === 'warning' ? '!' : '×'} {signal.status}
        </span>
      </div>

      <div style={{ display: 'flex', gap: 14, margin: '10px 0 4px', fontSize: 12.5 }}>
        <span title="How strongly this signal suggests AI generation">
          AI <strong className="mono" style={{ color: signal.ai_score >= 60 ? '#f43f5e' : 'var(--text)' }}>{signal.ai_score.toFixed(0)}</strong>
        </span>
        <span title="How strongly this signal suggests manipulation">
          Manip <strong className="mono" style={{ color: signal.manip_score >= 60 ? '#f59e0b' : 'var(--text)' }}>{signal.manip_score.toFixed(0)}</strong>
        </span>
        <span title="How reliable this signal is for this image">
          Conf <strong className="mono">{signal.confidence.toFixed(0)}%</strong>
        </span>
        <span className="faint mono">{signal.duration_ms.toFixed(0)}ms</span>
      </div>

      <div style={{ marginTop: 6 }}>
        {signal.evidence.slice(0, open ? undefined : 3).map((e, i) => {
          const d = DIR_ICON[e.direction] || DIR_ICON.neutral;
          return (
            <div key={i} className="evidence-item" title={DIR_LABEL[e.direction]}>
              <span className="ev-dot" style={{ color: d.color }} aria-hidden="true">{d.icon}</span>
              <span>
                {e.text}
                {e.detail && open && <div className="faint" style={{ fontSize: 12, marginTop: 2 }}>{e.detail}</div>}
              </span>
            </div>
          );
        })}
      </div>

      {signal.warnings.map((w, i) => (
        <div key={i} className="alert warn" style={{ marginTop: 10, padding: '8px 12px' }}>⚠ {w}</div>
      ))}

      <button className="btn btn-ghost btn-sm" style={{ marginTop: 10, alignSelf: 'flex-start' }}
        onClick={() => setOpen(!open)} aria-expanded={open}>
        {open ? 'Show less' : `Technical details (${signal.evidence.length} findings)`}
      </button>

      {open && (
        <div style={{ marginTop: 6 }}>
          {Object.keys(signal.technical).length > 0 && (
            <details className="expandable" open>
              <summary>Technical details</summary>
              <div className="body">
                <dl className="kv">
                  {Object.entries(signal.technical).map(([k, v]) => (
                    <div key={k} style={{ display: 'contents' }}>
                      <dt>{k.replace(/_/g, ' ')}</dt>
                      <dd>{typeof v === 'object' ? JSON.stringify(v).slice(0, 220) : String(v)}</dd>
                    </div>
                  ))}
                </dl>
              </div>
            </details>
          )}
          {learn && (
            <details className="expandable">
              <summary>Learn: what is {signal.display_name.toLowerCase()}?</summary>
              <div className="body">
                <p><strong>What it is:</strong> {learn.what}</p>
                <p><strong>Why it matters:</strong> {learn.why}</p>
                <p><strong>Limitations:</strong> {learn.limits}</p>
              </div>
            </details>
          )}
        </div>
      )}
    </div>
  );
}
