import { useEffect, useState } from 'react';
import { isFiniteNumber } from '../types';

/** Semicircular gauge for a 0-100 probability. Labelled as estimate, never certainty.
 * Stage 1: a null/undefined/non-finite value renders as 'unavailable' — never a
 * zero, never a crash. The arc stays empty and no probability is implied. */
export default function Gauge({
  value,
  label,
  color = '#38bdf8',
  sub,
}: {
  value: number | null | undefined;
  label: string;
  color?: string;
  sub?: string;
}) {
  const [v, setV] = useState(0);
  useEffect(() => {
    if (!isFiniteNumber(value)) return;
    const t = setTimeout(() => setV(value), 60);
    const id = requestAnimationFrame(function tick() {});
    cancelAnimationFrame(id);
    return () => clearTimeout(t);
  }, [value]);

  const available = isFiniteNumber(value);
  const R = 80;
  const circ = Math.PI * R;
  const frac = Math.max(0, Math.min(100, v)) / 100;

  return (
    <div style={{ textAlign: 'center' }} role="img" aria-label={available ? `${label}: ${value.toFixed(0)} percent (estimate)` : `${label}: unavailable`}>
      <svg width="200" height="118" viewBox="0 0 200 118">
        <path d={`M 20 105 A ${R} ${R} 0 0 1 180 105`} fill="none" stroke="var(--border)" strokeWidth="14" strokeLinecap="round" />
        <path
          d={`M 20 105 A ${R} ${R} 0 0 1 180 105`}
          fill="none" stroke={color} strokeWidth="14" strokeLinecap="round"
          strokeDasharray={`${frac * circ} ${circ}`}
          style={{ transition: 'stroke-dasharray 1.1s cubic-bezier(0.22,1,0.36,1)', filter: `drop-shadow(0 0 6px ${color}55)` }}
        />
        {available ? (
          <text x="100" y="88" textAnchor="middle" fill="var(--text)" fontSize="30" fontWeight="800" fontFamily="var(--font)">
            {v.toFixed(0)}%
          </text>
        ) : (
          <text x="100" y="88" textAnchor="middle" fill="var(--text-faint)" fontSize="16" fontWeight="700" fontFamily="var(--font)">
            unavailable
          </text>
        )}
        <text x="100" y="106" textAnchor="middle" fill="var(--text-faint)" fontSize="10" letterSpacing="1.5">
          ESTIMATE
        </text>
      </svg>
      <div className="metric" style={{ textAlign: 'center', marginTop: 2 }}>
        <div className="lbl">{label}</div>
        {sub && <div className="faint" style={{ fontSize: 11.5, marginTop: 4 }}>{sub}</div>}
      </div>
    </div>
  );
}

/** Small horizontal probability bar. Same null contract as Gauge. */
export function ProbBar({ value, color = '#38bdf8', label }: { value: number | null | undefined; color?: string; label: string }) {
  const available = isFiniteNumber(value);
  return (
    <div style={{ margin: '8px 0' }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', fontSize: 12.5, marginBottom: 4 }}>
        <span className="dim">{label}</span>
        <span className="mono" style={{ color }}>{available ? `${value.toFixed(1)}%` : 'unavailable'}</span>
      </div>
      <div className="progress" role="img" aria-label={available ? `${label} ${value.toFixed(1)} percent` : `${label} unavailable`}>
        <div style={{ width: `${available ? Math.max(0, Math.min(100, value)) : 0}%`, background: color }} />
      </div>
    </div>
  );
}
