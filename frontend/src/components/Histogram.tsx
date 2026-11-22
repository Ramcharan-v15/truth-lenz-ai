import { useState } from 'react';

const COLORS: Record<string, string> = {
  r: '#f43f5e', g: '#22c55e', b: '#38bdf8', luminance: '#cbd5e1', rgb: '#94a3b8',
};

export default function Histogram({ histograms }: { histograms: Record<string, { counts: number[]; bins: number[] }> }) {
  const [mode, setMode] = useState<'rgb' | 'r' | 'g' | 'b' | 'luminance'>('rgb');
  const W = 560, H = 190, PAD = 8;
  const modes = (['rgb', 'r', 'g', 'b', 'luminance'] as const).filter((m) => histograms[m]);

  const render = (key: string, color: string, opacity: number) => {
    const h = histograms[key];
    if (!h) return null;
    const max = Math.max(...h.counts, 1);
    const bw = (W - PAD * 2) / h.counts.length;
    return (
      <g key={key}>
        {h.counts.map((c, i) => {
          const bh = (c / max) * (H - 30);
          return (
            <rect
              key={i} className="hist-bar"
              x={PAD + i * bw} y={H - 18 - bh} width={Math.max(0.5, bw - 0.6)} height={bh}
              fill={color} opacity={opacity}
            >
              <title>{`bin ${h.bins[i].toFixed(0)}: ${c} px`}</title>
            </rect>
          );
        })}
      </g>
    );
  };

  return (
    <div>
      <div className="viewer-tabs" role="tablist" aria-label="Histogram channel">
        {modes.map((m) => (
          <button key={m} role="tab" aria-selected={mode === m} className={mode === m ? 'active' : ''}
            onClick={() => setMode(m)}>
            {m === 'rgb' ? 'RGB' : m === 'luminance' ? 'Luminance' : m.toUpperCase()}
          </button>
        ))}
      </div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', background: 'var(--panel-solid)', borderRadius: 8, border: '1px solid var(--border)' }} role="img" aria-label={`${mode} histogram`}>
        {mode === 'rgb'
          ? (['r', 'g', 'b'] as const).map((k) => render(k, COLORS[k], 0.55))
          : render(mode, COLORS[mode], 0.9)}
        <line x1={PAD} y1={H - 18} x2={W - PAD} y2={H - 18} stroke="var(--border-strong)" />
        {[0, 64, 128, 192, 255].map((t) => (
          <text key={t} x={PAD + (t / 255) * (W - PAD * 2)} y={H - 5} fill="var(--text-faint)" fontSize="9" textAnchor="middle">{t}</text>
        ))}
      </svg>
      <p className="faint" style={{ fontSize: 12, marginTop: 8 }}>
        {mode === 'luminance'
          ? 'Luminance histogram: spikes at the extremes suggest clipping; gaps can indicate heavy processing.'
          : 'Channel histogram: look for comb-like gaps (heavy quantization) or unnatural spikes.'}
      </p>
    </div>
  );
}

/** Radial frequency profile chart. */
export function RadialChart({ profile }: { profile: { frequencies: number[]; power: number[] } }) {
  const W = 560, H = 190, PAD = 30;
  const max = Math.max(...profile.power, 1e-9);
  const pts = profile.power.map((p, i) => {
    const x = PAD + (i / (profile.power.length - 1)) * (W - PAD * 2);
    const y = H - 24 - (p / max) * (H - 50);
    return `${x.toFixed(1)},${y.toFixed(1)}`;
  }).join(' ');
  return (
    <div>
      <svg viewBox={`0 0 ${W} ${H}`} style={{ width: '100%', background: 'var(--panel-solid)', borderRadius: 8, border: '1px solid var(--border)' }} role="img" aria-label="Radial frequency power profile">
        <polyline points={pts} fill="none" stroke="#38bdf8" strokeWidth="2" />
        <line x1={PAD} y1={H - 24} x2={W - PAD} y2={H - 24} stroke="var(--border-strong)" />
        <text x={PAD} y={H - 8} fill="var(--text-faint)" fontSize="10">low freq</text>
        <text x={W - PAD} y={H - 8} fill="var(--text-faint)" fontSize="10" textAnchor="end">high freq</text>
        <text x={PAD} y={16} fill="var(--text-faint)" fontSize="10">relative power (log scale visual)</text>
      </svg>
      <p className="faint" style={{ fontSize: 12, marginTop: 8 }}>
        Natural images decay smoothly from low to high frequencies. Sharp bumps in the mid/high
        range can indicate generative upsampling artifacts.
      </p>
    </div>
  );
}
