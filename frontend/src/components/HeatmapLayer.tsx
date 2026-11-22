import { useEffect, useMemo, useRef, useState } from 'react';
import type { AnalysisReport } from '../types';

interface Layer {
  id: string;
  label: string;
  kind: 'image' | 'grid';
  src?: string;       // data URI
  grid?: number[][];  // normalized 0..1
  note: string;
}

/**
 * Unified forensic visualization layer (Phase 15).
 * Switch between ELA / Noise / Frequency / Manipulation / ML attention /
 * Copy-move / Splicing. Legend, opacity, zoom, download, side-by-side.
 * Visualizations highlight regions of interest — they never imply certainty.
 */
export default function HeatmapLayer({ report }: { report: AnalysisReport }) {
  const layers = useMemo<Layer[]>(() => {
    const sig = (n: string) => report.signals.find((s) => s.name === n);
    const out: Layer[] = [];
    const ela = sig('pixel')?.visuals?.ela;
    if (ela) out.push({ id: 'ela', label: 'ELA', kind: 'image', src: ela, note: 'Error-level response. Bright areas differ in compression history — they do not prove editing.' });
    const nmap = sig('noise')?.visuals?.noise_map?.grid;
    if (nmap) out.push({ id: 'noise', label: 'Noise', kind: 'grid', grid: nmap, note: 'Estimated noise per block. Inconsistent noise can suggest compositing; it is not proof.' });
    const spec = sig('frequency')?.visuals?.spectrum;
    if (spec) out.push({ id: 'frequency', label: 'Frequency', kind: 'image', src: spec, note: 'Fourier spectrum of the luminance channel.' });
    const mov = sig('manipulation')?.visuals?.manipulation_overlay;
    if (mov) {
      out.push({ id: 'manipulation', label: 'Manipulation', kind: 'image', src: mov, note: 'Combined anomaly overlay from copy-move, texture and edge analysis.' });
      out.push({ id: 'copymove', label: 'Copy-Move', kind: 'image', src: mov, note: 'Same overlay as Manipulation — copy-move evidence is derived from it, not separately localized.' });
      if (ela) out.push({ id: 'splicing', label: 'Splicing', kind: 'image', src: ela, note: 'Splicing view reuses the ELA response: spliced regions often compress differently.' });
    }
    const dets = report.ensemble.detectors ?? [];
    const attn = dets.find((d) => d.attention_heatmap)?.attention_heatmap;
    if (attn) out.push({ id: 'ml', label: 'ML Attention', kind: 'image', src: attn, note: 'Model attention visualization — shows where the model looked, NOT proof of AI generation.' });
    return out;
  }, [report]);

  const [activeId, setActiveId] = useState(layers[0]?.id ?? '');
  const [opacity, setOpacity] = useState(0.65);
  const [zoom, setZoom] = useState(1);
  const [sideBySide, setSideBySide] = useState(false);
  const active = layers.find((l) => l.id === activeId) ?? layers[0];

  useEffect(() => {
    if (!activeId && layers[0]) setActiveId(layers[0].id);
  }, [layers, activeId]);

  if (!active) {
    return (
      <div style={{ border: '1px solid var(--border)', borderRadius: 12, padding: 16, background: 'var(--card)' }}>
        <b>Forensic visualizations</b>
        <div style={{ fontSize: 13, color: 'var(--muted)', marginTop: 6 }}>No heatmap layers available for this analysis.</div>
      </div>
    );
  }

  const download = () => {
    const a = document.createElement('a');
    if (active.kind === 'image' && active.src) {
      a.href = active.src;
    } else if (active.grid) {
      a.href = gridToDataUri(active.grid);
    } else return;
    a.download = `truthlenz-${report.analysis_id}-${active.id}.png`;
    a.click();
  };

  const view = (key: string, showBase: boolean) => (
    <div key={key} style={{ position: 'relative', overflow: 'hidden', borderRadius: 8, background: '#000', flex: 1, minWidth: 0 }}>
      <div style={{ transform: `scale(${zoom})`, transformOrigin: 'center', transition: 'transform 0.15s' }}>
        {showBase && (
          <img src={report.preview} alt="original" style={{ width: '100%', display: 'block' }} />
        )}
        {active.kind === 'image' && active.src ? (
          <img
            src={active.src} alt={`${active.label} heatmap`}
            style={{
              width: '100%', display: 'block',
              position: showBase ? 'absolute' : 'static', inset: 0,
              opacity: showBase ? opacity : 1, mixBlendMode: showBase ? 'screen' : 'normal',
            }}
          />
        ) : active.grid ? (
          <GridHeatmap grid={active.grid} overlay={showBase} opacity={opacity} />
        ) : null}
      </div>
    </div>
  );

  return (
    <div style={{ border: '1px solid var(--border)', borderRadius: 12, background: 'var(--card)', padding: 16 }}>
      <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', flexWrap: 'wrap', gap: 8 }}>
        <div>
          <div style={{ fontWeight: 700 }}>Forensic visualizations</div>
          <div style={{ fontSize: 12, color: 'var(--muted)' }}>Regions of interest — brighter response means stronger local signal, never certainty.</div>
        </div>
        <div style={{ display: 'flex', gap: 8, alignItems: 'center', flexWrap: 'wrap' }}>
          <label style={{ fontSize: 12, color: 'var(--muted)' }}>
            Opacity <input type="range" min={0} max={1} step={0.05} value={opacity} onChange={(e) => setOpacity(Number(e.target.value))} style={{ verticalAlign: 'middle', width: 90 }} />
          </label>
          <button className="btn-ghost" onClick={() => setZoom((z) => Math.min(3, +(z + 0.25).toFixed(2)))}>Zoom +</button>
          <button className="btn-ghost" onClick={() => setZoom((z) => Math.max(0.5, +(z - 0.25).toFixed(2)))}>Zoom −</button>
          <button className="btn-ghost" onClick={() => setSideBySide((v) => !v)}>{sideBySide ? 'Overlay' : 'Side-by-side'}</button>
          <button className="btn-ghost" onClick={download}>Download</button>
        </div>
      </div>

      <div style={{ display: 'flex', gap: 6, marginTop: 12, flexWrap: 'wrap' }}>
        {layers.map((l) => (
          <button
            key={l.id}
            onClick={() => setActiveId(l.id)}
            style={{
              padding: '6px 12px', borderRadius: 8, fontSize: 12, fontWeight: 600,
              border: `1px solid ${l.id === active.id ? '#38bdf8' : 'var(--border)'}`,
              background: l.id === active.id ? 'rgba(56,189,248,0.12)' : 'transparent',
              color: 'var(--text)', cursor: 'pointer',
            }}
          >
            {l.label}
          </button>
        ))}
        {!report.ensemble.detectors?.some((d) => d.attention_heatmap) && (
          <span style={{ fontSize: 11, color: 'var(--muted)', alignSelf: 'center', marginLeft: 4 }}>
            ML attention: unavailable (no model with explainability connected)
          </span>
        )}
      </div>

      <div style={{ display: 'flex', gap: 12, marginTop: 12 }}>
        {sideBySide ? (
          <>{view('base', true)}{view('heat', false)}</>
        ) : (
          view('overlay', true)
        )}
      </div>

      <div style={{ marginTop: 12, display: 'flex', alignItems: 'center', gap: 10 }}>
        <span style={{ fontSize: 11, color: 'var(--muted)' }}>lower response</span>
        <div style={{ flex: 1, height: 10, borderRadius: 5, background: 'linear-gradient(to right, #1e3a8a, #06b6d4, #22c55e, #f59e0b, #ef4444)' }} />
        <span style={{ fontSize: 11, color: 'var(--muted)' }}>higher response</span>
      </div>
      <div style={{ fontSize: 12, color: 'var(--muted)', marginTop: 6 }}>{active.note}</div>
    </div>
  );
}

function gridToDataUri(grid: number[][]): string {
  const c = document.createElement('canvas');
  const h = grid.length, w = grid[0]?.length ?? 1;
  c.width = w; c.height = h;
  const ctx = c.getContext('2d')!;
  const img = ctx.createImageData(w, h);
  for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
    const v = Math.max(0, Math.min(1, grid[y][x]));
    const [r, g, b] = colormap(v);
    const i = (y * w + x) * 4;
    img.data[i] = r; img.data[i + 1] = g; img.data[i + 2] = b; img.data[i + 3] = 255;
  }
  ctx.putImageData(img, 0, 0);
  return c.toDataURL('image/png');
}

function colormap(v: number): [number, number, number] {
  // blue -> cyan -> green -> yellow -> red
  const stops: [number, [number, number, number]][] = [
    [0, [30, 58, 138]], [0.25, [6, 182, 212]], [0.5, [34, 197, 94]], [0.75, [245, 158, 11]], [1, [239, 68, 68]],
  ];
  for (let i = 1; i < stops.length; i++) {
    if (v <= stops[i][0]) {
      const [v0, c0] = stops[i - 1]; const [v1, c1] = stops[i];
      const t = (v - v0) / (v1 - v0);
      return [Math.round(c0[0] + t * (c1[0] - c0[0])), Math.round(c0[1] + t * (c1[1] - c0[1])), Math.round(c0[2] + t * (c1[2] - c0[2]))];
    }
  }
  return [239, 68, 68];
}

function GridHeatmap({ grid, overlay, opacity }: { grid: number[][]; overlay: boolean; opacity: number }) {
  const ref = useRef<HTMLCanvasElement>(null);
  useEffect(() => {
    const c = ref.current; if (!c) return;
    const h = grid.length, w = grid[0]?.length ?? 1;
    c.width = w; c.height = h;
    const ctx = c.getContext('2d')!;
    const img = ctx.createImageData(w, h);
    for (let y = 0; y < h; y++) for (let x = 0; x < w; x++) {
      const v = Math.max(0, Math.min(1, grid[y][x]));
      const [r, g, b] = colormap(v);
      const i = (y * w + x) * 4;
      img.data[i] = r; img.data[i + 1] = g; img.data[i + 2] = b; img.data[i + 3] = 255;
    }
    ctx.putImageData(img, 0, 0);
  }, [grid]);
  return (
    <canvas
      ref={ref}
      style={{
        width: '100%', display: 'block', imageRendering: 'pixelated',
        position: overlay ? 'absolute' : 'static', inset: 0,
        opacity: overlay ? opacity : 1, mixBlendMode: overlay ? 'screen' : 'normal',
      }}
    />
  );
}
