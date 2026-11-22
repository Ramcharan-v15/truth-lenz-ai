import { useEffect, useRef, useState } from 'react';
import type { Signal } from '../types';

interface View { id: string; label: string; src?: string; caption: string }

interface Sample {
  view: string;
  x: number; y: number;
  r: number; g: number; b: number; a: number;
  hex: string; lum: number;
}

/**
 * Forensic viewer with synchronized pixel inspector (Phase 16).
 * One cursor drives all views: X/Y/R/G/B/A/HEX/luminance on every view,
 * plus local noise estimate and ELA anomaly score on the original.
 */
export default function ForensicsViewer({ signals, preview }: { signals: Signal[]; preview: string }) {
  const byName = (n: string) => signals.find((s) => s.name === n);
  const pixel = byName('pixel');
  const freq = byName('frequency');
  const manip = byName('manipulation');
  const noiseSig = byName('noise');

  const views: View[] = [
    { id: 'original', label: 'Original', src: preview, caption: 'Uploaded image (preview resolution).' },
  ];
  if (pixel?.visuals.ela) views.push({
    id: 'ela', label: 'ELA', src: pixel.visuals.ela,
    caption: 'Error Level Analysis: re-saved at fixed JPEG quality, differences amplified. Bright regions have a different compression history — potentially suspicious, never proof of editing.',
  });
  if (freq?.visuals.spectrum) views.push({
    id: 'spectrum', label: 'Frequency Map', src: freq.visuals.spectrum,
    caption: 'Log-magnitude Fourier spectrum (DC centered). Sharp isolated peaks can indicate generative upsampling fingerprints.',
  });
  if (manip?.visuals.manipulation_overlay) views.push({
    id: 'overlay', label: 'Manipulation Map', src: manip.visuals.manipulation_overlay,
    caption: 'Potential manipulation heatmap overlaid on the image. Warm areas mark duplicated regions or noise anomalies worth inspecting.',
  });

  const elaGrid: number[][] | null = pixel?.visuals?.ela_heat_grid ?? null;

  const [viewId, setViewId] = useState('original');
  const [zoom, setZoom] = useState(1);
  const [samples, setSamples] = useState<Sample[] | null>(null);
  const [cursor, setCursor] = useState<{ fx: number; fy: number } | null>(null);
  const imgRef = useRef<HTMLImageElement>(null);
  const canvases = useRef<Record<string, HTMLCanvasElement>>({});
  const view = views.find((v) => v.id === viewId) || views[0];

  // preload every view into a sampling canvas
  useEffect(() => {
    views.forEach((v) => {
      if (!v.src || canvases.current[v.id]) return;
      const img = new Image();
      img.onload = () => {
        const c = document.createElement('canvas');
        c.width = img.naturalWidth; c.height = img.naturalHeight;
        const ctx = c.getContext('2d', { willReadFrequently: true });
        if (ctx) { ctx.drawImage(img, 0, 0); canvases.current[v.id] = c; }
      };
      img.src = v.src;
    });
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [signals, preview]);

  const sampleAt = (e: React.MouseEvent): void => {
    const img = imgRef.current;
    if (!img) { setSamples(null); setCursor(null); return; }
    const rect = img.getBoundingClientRect();
    const fx = (e.clientX - rect.left) / rect.width;
    const fy = (e.clientY - rect.top) / rect.height;
    if (fx < 0 || fy < 0 || fx > 1 || fy > 1) { setSamples(null); setCursor(null); return; }
    setCursor({ fx, fy });
    const out: Sample[] = [];
    for (const v of views) {
      const c = canvases.current[v.id];
      if (!c) continue;
      const x = Math.min(c.width - 1, Math.max(0, Math.floor(fx * c.width)));
      const y = Math.min(c.height - 1, Math.max(0, Math.floor(fy * c.height)));
      try {
        const ctx = c.getContext('2d', { willReadFrequently: true });
        if (!ctx) continue;
        const d = ctx.getImageData(x, y, 1, 1).data;
        const lum = Math.round(0.299 * d[0] + 0.587 * d[1] + 0.114 * d[2]);
        out.push({
          view: v.label, x, y, r: d[0], g: d[1], b: d[2], a: d[3],
          hex: '#' + [d[0], d[1], d[2]].map((n) => n.toString(16).padStart(2, '0')).join('').toUpperCase(),
          lum,
        });
      } catch { /* tainted canvas */ }
    }
    setSamples(out.length ? out : null);
  };

  // local noise estimate: std-dev of luminance in a 5x5 neighborhood on original
  const localNoise = (): number | null => {
    const c = canvases.current['original'];
    const s0 = samples?.find((s) => s.view === 'Original');
    if (!c || !s0) return null;
    try {
      const ctx = c.getContext('2d', { willReadFrequently: true });
      if (!ctx) return null;
      const r = 2;
      const x0 = Math.max(0, s0.x - r), y0 = Math.max(0, s0.y - r);
      const w = Math.min(c.width - x0, 2 * r + 1), h = Math.min(c.height - y0, 2 * r + 1);
      const d = ctx.getImageData(x0, y0, w, h).data;
      const lums: number[] = [];
      for (let i = 0; i < d.length; i += 4) lums.push(0.299 * d[i] + 0.587 * d[i + 1] + 0.114 * d[i + 2]);
      const mean = lums.reduce((a, b) => a + b, 0) / lums.length;
      const sd = Math.sqrt(lums.reduce((a, b) => a + (b - mean) ** 2, 0) / lums.length);
      return Math.round(sd * 10) / 10;
    } catch { return null; }
  };

  const anomalyScore = (): number | null => {
    const s0 = samples?.find((s) => s.view === 'Original');
    const c = canvases.current['original'];
    if (!s0 || !c || !elaGrid?.length) return null;
    const gh = elaGrid.length, gw = elaGrid[0].length;
    const gx = Math.min(gw - 1, Math.floor((s0.x / c.width) * gw));
    const gy = Math.min(gh - 1, Math.floor((s0.y / c.height) * gh));
    return Math.round(elaGrid[gy][gx] * 1000) / 10; // 0..100-ish
  };

  const orig = samples?.find((s) => s.view === 'Original');
  const noise = localNoise();
  const anom = anomalyScore();

  return (
    <div>
      <div className="viewer-tabs" role="tablist" aria-label="Forensic views">
        {views.map((v) => (
          <button key={v.id} role="tab" aria-selected={viewId === v.id}
            className={viewId === v.id ? 'active' : ''} onClick={() => setViewId(v.id)}>
            {v.label}
          </button>
        ))}
        <span style={{ flex: 1 }} />
        <button onClick={() => setZoom(Math.min(4, +(zoom + 0.5).toFixed(1)))} aria-label="Zoom in">+</button>
        <button onClick={() => setZoom(Math.max(1, +(zoom - 0.5).toFixed(1)))} aria-label="Zoom out">−</button>
        <button onClick={() => setZoom(1)}>Reset</button>
      </div>
      <div className="viewer-canvas-wrap" style={{ cursor: 'crosshair', position: 'relative' }}>
        {view.src ? (
          <>
            <img
              ref={imgRef} src={view.src} alt={`${view.label} view`}
              onMouseMove={sampleAt} onMouseLeave={() => { setSamples(null); setCursor(null); }}
              style={{ transform: `scale(${zoom})`, transformOrigin: 'center', transition: 'transform 0.2s', maxHeight: 520 }}
            />
            {cursor && (
              <>
                <div style={{ position: 'absolute', left: 0, right: 0, top: `${cursor.fy * 100}%`, height: 1, background: 'rgba(56,189,248,0.7)', pointerEvents: 'none' }} />
                <div style={{ position: 'absolute', top: 0, bottom: 0, left: `${cursor.fx * 100}%`, width: 1, background: 'rgba(56,189,248,0.7)', pointerEvents: 'none' }} />
              </>
            )}
          </>
        ) : (
          <div className="empty">View unavailable</div>
        )}
      </div>
      <p className="faint" style={{ fontSize: 12.5, marginTop: 8 }}>{view.caption}</p>

      {samples ? (
        <div className="pixel-readout" aria-live="polite" style={{ fontSize: 12 }}>
          <div style={{ fontWeight: 700, marginBottom: 6 }}>
            ◉ X {orig?.x} · Y {orig?.y}
            {orig && (
              <span style={{ fontWeight: 400, color: 'var(--muted)' }}>
                {' '}· RGB({orig.r}, {orig.g}, {orig.b}) · A {orig.a} · {orig.hex} · Lum {orig.lum}
              </span>
            )}
          </div>
          <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap' }}>
            {noise !== null && <span>Local noise (5×5 σ): <b>{noise}</b></span>}
            {anom !== null && <span>ELA anomaly here: <b>{anom}%</b></span>}
          </div>
          <div style={{ marginTop: 6, display: 'flex', gap: 12, flexWrap: 'wrap' }}>
            {samples.filter((s) => s.view !== 'Original').map((s) => (
              <span key={s.view} style={{ color: 'var(--muted)' }}>
                {s.view}: <span style={{ color: 'var(--text)' }}>{s.hex} L{s.lum}</span>
              </span>
            ))}
          </div>
          <div className="faint" style={{ marginTop: 4, fontSize: 11 }}>
            Cursor is synchronized across all views — move over any view to sample them all.
          </div>
        </div>
      ) : (
        <div className="faint" style={{ fontSize: 12, marginTop: 6 }}>
          Hover over the image for the synchronized pixel inspector (all views sampled at once).
        </div>
      )}
      {noiseSig && <span style={{ display: 'none' }} />}
    </div>
  );
}
