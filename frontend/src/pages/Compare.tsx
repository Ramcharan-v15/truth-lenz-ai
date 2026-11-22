import { useEffect, useRef, useState } from 'react';
import UploadZone, { type PickedFile } from '../components/UploadZone';
import { compareImages } from '../api';
import { VERDICT_META, fmtBytes } from '../types';

/**
 * Pixel difference view (Phase 17). Computes absolute per-pixel difference
 * between the two previews. Labelled honestly: differences are differences,
 * not automatic proof of manipulation.
 */
function DifferenceView({ srcA, srcB }: { srcA: string; srcB: string }) {
  const ref = useRef<HTMLCanvasElement>(null);
  const [stats, setStats] = useState<{ meanDiff: number; changedPct: number } | null>(null);
  const [threshold, setThreshold] = useState(25);

  useEffect(() => {
    const c = ref.current;
    if (!c || !srcA || !srcB) return;
    const ia = new Image(); const ib = new Image();
    let loaded = 0;
    const go = () => {
      if (++loaded < 2) return;
      const w = Math.min(ia.naturalWidth, ib.naturalWidth, 480);
      const h = Math.min(ia.naturalHeight, ib.naturalHeight, 480);
      if (!w || !h) return;
      c.width = w; c.height = h;
      const ctx = c.getContext('2d', { willReadFrequently: true });
      if (!ctx) return;
      const off = document.createElement('canvas');
      off.width = w; off.height = h;
      const octx = off.getContext('2d', { willReadFrequently: true })!;
      ctx.drawImage(ia, 0, 0, w, h);
      const da = ctx.getImageData(0, 0, w, h);
      octx.drawImage(ib, 0, 0, w, h);
      const db = octx.getImageData(0, 0, w, h);
      const out = ctx.createImageData(w, h);
      let sum = 0, changed = 0;
      for (let i = 0; i < da.data.length; i += 4) {
        const d = (Math.abs(da.data[i] - db.data[i]) + Math.abs(da.data[i + 1] - db.data[i + 1]) + Math.abs(da.data[i + 2] - db.data[i + 2])) / 3;
        sum += d;
        if (d > threshold) changed++;
        const v = Math.min(255, d * 2.2);
        out.data[i] = v; out.data[i + 1] = Math.round(v * 0.35); out.data[i + 2] = Math.round(v * 0.2); out.data[i + 3] = 255;
      }
      ctx.putImageData(out, 0, 0);
      setStats({
        meanDiff: Math.round((sum / (w * h)) * 10) / 10,
        changedPct: Math.round((changed / (w * h)) * 1000) / 10,
      });
    };
    ia.onload = go; ib.onload = go;
    ia.src = srcA; ib.src = srcB;
  }, [srcA, srcB, threshold]);

  if (!srcA || !srcB) return null;
  return (
    <div className="panel panel-pad" style={{ marginTop: 16 }}>
      <div className="eyebrow">Pixel difference (A − B)</div>
      <div style={{ display: 'flex', gap: 16, flexWrap: 'wrap', marginTop: 8, alignItems: 'flex-start' }}>
        <canvas ref={ref} style={{ maxWidth: '100%', borderRadius: 8, background: '#000' }} />
        <div style={{ minWidth: 220 }}>
          <label style={{ fontSize: 12, color: 'var(--muted)' }}>
            Changed-region threshold: {threshold}
            <input type="range" min={5} max={80} value={threshold} onChange={(e) => setThreshold(Number(e.target.value))} style={{ display: 'block', width: 200, marginTop: 4 }} />
          </label>
          {stats && (
            <dl className="kv" style={{ marginTop: 10 }}>
              <div style={{ display: 'contents' }}><dt>Mean abs. difference</dt><dd className="mono">{stats.meanDiff}/255</dd></div>
              <div style={{ display: 'contents' }}><dt>Changed regions</dt><dd className="mono">{stats.changedPct}% of pixels</dd></div>
            </dl>
          )}
          <p className="faint" style={{ fontSize: 11.5, marginTop: 8, maxWidth: 300 }}>
            Bright areas differ between the images. A difference is a difference — it is
            <em> not </em> automatically labelled manipulation. Compare with the manipulation-signal
            deltas below for forensic context.
          </p>
        </div>
      </div>
    </div>
  );
}

export default function Compare() {
  const [a, setA] = useState<PickedFile | null>(null);
  const [b, setB] = useState<PickedFile | null>(null);
  const [result, setResult] = useState<any>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  const run = async () => {
    if (!a || !b) return;
    setLoading(true); setError(null);
    try { setResult(await compareImages(a.file, b.file)); }
    catch (e: any) { setError(e.message); }
    finally { setLoading(false); }
  };

  const c = result?.comparison;
  // V5.1.8.64: verdict probabilities may be null when numerical
  // validation withheld them. Render null as "unavailable", never as
  // zero and never by crashing on toFixed(null).
  const fmtProb = (v: any) => (v === null || v === undefined ? 'unavailable' : `${v}%`);
  const verdictChip = (v: string) => {
    const m = VERDICT_META[v as keyof typeof VERDICT_META];
    return <span className="tag" style={{ color: m.color, borderColor: m.color + '66' }}>{m.label}</span>;
  };

  return (
    <div className="shell" style={{ paddingTop: 30, paddingBottom: 20 }}>
      <div className="eyebrow">SIDE-BY-SIDE FORENSICS</div>
      <h1 style={{ margin: '0 0 8px', fontSize: 30 }}>Compare Two Images</h1>
      <p className="section-sub">Original vs edited, photo vs AI render, or two versions of the same shot. Differences across every forensic signal, side by side.</p>

      <div className="grid grid-2">
        <div>
          <div className="eyebrow">Image A</div>
          {!a ? <UploadZone compact onPick={(f) => setA(f[0])} /> : (
            <div className="file-row">
              <img className="thumb" src={a.previewUrl} alt="A" />
              <div style={{ flex: 1 }}><div style={{ fontWeight: 600, fontSize: 13.5 }}>{a.file.name}</div>
                <div className="faint mono" style={{ fontSize: 11.5 }}>{fmtBytes(a.file.size)} · {a.dims}</div></div>
              <button className="btn btn-ghost btn-sm" onClick={() => { if (a) URL.revokeObjectURL(a.previewUrl); setA(null); setResult(null); }}>✕</button>
            </div>
          )}
        </div>
        <div>
          <div className="eyebrow">Image B</div>
          {!b ? <UploadZone compact onPick={(f) => setB(f[0])} /> : (
            <div className="file-row">
              <img className="thumb" src={b.previewUrl} alt="B" />
              <div style={{ flex: 1 }}><div style={{ fontWeight: 600, fontSize: 13.5 }}>{b.file.name}</div>
                <div className="faint mono" style={{ fontSize: 11.5 }}>{fmtBytes(b.file.size)} · {b.dims}</div></div>
              <button className="btn btn-ghost btn-sm" onClick={() => { if (b) URL.revokeObjectURL(b.previewUrl); setB(null); setResult(null); }}>✕</button>
            </div>
          )}
        </div>
      </div>

      <div style={{ marginTop: 16 }}>
        <button className="btn btn-primary" disabled={!a || !b || loading} onClick={run}>
          {loading ? 'Comparing…' : 'Compare Images'}
        </button>
      </div>
      {error && <div className="alert error" style={{ marginTop: 12 }}>{error}</div>}
      {loading && <div className="skeleton" style={{ height: 220, marginTop: 16 }} />}

      {c && (
        <div style={{ marginTop: 20 }}>
          <div className="grid grid-2">
            <div className="card" style={{ textAlign: 'center' }}>
              <img src={result.image_a.preview} alt="A preview" style={{ maxHeight: 220, borderRadius: 8 }} />
              <div style={{ marginTop: 8, fontWeight: 600, fontSize: 13 }}>{result.image_a.filename}</div>
              <div style={{ marginTop: 6 }}>{verdictChip(c.verdict_a)}</div>
              <div className="mono faint" style={{ fontSize: 12, marginTop: 4 }}>AI {fmtProb(result.image_a.ensemble.ai_probability)} · Manip {fmtProb(result.image_a.ensemble.manipulation_probability)}</div>
            </div>
            <div className="card" style={{ textAlign: 'center' }}>
              <img src={result.image_b.preview} alt="B preview" style={{ maxHeight: 220, borderRadius: 8 }} />
              <div style={{ marginTop: 8, fontWeight: 600, fontSize: 13 }}>{result.image_b.filename}</div>
              <div style={{ marginTop: 6 }}>{verdictChip(c.verdict_b)}</div>
              <div className="mono faint" style={{ fontSize: 12, marginTop: 4 }}>AI {fmtProb(result.image_b.ensemble.ai_probability)} · Manip {fmtProb(result.image_b.ensemble.manipulation_probability)}</div>
            </div>
          </div>

          <div className="panel panel-pad" style={{ marginTop: 16, display: 'flex', gap: 24, flexWrap: 'wrap' }}>
            <div>
              <div className="eyebrow">AI probability delta (A − B)</div>
              {c.ai_delta !== null && c.ai_delta !== undefined ? (
                <div className="mono" style={{ fontSize: 22, fontWeight: 800, color: Math.abs(c.ai_delta) >= 10 ? (c.ai_delta > 0 ? '#f43f5e' : '#22c55e') : 'var(--text)' }}>
                  {c.ai_delta > 0 ? '+' : ''}{c.ai_delta.toFixed(1)} pts
                </div>
              ) : (
                <div className="mono" style={{ fontSize: 22, fontWeight: 800 }}>unavailable</div>
              )}
            </div>
            <div>
              <div className="eyebrow">Manipulation delta (A − B)</div>
              {c.manip_delta !== null && c.manip_delta !== undefined ? (
                <div className="mono" style={{ fontSize: 22, fontWeight: 800, color: Math.abs(c.manip_delta) >= 10 ? (c.manip_delta > 0 ? '#f59e0b' : '#22c55e') : 'var(--text)' }}>
                  {c.manip_delta > 0 ? '+' : ''}{c.manip_delta.toFixed(1)} pts
                </div>
              ) : (
                <div className="mono" style={{ fontSize: 22, fontWeight: 800 }}>unavailable</div>
              )}
            </div>
            {c.ml_ai_delta !== null && c.ml_ai_delta !== undefined && (
              <div>
                <div className="eyebrow">ML detector delta (A − B)</div>
                <div className="mono" style={{ fontSize: 22, fontWeight: 800 }}>
                  {c.ml_ai_delta > 0 ? '+' : ''}{c.ml_ai_delta.toFixed(1)} pts
                </div>
              </div>
            )}
          </div>
          {c.comparison_note && (
            <div className="mono faint" style={{ marginTop: 12 }}>{c.comparison_note}</div>
          )}

          <DifferenceView srcA={result.image_a.preview} srcB={result.image_b.preview} />

          <h2 className="section-title">Signal differences <span className="faint" style={{ fontSize: 13 }}>(A − B)</span></h2>
          <div className="panel panel-pad" style={{ overflowX: 'auto' }}>
            <table className="data">
              <thead><tr><th>Signal</th><th>AI (A)</th><th>AI (B)</th><th>Δ AI</th><th>Manip (A)</th><th>Manip (B)</th><th>Δ Manip</th></tr></thead>
              <tbody>
                {c.signal_deltas.map((d: any) => (
                  <tr key={d.signal}>
                    <td>{d.signal}</td>
                    <td className="mono">{d.ai_a.toFixed(0)}</td>
                    <td className="mono">{d.ai_b.toFixed(0)}</td>
                    <td className="mono" style={{ color: Math.abs(d.ai_delta) >= 10 ? (d.ai_delta > 0 ? '#f43f5e' : '#22c55e') : 'var(--text-faint)', fontWeight: 700 }}>
                      {d.ai_delta > 0 ? '+' : ''}{d.ai_delta.toFixed(0)}
                    </td>
                    <td className="mono">{d.manip_a.toFixed(0)}</td>
                    <td className="mono">{d.manip_b.toFixed(0)}</td>
                    <td className="mono" style={{ color: Math.abs(d.manip_delta) >= 10 ? (d.manip_delta > 0 ? '#f59e0b' : '#22c55e') : 'var(--text-faint)', fontWeight: 700 }}>
                      {d.manip_delta > 0 ? '+' : ''}{d.manip_delta.toFixed(0)}
                    </td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>

          <h2 className="section-title">Metadata comparison</h2>
          <div className="grid grid-2">
            {(['metadata_a', 'metadata_b'] as const).map((k, i) => (
              <div className="panel panel-pad" key={k}>
                <div className="eyebrow">{i === 0 ? 'Image A' : 'Image B'}</div>
                <dl className="kv">
                  <div style={{ display: 'contents' }}><dt>Format</dt><dd>{c[k].format}</dd></div>
                  <div style={{ display: 'contents' }}><dt>EXIF tags</dt><dd>{c[k].exif_tag_count ?? 0}</dd></div>
                  <div style={{ display: 'contents' }}><dt>Camera</dt><dd>{c[k].exif_summary?.Make || '—'} {c[k].exif_summary?.Model || ''}</dd></div>
                  <div style={{ display: 'contents' }}><dt>Software</dt><dd>{c[k].exif_summary?.Software || '—'}</dd></div>
                  <div style={{ display: 'contents' }}><dt>C2PA</dt><dd>{
                    c[k].c2pa_status ? String(c[k].c2pa_status).toUpperCase() : (c[k].c2pa_detected ? 'Detected' : 'Not detected')
                  }</dd></div>
                </dl>
              </div>
            ))}
          </div>
        </div>
      )}
    </div>
  );
}
