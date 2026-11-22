import { useEffect, useRef, useState } from 'react';
import UploadZone, { type PickedFile } from '../components/UploadZone';
import Pipeline from '../components/Pipeline';
import ResultsDashboard from '../components/ResultsDashboard';
import { analyzeImage } from '../api';
import type { AnalysisReport } from '../types';
import { fmtBytes } from '../types';
import { DEMO_REPORT } from '../demoReport';

const STAGE_COUNT = 10;

export default function Analyze() {
  const [picked, setPicked] = useState<PickedFile | null>(null);
  const [progress, setProgress] = useState(0);
  const [phase, setPhase] = useState<'idle' | 'uploading' | 'analyzing' | 'done' | 'error'>('idle');
  const [activeStage, setActiveStage] = useState(0);
  const [report, setReport] = useState<AnalysisReport | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [saveHistory, setSaveHistory] = useState(false);
  const timer = useRef<number | null>(null);

  useEffect(() => () => { if (timer.current) window.clearInterval(timer.current); }, []);

  const animateStages = () => {
    // Advance through forensic stages while the backend works.
    // Final statuses/durations come from the real response - this only paces the UI.
    setActiveStage(0);
    timer.current = window.setInterval(() => {
      setActiveStage((s) => (s < STAGE_COUNT - 1 ? s + 1 : s));
    }, 420);
  };

  const run = async () => {
    if (!picked) return;
    setError(null); setReport(null); setProgress(0);
    setPhase('uploading');
    try {
      const rep = await analyzeImage(picked.file, saveHistory, (p) => {
        setProgress(p);
        if (p >= 100) { setPhase('analyzing'); animateStages(); }
      });
      if (timer.current) window.clearInterval(timer.current);
      setReport(rep);
      setPhase('done');
    } catch (e: any) {
      if (timer.current) window.clearInterval(timer.current);
      setError(e.message || 'Analysis failed.');
      setPhase('error');
    }
  };

  const loadDemo = () => {
    setError(null); setPicked(null); setReport(DEMO_REPORT); setPhase('done');
  };

  const reset = () => {
    setPicked(null); setReport(null); setError(null); setProgress(0); setPhase('idle'); setActiveStage(0);
  };

  return (
    <div className="shell" style={{ paddingTop: 30, paddingBottom: 20 }}>
      <div className="eyebrow">FORENSIC ANALYSIS</div>
      <h1 style={{ margin: '0 0 8px', fontSize: 30 }}>Analyze an Image</h1>
      <p className="section-sub">
        Upload a JPG, PNG, WEBP or BMP. Ten forensic analyzers inspect it and a weighted
        ensemble explains its verdict. Processed in memory — nothing is stored unless you opt in.
      </p>

      {phase !== 'done' && (
        <>
          <UploadZone onPick={(f) => { if (picked) URL.revokeObjectURL(picked.previewUrl); setPicked(f[0]); setPhase('idle'); }} />
          {picked && (
            <div style={{ marginTop: 14 }}>
              <div className="file-row">
                <img className="thumb" src={picked.previewUrl} alt="preview" />
                <div style={{ flex: 1 }}>
                  <div style={{ fontWeight: 600, fontSize: 14 }}>{picked.file.name}</div>
                  <div className="faint mono" style={{ fontSize: 12 }}>
                    {fmtBytes(picked.file.size)} · {picked.dims} · {picked.file.type || 'image'}
                  </div>
                  {(phase === 'uploading' || phase === 'analyzing') && (
                    <div className="progress" style={{ marginTop: 8 }} aria-label="Upload progress">
                      <div style={{ width: `${phase === 'analyzing' ? 100 : progress}%` }} />
                    </div>
                  )}
                </div>
                {phase === 'idle' && (
                  <button className="btn btn-ghost btn-sm" onClick={() => { if (picked) URL.revokeObjectURL(picked.previewUrl); setPicked(null); }} aria-label="Remove image">✕</button>
                )}
              </div>

              {(phase === 'uploading' || phase === 'analyzing') && (
                <div className="panel panel-pad" style={{ marginTop: 14 }}>
                  <div className="eyebrow">
                    {phase === 'uploading' ? `Uploading… ${progress}%` : 'Analysis pipeline'}
                  </div>
                  {phase === 'analyzing' ? (
                    <Pipeline activeIndex={activeStage} />
                  ) : (
                    <div className="faint" style={{ fontSize: 13 }}>Transferring image to the forensic engine…</div>
                  )}
                </div>
              )}

              {phase === 'idle' && (
                <div style={{ display: 'flex', gap: 12, marginTop: 14, alignItems: 'center', flexWrap: 'wrap' }}>
                  <button className="btn btn-primary" onClick={run}>🔍 Analyze Image</button>
                  <label className="checkbox" title="Persist thumbnail + results to optional on-device history">
                    <input type="checkbox" checked={saveHistory} onChange={(e) => setSaveHistory(e.target.checked)} />
                    Save to history
                  </label>
                </div>
              )}
            </div>
          )}

          {phase === 'error' && error && (
            <div style={{ marginTop: 14 }}>
              <div className="alert error" role="alert">
                <strong>Unable to analyze this image.</strong><br />
                {error}<br />
                <span className="dim">What you can try: re-export the file as JPG/PNG and upload again, or use a smaller file.</span>
              </div>
              <button className="btn" style={{ marginTop: 10 }} onClick={reset}>Try another image</button>
            </div>
          )}

          <div className="divider" />
          <div className="panel panel-pad">
            <div className="eyebrow">Demo mode</div>
            <p className="dim" style={{ fontSize: 13.5, margin: '0 0 12px' }}>
              Explore the results dashboard with clearly-labelled sample output — no image upload,
              no backend detection involved.
            </p>
            <button className="btn btn-ghost" onClick={loadDemo}>Open demo result</button>
          </div>
        </>
      )}

      {phase === 'done' && report && (
        <>
          <div style={{ display: 'flex', gap: 10, marginBottom: 18, flexWrap: 'wrap' }}>
            <button className="btn btn-ghost btn-sm" onClick={reset}>← Analyze another image</button>
            {report.saved_to_history && <span className="tag present">Saved to history</span>}
          </div>
          <ResultsDashboard report={report} />
        </>
      )}
    </div>
  );
}

