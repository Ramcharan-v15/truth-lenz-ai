import { useEffect, useState } from 'react';
import { getHealth, getConfig, getModelStatus, reloadModels, type ModelRegistryStatus } from '../api';
import ModelStatus from '../components/ModelStatus';

export default function Settings() {
  const [health, setHealth] = useState<any>(null);
  const [config, setConfig] = useState<any>(null);
  const [models, setModels] = useState<ModelRegistryStatus | null>(null);
  const [reloading, setReloading] = useState(false);
  const [theme, setTheme] = useState(() => localStorage.getItem('tl-theme') || 'dark');
  const [demoMode, setDemoMode] = useState(() => localStorage.getItem('tl-demo') === '1');

  const loadModels = () => getModelStatus().then(setModels).catch(() => setModels(null));

  useEffect(() => {
    getHealth().then(setHealth).catch(() => setHealth({ status: 'unreachable' }));
    getConfig().then(setConfig).catch(() => {});
    loadModels();
  }, []);

  const applyTheme = (t: string) => {
    setTheme(t);
    localStorage.setItem('tl-theme', t);
    document.documentElement.setAttribute('data-theme', t);
  };

  const doReload = async () => {
    setReloading(true);
    try { setModels(await reloadModels()); } catch { /* keep old */ }
    setReloading(false);
  };

  return (
    <div className="shell" style={{ paddingTop: 30, paddingBottom: 20 }}>
      <div className="eyebrow">PREFERENCES & DIAGNOSTICS</div>
      <h1 style={{ margin: '0 0 8px', fontSize: 30 }}>Settings</h1>

      <div className="grid grid-2">
        <div className="panel panel-pad">
          <div className="eyebrow">Appearance</div>
          <div className="viewer-tabs">
            {(['dark', 'light'] as const).map((t) => (
              <button key={t} className={theme === t ? 'active' : ''} onClick={() => applyTheme(t)} aria-pressed={theme === t}>
                {t === 'dark' ? '☾ Dark forensic' : '☀ Light'}
              </button>
            ))}
          </div>
          <p className="faint" style={{ fontSize: 12.5 }}>Your preference is remembered on this device.</p>
        </div>

        <div className="panel panel-pad">
          <div className="eyebrow">Demo mode</div>
          <label className="checkbox">
            <input type="checkbox" checked={demoMode}
              onChange={(e) => { setDemoMode(e.target.checked); localStorage.setItem('tl-demo', e.target.checked ? '1' : '0'); }} />
            Prefer demo results on the Analyze page
          </label>
          <p className="faint" style={{ fontSize: 12.5 }}>
            Demo results are always labelled “DEMO RESULT — NOT REAL ANALYSIS” and are never
            saved to history or mixed with real analyses.
          </p>
        </div>
      </div>

      <h2 className="section-title">Model registry</h2>
      <p className="section-sub">Trained AI-detection models plug in here as one ensemble signal. Missing models are reported honestly — never faked.</p>
      <ModelStatus />
      <div className="panel panel-pad" style={{ marginTop: 12 }}>
        <div style={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <div className="eyebrow">Registered models ({models?.total ?? '…'})</div>
          <button className="btn btn-ghost btn-sm" onClick={doReload} disabled={reloading}>
            {reloading ? 'Reloading…' : 'Reload registry'}
          </button>
        </div>
        {models && models.models.length === 0 && (
          <p className="faint" style={{ fontSize: 13, marginTop: 8 }}>
            No models registered. Add model files to <code>models/</code> and register them in{' '}
            <code>models/registry.json</code> (see <code>models/README.md</code>), then reload.
            The platform continues to work on forensic signals alone.
          </p>
        )}
        {models?.models.map((m) => (
          <div key={m.name} className="panel panel-pad" style={{ marginTop: 10 }}>
            <div style={{ fontWeight: 700 }}>{m.name} <span className="faint mono" style={{ fontSize: 12 }}>v{m.version} · {m.model_type}</span></div>
            <div style={{ marginTop: 6 }}>
              <span style={{
                padding: '3px 10px', borderRadius: 20, fontSize: 11.5, fontWeight: 700,
                background: m.status === 'ready' ? 'rgba(34,197,94,0.15)' : 'rgba(100,116,139,0.15)',
                color: m.status === 'ready' ? '#22c55e' : 'var(--muted)',
              }}>{m.status.replace(/_/g, ' ').toUpperCase()}</span>
            </div>
            <dl className="kv" style={{ marginTop: 8 }}>
              <div style={{ display: 'contents' }}><dt>Formats</dt><dd className="mono" style={{ fontSize: 12 }}>{m.supported_formats.join(', ')}</dd></div>
              <div style={{ display: 'contents' }}><dt>Preprocessing</dt><dd className="mono" style={{ fontSize: 12 }}>{JSON.stringify(m.preprocessing)}</dd></div>
              <div style={{ display: 'contents' }}><dt>Limitations</dt><dd style={{ fontFamily: 'var(--font)', fontSize: 12.5 }}>{m.limitations}</dd></div>
            </dl>
          </div>
        ))}
        {models && Object.keys(models.errors).length > 0 && (
          <div className="alert error" style={{ marginTop: 10 }}>
            {Object.entries(models.errors).map(([k, v]) => <div key={k}><b>{k}:</b> {v}</div>)}
          </div>
        )}
        <p className="faint mono" style={{ fontSize: 11.5, marginTop: 8 }}>models_dir: {models?.models_dir}</p>
      </div>

      <h2 className="section-title">Developer diagnostics</h2>
      <p className="section-sub">Backend status, analyzer inventory and configuration version. Intended for developers and debugging.</p>
      <div className="panel panel-pad">
        <dl className="kv">
          <div style={{ display: 'contents' }}><dt>Backend status</dt><dd>{health ? health.status : '…'}</dd></div>
          <div style={{ display: 'contents' }}><dt>Service</dt><dd>{health?.service ?? '—'}</dd></div>
          <div style={{ display: 'contents' }}><dt>Backend version</dt><dd>{health?.version ?? '—'}</dd></div>
          <div style={{ display: 'contents' }}><dt>Config version</dt><dd>{config?.version ?? '—'}</dd></div>
          <div style={{ display: 'contents' }}><dt>Analyzers</dt><dd style={{ fontFamily: 'var(--font)' }}>{health?.analyzers?.join(', ') ?? '—'}</dd></div>
          <div style={{ display: 'contents' }}><dt>API docs</dt><dd><a href="/api/docs" target="_blank" rel="noreferrer">/api/docs</a> (interactive)</dd></div>
        </dl>
      </div>

      <h2 className="section-title">Responsible use</h2>
      <div className="disclaimer-strip">
        AI image detection is probabilistic. No detector can guarantee whether an image is AI-generated
        or authentic. Do not use Truth Lenz AI as the sole basis for legal, academic, employment,
        financial or other high-stakes decisions.
      </div>
    </div>
  );
}
