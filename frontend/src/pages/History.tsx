import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import { getHistory, deleteAnalysis, deleteAllHistory, getAnalysis } from '../api';
import { VERDICT_META, fmtPct } from '../types';
import ResultsDashboard from '../components/ResultsDashboard';
import type { AnalysisReport } from '../types';

export default function History() {
  const [data, setData] = useState<any>(null);
  const [open, setOpen] = useState<AnalysisReport | null>(null);
  const [loading, setLoading] = useState(true);

  const refresh = async () => {
    setLoading(true);
    try { setData(await getHistory()); } catch { setData({ analyses: [], stats: { total: 0, by_verdict: {}, avg_confidence: null } }); }
    setLoading(false);
  };
  useEffect(() => { refresh(); }, []);

  const remove = async (id: string) => {
    await deleteAnalysis(id);
    if (open?.analysis_id === id) setOpen(null);
    refresh();
  };
  const clear = async () => {
    if (window.confirm('Delete all saved analyses? This cannot be undone.')) {
      await deleteAllHistory(); refresh();
    }
  };

  if (open) {
    return (
      <div className="shell" style={{ paddingTop: 24, paddingBottom: 20 }}>
        <button className="btn btn-ghost btn-sm" onClick={() => setOpen(null)} style={{ marginBottom: 16 }}>← Back to history</button>
        <ResultsDashboard report={open} />
      </div>
    );
  }

  const stats = data?.stats || { total: 0, by_verdict: {}, avg_confidence: null };
  const analyses = data?.analyses || [];

  return (
    <div className="shell" style={{ paddingTop: 30, paddingBottom: 20 }}>
      <div className="eyebrow">ANALYSIS HISTORY</div>
      <h1 style={{ margin: '0 0 8px', fontSize: 30 }}>History & Dashboard</h1>
      <p className="section-sub">
        Only analyses you explicitly chose to save appear here. Thumbnails only — full images are never stored.
      </p>

      <div className="grid grid-4" style={{ marginBottom: 20 }}>
        <div className="card"><div className="eyebrow">Total analyses</div><div className="metric"><div className="val">{stats.total}</div></div></div>
        {(Object.keys(VERDICT_META) as (keyof typeof VERDICT_META)[]).map((v) => (
          <div className="card" key={v}>
            <div className="eyebrow">{VERDICT_META[v].label}</div>
            <div className="metric"><div className="val" style={{ color: VERDICT_META[v].color }}>{stats.by_verdict[v] || 0}</div></div>
          </div>
        ))}
      </div>
      <div className="card" style={{ marginBottom: 20 }}>
        <span className="dim" style={{ fontSize: 13.5 }}>Average confidence across saved analyses: </span>
        <strong className="mono">{stats.avg_confidence == null ? 'unavailable' : `${stats.avg_confidence.toFixed(0)}%`}</strong>
      </div>

      {loading ? (
        <div className="skeleton" style={{ height: 160 }} />
      ) : analyses.length === 0 ? (
        <div className="empty panel">
          <div style={{ fontSize: 30, marginBottom: 10 }}>◌</div>
          <p>No saved analyses yet.</p>
          <p className="faint" style={{ fontSize: 13 }}>Enable “Save to history” on the Analyze page to keep results here.</p>
          <Link to="/analyze" className="btn btn-primary" style={{ marginTop: 8 }}>Analyze an image</Link>
        </div>
      ) : (
        <>
          <div style={{ display: 'flex', justifyContent: 'flex-end', marginBottom: 12 }}>
            <button className="btn btn-danger btn-sm" onClick={clear}>Delete all history</button>
          </div>
          <div className="grid grid-3">
            {analyses.map((a: any) => {
              const m = VERDICT_META[a.verdict as keyof typeof VERDICT_META];
              return (
                <div key={a.id} className="card hoverable">
                  <div style={{ display: 'flex', gap: 12 }}>
                    {a.thumbnail && <img src={a.thumbnail} alt="" style={{ width: 64, height: 64, objectFit: 'cover', borderRadius: 8, border: '1px solid var(--border)' }} />}
                    <div style={{ flex: 1, minWidth: 0 }}>
                      <div style={{ fontWeight: 600, fontSize: 13.5, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>{a.filename}</div>
                      <div className="faint mono" style={{ fontSize: 11 }}>{a.created_at} · {a.width}×{a.height}</div>
                      <div style={{ marginTop: 6 }}><span className="tag" style={{ color: m.color, borderColor: m.color + '66' }}>{m.label}</span></div>
                      <div className="mono faint" style={{ fontSize: 11.5, marginTop: 4 }}>
                        AI {fmtPct(a.ai_probability, 0)} · Manip {fmtPct(a.manipulation_probability, 0)} · Conf {fmtPct(a.confidence, 0)}
                      </div>
                    </div>
                  </div>
                  <div style={{ display: 'flex', gap: 8, marginTop: 12 }}>
                    <button className="btn btn-sm" onClick={async () => setOpen(await getAnalysis(a.id))}>Open</button>
                    <button className="btn btn-ghost btn-sm" onClick={() => remove(a.id)}>Delete</button>
                  </div>
                </div>
              );
            })}
          </div>
        </>
      )}
    </div>
  );
}
