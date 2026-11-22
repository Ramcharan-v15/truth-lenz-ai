import { useEffect, useState } from 'react';
import { getExperiments, getExperiment } from '../api';

/**
 * Experiments (V4): experiment tracking list + reproducibility detail.
 * Everything shown comes from the backend — never invented.
 */

function Kv({ k, v }: { k: string; v: any }) {
  return (
    <tr><td className="faint">{k}</td><td className="mono" style={{ wordBreak: 'break-all' }}>{String(v ?? '—')}</td></tr>
  );
}

export default function Experiments() {
  const [list, setList] = useState<any[]>([]);
  const [selected, setSelected] = useState<string | null>(null);
  const [detail, setDetail] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [detailLoading, setDetailLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const r = await getExperiments();
        setList(r?.experiments ?? []);
      } catch (e: any) {
        setError(e?.message || 'Failed to load experiments.');
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  const openDetail = async (id: string) => {
    setSelected(id);
    setDetailLoading(true);
    setDetail(null);
    try {
      const r = await getExperiment(id);
      setDetail(r);
    } catch (e: any) {
      setDetail({ error: e?.message || 'Failed to load experiment.' });
    } finally {
      setDetailLoading(false);
    }
  };

  if (loading) return <div className="page"><p className="dim">Loading experiments…</p></div>;
  if (error) return <div className="page"><div className="alert">{error}</div></div>;

  return (
    <div className="page">
      <div className="eyebrow">RESEARCH TRACKING</div>
      <h1>Experiments</h1>
      <p className="section-sub">
        Tracked evaluation runs with full reproducibility bundles. Each experiment records
        dataset, model, configuration, and result hashes so results can be audited — or reproduced.
      </p>

      {list.length === 0 ? (
        <div className="panel panel-pad" style={{ textAlign: 'center', padding: 40 }}>
          <div style={{ fontSize: 18, fontWeight: 700 }}>No experiments tracked yet.</div>
          <p className="dim" style={{ fontSize: 13 }}>
            Benchmarks and validations will appear here once they are run and recorded.
          </p>
        </div>
      ) : (
        <div className="panel panel-pad">
          <table className="data">
            <thead>
              <tr><th>Experiment ID</th><th>Status</th><th>Created</th><th>Dataset</th><th>Model</th><th>Results hash</th></tr>
            </thead>
            <tbody>
              {list.map((e: any) => (
                <tr
                  key={e.experiment_id}
                  onClick={() => openDetail(e.experiment_id)}
                  style={{ cursor: 'pointer', background: selected === e.experiment_id ? 'rgba(56,189,248,0.08)' : undefined }}
                >
                  <td className="mono">{e.experiment_id}</td>
                  <td><span className="tag">{e.status ?? '—'}</span></td>
                  <td className="faint" style={{ fontSize: 12 }}>{e.created_at ?? '—'}</td>
                  <td className="mono">{e.dataset_id ?? '—'}</td>
                  <td className="mono">{e.model_id ?? '—'}</td>
                  <td className="mono faint" style={{ fontSize: 11 }}>{(e.results_hash ?? '—').slice(0, 16)}</td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {selected && (
        <div className="panel panel-pad" style={{ marginTop: 16 }}>
          <div className="eyebrow">EXPERIMENT DETAIL</div>
          <h2 className="mono" style={{ fontSize: 18 }}>{selected}</h2>
          {detailLoading && <p className="dim">Loading…</p>}
          {detail?.error && <div className="alert">{detail.error}</div>}
          {detail && !detail.error && (
            <>
              <h3 className="section-title" style={{ fontSize: 14 }}>Reproducibility</h3>
              <table className="data">
                <tbody>
                  {Object.entries(detail.reproducibility ?? detail).filter(([k]) =>
                    /hash|seed|version|id|config|timestamp|software|depend/i.test(k)
                  ).map(([k, v]) => <Kv key={k} k={k} v={typeof v === 'object' ? JSON.stringify(v) : v} />)}
                </tbody>
              </table>
              <h3 className="section-title" style={{ fontSize: 14, marginTop: 16 }}>Metrics</h3>
              <pre className="mono" style={{
                fontSize: 12, background: 'var(--bg)', border: '1px solid var(--border)',
                borderRadius: 8, padding: 12, overflowX: 'auto', whiteSpace: 'pre-wrap',
              }}>
                {JSON.stringify(detail.metrics ?? detail.results ?? detail, null, 2)}
              </pre>
            </>
          )}
        </div>
      )}
    </div>
  );
}
