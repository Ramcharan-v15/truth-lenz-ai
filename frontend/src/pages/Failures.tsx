import { useEffect, useState } from 'react';
import { getExperiments, getFailures } from '../api';

/**
 * Failure Analysis (V5): false positives, false negatives, inconclusive
 * clusters per completed experiment. All clusters come from
 * /api/benchmark/failures/{experiment_id} — never invented.
 */

function Cluster({ title, color, data }: { title: string; color: string; data: any }) {
  if (!data) return null;
  const cases: any[] = Array.isArray(data) ? data : (data.cases ?? []);
  const count = data.count ?? cases.length;
  return (
    <div className="panel panel-pad" style={{ marginTop: 16 }}>
      <h2 className="section-title" style={{ marginTop: 0 }}>
        <span className="tag" style={{ borderColor: `${color}66`, color, fontWeight: 700, marginRight: 8 }}>
          {title}
        </span>
        <span className="dim" style={{ fontSize: 14 }}>{count} case{count === 1 ? '' : 's'}</span>
      </h2>
      {data.note && <p className="dim" style={{ fontSize: 13 }}>{data.note}</p>}
      {cases.length === 0 ? (
        <p className="faint" style={{ fontSize: 13 }}>No cases in this cluster.</p>
      ) : (
        <table className="data">
          <thead><tr><th>Case ID</th><th>Category</th><th>Detail</th></tr></thead>
          <tbody>
            {cases.map((c: any, i: number) => (
              <tr key={i}>
                <td className="mono">{c.case_id ?? c.image_id ?? `case-${i + 1}`}</td>
                <td style={{ fontSize: 13 }}>{c.category ?? c.group ?? '—'}</td>
                <td className="faint" style={{ fontSize: 12 }}>
                  {typeof c === 'string' ? c : (c.detail ?? c.note ?? JSON.stringify(c).slice(0, 120))}
                </td>
              </tr>
            ))}
          </tbody>
        </table>
      )}
    </div>
  );
}

export default function Failures() {
  const [experiments, setExperiments] = useState<any[]>([]);
  const [expId, setExpId] = useState<string>('');
  const [failures, setFailures] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [fetching, setFetching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const r = await getExperiments().catch(() => ({ experiments: [] }));
        setExperiments(r?.experiments ?? []);
      } catch (e: any) {
        setError(e?.message || 'Failed to load experiments.');
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  useEffect(() => {
    if (!expId) { setFailures(null); return; }
    setFetching(true);
    getFailures(expId)
      .then(setFailures)
      .catch((e: any) => setFailures({ error: e?.message || 'Failed to load failure analysis.' }))
      .finally(() => setFetching(false));
  }, [expId]);

  if (loading) return <div className="page"><p className="dim">Loading experiments…</p></div>;
  if (error) return <div className="page"><div className="alert">{error}</div></div>;

  return (
    <div className="page">
      <div className="eyebrow">FAILURE ANALYSIS</div>
      <h1>Failures</h1>
      <p className="section-sub" style={{ marginTop: -8 }}>
        Where the detector gets it wrong — false positives, false negatives, and inconclusive
        cases, grouped by failure mode. Exposing weaknesses is the point.
      </p>

      {experiments.length === 0 ? (
        <div className="alert" style={{ borderColor: '#f59e0b66', background: 'rgba(245,158,11,0.06)' }}>
          <strong>No failure analysis available — no completed experiments.</strong>
          <p className="dim" style={{ fontSize: 13, margin: '8px 0 0' }}>
            Failure analysis requires a completed validation experiment. Run one from the
            Experiments page first.
          </p>
        </div>
      ) : (
        <>
          <div className="panel panel-pad" style={{ marginTop: 16 }}>
            <label className="faint" style={{ fontSize: 13 }}>Experiment
              <select
                value={expId}
                onChange={(e) => setExpId(e.target.value)}
                style={{ display: 'block', marginTop: 6, maxWidth: 420, width: '100%' }}
              >
                <option value="">— select an experiment —</option>
                {experiments.map((x: any) => (
                  <option key={x.experiment_id} value={x.experiment_id}>
                    {x.experiment_id} — {x.status ?? 'unknown'}
                  </option>
                ))}
              </select>
            </label>
          </div>

          {fetching && <p className="dim" style={{ marginTop: 12 }}>Loading failure analysis…</p>}

          {!fetching && expId && failures?.error && (
            <div className="alert" style={{ marginTop: 16 }}>{failures.error}</div>
          )}

          {!fetching && expId && failures && !failures.error && (
            <>
              <Cluster title="FALSE POSITIVES" color="#f43f5e" data={failures.false_positives} />
              <Cluster title="FALSE NEGATIVES" color="#f59e0b" data={failures.false_negatives} />
              <Cluster title="INCONCLUSIVE" color="#94a3b8" data={failures.inconclusive} />
              {failures.inconclusive_rate != null && (
                <p className="dim" style={{ fontSize: 13, marginTop: 12 }}>
                  Inconclusive rate: <span className="mono">{failures.inconclusive_rate}</span> —
                  a legitimate metric, not something to optimize away.
                </p>
              )}
            </>
          )}
        </>
      )}
    </div>
  );
}
