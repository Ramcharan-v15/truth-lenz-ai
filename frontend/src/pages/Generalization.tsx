import { useEffect, useState } from 'react';
import { getScorecard, getExperiments, getGenerators } from '../api';

/**
 * Generalization Scorecard (V5): 13-row evaluation matrix from
 * /api/benchmark/scorecard, plus seen-vs-unseen generator comparison per
 * experiment. Every cell comes from an experiment artifact — NOT_MEASURED
 * cells say what would be needed.
 */

const STATUS_COLORS: Record<string, string> = {
  MEASURED: '#22c55e',
  NOT_MEASURED: '#f59e0b',
  INSUFFICIENT_SAMPLE: '#f43f5e',
  ESTIMATED: '#38bdf8',
};

function ScoreBadge({ status }: { status: string }) {
  const s = (status || 'UNKNOWN').toUpperCase();
  const color = STATUS_COLORS[s] ?? '#94a3b8';
  return (
    <span className="tag" style={{ borderColor: `${color}66`, color, fontWeight: 700 }}>
      {s.replace(/_/g, ' ')}
    </span>
  );
}

const NEEDS: Record<string, string> = {
  synthetic_heldout: 'Requires a recorded synthetic benchmark experiment.',
  real_authentic: 'Requires real camera photographs with provenance, evaluated blind.',
  real_ai: 'Requires outputs from real AI generators, evaluated blind.',
  real_manipulation: 'Requires real edited photographs with known ground truth.',
  seen_generators: 'Requires labeled outputs from known generator families.',
  unseen_generators: 'Requires an entire generator family held out of all development.',
  source_heldout: 'Requires source-aware splitting with held-out sources.',
  dataset_heldout: 'Requires an entire external dataset excluded from development.',
  hard_negatives: 'Requires real hard-negative photographs (heavy edits, HDR, night mode…).',
  screenshot_subset: 'Requires real screenshot / recompression robustness data.',
  compression_subset: 'Requires real compression-level stratified evaluation.',
  ood_subset: 'Requires OOD-stratified error-rate measurement on real data.',
  blind_test: 'Requires a sealed blind test set, run with frozen config.',
};

export default function Generalization() {
  const [rows, setRows] = useState<any[]>([]);
  const [experiments, setExperiments] = useState<any[]>([]);
  const [expId, setExpId] = useState<string>('');
  const [genData, setGenData] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [fetching, setFetching] = useState(false);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const [sc, ex] = await Promise.all([
          getScorecard().catch(() => ({ rows: [] })),
          getExperiments().catch(() => ({ experiments: [] })),
        ]);
        setRows(sc?.rows ?? []);
        setExperiments(ex?.experiments ?? []);
      } catch (e: any) {
        setError(e?.message || 'Failed to load generalization data.');
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  useEffect(() => {
    if (!expId) { setGenData(null); return; }
    setFetching(true);
    getGenerators(expId)
      .then(setGenData)
      .catch((e: any) => setGenData({ error: e?.message || 'Failed to load generator comparison.' }))
      .finally(() => setFetching(false));
  }, [expId]);

  if (loading) return <div className="page"><p className="dim">Loading generalization scorecard…</p></div>;
  if (error) return <div className="page"><div className="alert">{error}</div></div>;

  return (
    <div className="page">
      <div className="eyebrow">GENERALIZATION</div>
      <h1>Generalization Scorecard</h1>
      <div className="alert" style={{ borderColor: '#f43f5e66', background: 'rgba(244,63,94,0.07)', marginBottom: 8 }}>
        <strong>NOT YET VALIDATED ON REAL-WORLD DATA</strong>
        <p className="dim" style={{ fontSize: 13, margin: '8px 0 0' }}>
          Seen-generator performance is not unseen-generator performance. Synthetic accuracy is
          not real-world accuracy. Every row below is generated from experiment artifacts —
          unmeasured rows say what is missing.
        </p>
      </div>

      <div className="panel panel-pad" style={{ marginTop: 16 }}>
        <h2 className="section-title" style={{ marginTop: 0 }}>Evaluation matrix</h2>
        {rows.length === 0 ? (
          <p className="dim">No scorecard rows available.</p>
        ) : (
          <table className="data">
            <thead><tr><th>Evaluation</th><th>Status</th><th>Value</th><th>n</th><th>Experiment</th><th>Note</th></tr></thead>
            <tbody>
              {rows.map((r: any, i: number) => {
                const measured = (r.status || '').toUpperCase() === 'MEASURED';
                return (
                  <tr key={r.id ?? i}>
                    <td style={{ fontSize: 13 }}>{r.label ?? r.evaluation ?? r.id ?? '—'}</td>
                    <td><ScoreBadge status={r.status} /></td>
                    <td className="mono">{measured && r.value != null ? String(r.value) : '—'}</td>
                    <td className="mono">{measured && r.n != null ? String(r.n) : '—'}</td>
                    <td className="mono" style={{ fontSize: 12 }}>{r.experiment_id ?? '—'}</td>
                    <td className="faint" style={{ fontSize: 12 }}>
                      {measured ? (r.note ?? '—') : (r.note ?? NEEDS[r.id] ?? 'Measurement required.')}
                    </td>
                  </tr>
                );
              })}
            </tbody>
          </table>
        )}
      </div>

      <div className="panel panel-pad" style={{ marginTop: 16 }}>
        <h2 className="section-title" style={{ marginTop: 0 }}>Seen vs unseen generators</h2>
        <p className="dim" style={{ fontSize: 13 }}>
          A detector must be evaluated on generator families it never saw during development.
          Select a completed experiment to compare.
        </p>
        {experiments.length === 0 ? (
          <div className="alert" style={{ borderColor: '#f59e0b66', background: 'rgba(245,158,11,0.06)', marginTop: 12 }}>
            <strong>No completed experiments — generator comparison NOT MEASURED.</strong>
            <p className="dim" style={{ fontSize: 13, margin: '8px 0 0' }}>
              Requires a blind experiment with at least one generator family held out of training,
              calibration, and threshold tuning.
            </p>
          </div>
        ) : (
          <>
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
            {fetching && <p className="dim" style={{ marginTop: 12 }}>Loading generator comparison…</p>}
            {!fetching && expId && genData?.error && (
              <div className="alert" style={{ marginTop: 12 }}>{genData.error}</div>
            )}
            {!fetching && expId && genData && !genData.error && (
              <table className="data" style={{ marginTop: 12 }}>
                <thead><tr><th>Split</th><th>Accuracy</th><th>n</th><th>Note</th></tr></thead>
                <tbody>
                  <tr>
                    <td style={{ fontSize: 13 }}>Seen generators</td>
                    <td className="mono">{genData.seen?.accuracy ?? '—'}</td>
                    <td className="mono">{genData.seen?.n ?? '—'}</td>
                    <td className="faint" style={{ fontSize: 12 }}>{genData.seen?.note ?? '—'}</td>
                  </tr>
                  <tr>
                    <td style={{ fontSize: 13 }}>Unseen generators</td>
                    <td className="mono">{genData.unseen?.accuracy ?? '—'}</td>
                    <td className="mono">{genData.unseen?.n ?? '—'}</td>
                    <td className="faint" style={{ fontSize: 12 }}>{genData.unseen?.note ?? '—'}</td>
                  </tr>
                  <tr>
                    <td style={{ fontSize: 13 }}><strong>Degradation</strong></td>
                    <td className="mono">{genData.degradation ?? '—'}</td>
                    <td className="mono">—</td>
                    <td className="faint" style={{ fontSize: 12 }}>
                      Performance drop from seen to unseen. A large drop means the detector
                      memorized generator signatures instead of learning general artifacts.
                    </td>
                  </tr>
                </tbody>
              </table>
            )}
          </>
        )}
      </div>
    </div>
  );
}
