import { useEffect, useState } from 'react';
import { Link } from 'react-router-dom';
import {
  getValidation, getDatasets, getCrossGenerator, getHardNegatives,
  getOodInfo, getCalibrationInfo,
} from '../api';

/**
 * Validation Dashboard (V4): separates synthetic evaluation from real-world
 * validation. Every number comes from a backend endpoint; anything unmeasured
 * is shown as NOT_MEASURED with what would be needed — never invented.
 */

function StatusBadge({ status }: { status: string }) {
  const color =
    status === 'MEASURED' ? '#22c55e'
    : status === 'NOT_MEASURED' ? '#f59e0b'
    : status === 'NOT_YET_VALIDATED' ? '#f43f5e'
    : status === 'PARTIAL_REAL_WORLD_VALIDATION' ? '#f59e0b'
    : '#94a3b8';
  return (
    <span className="tag" style={{ borderColor: `${color}66`, color, fontWeight: 700 }}>
      {status.replace(/_/g, ' ')}
    </span>
  );
}

function Section({ title, sub, children }: { title: string; sub?: string; children: React.ReactNode }) {
  return (
    <div className="panel panel-pad" style={{ marginTop: 16 }}>
      <h2 className="section-title" style={{ marginTop: 0 }}>{title}</h2>
      {sub && <p className="section-sub">{sub}</p>}
      {children}
    </div>
  );
}

function NotMeasured({ what, needed }: { what: string; needed: string }) {
  return (
    <div className="alert" style={{ borderColor: '#f59e0b66', background: 'rgba(245,158,11,0.06)' }}>
      <strong>{what}: <StatusBadge status="NOT_MEASURED" /></strong>
      <p className="dim" style={{ fontSize: 13, margin: '8px 0 0' }}>{needed}</p>
    </div>
  );
}

export default function Validation() {
  const [validation, setValidation] = useState<any>(null);
  const [datasets, setDatasets] = useState<any[]>([]);
  const [crossGen, setCrossGen] = useState<any>(null);
  const [hardNeg, setHardNeg] = useState<any>(null);
  const [ood, setOod] = useState<any>(null);
  const [calib, setCalib] = useState<any>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);

  useEffect(() => {
    (async () => {
      try {
        const [v, ds, cg, hn, o, c] = await Promise.all([
          getValidation().catch(() => null),
          getDatasets().catch(() => ({ datasets: [] })),
          getCrossGenerator().catch(() => null),
          getHardNegatives().catch(() => null),
          getOodInfo().catch(() => null),
          getCalibrationInfo().catch(() => null),
        ]);
        setValidation(v);
        setDatasets(ds?.datasets ?? []);
        setCrossGen(cg);
        setHardNeg(hn);
        setOod(o);
        setCalib(c);
      } catch (e: any) {
        setError(e?.message || 'Failed to load validation status.');
      } finally {
        setLoading(false);
      }
    })();
  }, []);

  if (loading) return <div className="page"><p className="dim">Loading validation status…</p></div>;
  if (error) return <div className="page"><div className="alert">{error}</div></div>;

  const hnSyn = hardNeg?.synthetic;
  const calibModels = calib?.models ?? [];

  return (
    <div className="page">
      <div className="eyebrow">SCIENTIFIC VALIDATION</div>
      <h1>Validation Dashboard</h1>
      <div className="alert" style={{ borderColor: '#f43f5e66', background: 'rgba(244,63,94,0.07)', marginBottom: 8 }}>
        <strong>{(validation?.validation_status ?? 'NOT_YET_VALIDATED_ON_REAL_WORLD_DATA').replace(/_/g, ' ')}</strong>
        <p className="dim" style={{ fontSize: 13, margin: '8px 0 0' }}>
          Everything below this line is synthetic evaluation only. Real-world validation requires a
          real-world dataset — see §4. Until then, synthetic metrics must not be presented as
          real-world performance.
        </p>
      </div>

      {/* 1. Dataset overview */}
      <Section title="Dataset Overview" sub="Registered evaluation datasets and their validation status.">
        {datasets.length === 0 ? (
          <p className="dim">No datasets registered.</p>
        ) : (
          <table className="data">
            <thead><tr><th>Dataset</th><th>Version</th><th>Images</th><th>Status</th><th>Description</th></tr></thead>
            <tbody>
              {datasets.map((d: any) => (
                <tr key={d.dataset_id}>
                  <td className="mono">{d.dataset_id}</td>
                  <td className="mono">{d.version ?? '—'}</td>
                  <td className="mono">{d.total_images ?? '—'}</td>
                  <td><StatusBadge status={d.validation_status ?? 'UNKNOWN'} /></td>
                  <td className="faint" style={{ fontSize: 12 }}>{d.description ?? '—'}</td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Section>

      {/* 2. Synthetic performance */}
      <Section title="Synthetic Performance"
        sub="Performance on the synthetic training/evaluation set. This is NOT real-world performance.">
        <p className="dim" style={{ fontSize: 13 }}>
          Run and inspect synthetic benchmarks on the <Link to="/benchmark">Benchmark page</Link>.
          The model's published synthetic metrics live in <code>MODEL_CARD.md</code> and are loaded
          from the evaluation artifact — never hardcoded.
        </p>
      </Section>

      {/* 3. Real-world performance */}
      <Section title="Real-World Performance"
        sub="Measured on genuine datasets — never synthetic.">
        {validation?.real_world?.status === 'MEASURED' ? (
          <>
            <div className="tag" style={{ borderColor: '#22c55e66', color: '#22c55e', marginBottom: 8 }}>MEASURED — REAL DATA</div>
            <p className="dim" style={{ fontSize: 13 }}>
              <strong>{validation.real_world.dataset}</strong> ({validation.real_world.dataset_id})<br />
              Experiment {validation.real_world.experiment_id} · n={validation.real_world.n} ·
              Task: {validation.real_world.task}
            </p>
            <table className="data">
              <tbody>
                <tr><td>Inconclusive rate (n=1831)</td><td className="mono">{(validation.real_world.inconclusive_rate * 100).toFixed(1)}%</td></tr>
                <tr><td>Decisive accuracy (n=207)</td><td className="mono">{(validation.real_world.accuracy_decisive * 100).toFixed(1)}% [{(validation.real_world.accuracy_95ci[0] * 100).toFixed(1)}, {(validation.real_world.accuracy_95ci[1] * 100).toFixed(1)}]</td></tr>
                <tr><td>Decisive balanced accuracy (n=207)</td><td className="mono">{(validation.real_world.balanced_accuracy_decisive * 100).toFixed(1)}%</td></tr>
                <tr><td>Decisive precision (n=207)</td><td className="mono">{(validation.real_world.precision_decisive * 100).toFixed(1)}%</td></tr>
                <tr><td>Decisive recall (n=207)</td><td className="mono">{(validation.real_world.recall_decisive * 100).toFixed(1)}%</td></tr>
                <tr><td>Decisive specificity (n=207)</td><td className="mono">{(validation.real_world.specificity_decisive * 100).toFixed(1)}%</td></tr>
                <tr><td>Population FPR (51/921 authentic)</td><td className="mono">{(validation.real_world.population_fpr * 100).toFixed(1)}%</td></tr>
                <tr><td>Population FNR (0/910 spliced)</td><td className="mono">{(validation.real_world.population_fnr * 100).toFixed(1)}%</td></tr>
                <tr><td>Brier (decisive)</td><td className="mono">{validation.real_world.brier_decisive}</td></tr>
                <tr><td>ECE (decisive)</td><td className="mono">{validation.real_world.ece_decisive}</td></tr>
                <tr><td>OOD</td><td className="mono">{validation.real_world.ood}</td></tr>
              </tbody>
            </table>
            <p className="faint" style={{ fontSize: 12 }}>
              {validation.real_world.note}
            </p>
            <p className="faint" style={{ fontSize: 12, fontStyle: 'italic' }}>
              Denominators: "decisive" = non-inconclusive predictions only (n=207);
              "population" = all ground-truth examples; inconclusive rate = over full dataset (n=1831).
            </p>
          </>
        ) : (
          <NotMeasured
            what="Real-world evaluation"
            needed="A real-world dataset (genuine camera photographs + outputs from real AI generators, with provenance and licenses) has not been supplied yet. Nothing is shown here until it is measured — synthetic numbers are never substituted."
          />
        )}
        {validation?.real_world_ai_generation && (
          <div style={{ marginTop: 12 }}>
            <NotMeasured
              what="Real-world AI-generation detection"
              needed={validation.real_world_ai_generation.note ?? "No real-world AI-generated image dataset evaluated."}
            />
          </div>
        )}
      </Section>

      {/* 4. Cross-generator */}
      <Section title="Cross-Generator Generalization"
        sub="Does the detector work on generators it was never trained on?">
        {crossGen ? (
          <>
            <StatusBadge status={crossGen.status ?? 'UNKNOWN'} />
            {crossGen.note && <p className="dim" style={{ fontSize: 13 }}>{crossGen.note}</p>}
          </>
        ) : (
          <NotMeasured
            what="Cross-generator evaluation"
            needed="Requires labeled outputs from at least two distinct generator families, with one family held out of training entirely. See CROSS_GENERATOR.md for the protocol."
          />
        )}
      </Section>

      {/* 5. Hard negatives */}
      <Section title="Hard-Negative Performance"
        sub="How often does the detector wrongly accuse a real photograph?">
        {hnSyn ? (
          <>
            <div className="tag" style={{ borderColor: '#38bdf866', color: '#38bdf8', marginBottom: 8 }}>SYNTHETIC</div>
            <table className="data">
              <tbody>
                <tr><td>n</td><td className="mono">{hnSyn.n ?? '—'}</td></tr>
                <tr><td>Accuracy</td><td className="mono">{hnSyn.accuracy ?? '—'}</td></tr>
                <tr><td>False-positive rate</td><td className="mono">{hnSyn.false_positive_rate ?? '—'}</td></tr>
              </tbody>
            </table>
            <p className="faint" style={{ fontSize: 12 }}>
              Synthetic hard negatives (edited/compressed screenshots of procedural renders).
            </p>
          </>
        ) : null}
        {hardNeg?.question && <p className="dim" style={{ fontSize: 13, fontStyle: 'italic' }}>“{hardNeg.question}”</p>}
        <NotMeasured
          what="Real-world hard negatives"
          needed="Requires authentic photographs processed with real editing software (Photoshop, Lightroom, etc.) — heavily edited real photos must not be accused of being AI."
        />
      </Section>

      {/* 6. OOD */}
      <Section title="OOD Performance" sub="Out-of-distribution behavior of the ML detector.">
        {ood ? (
          <>
            <table className="data">
              <tbody>
                <tr><td>Method</td><td style={{ fontSize: 13 }}>{ood.method ?? '—'}</td></tr>
                <tr><td>Policy</td><td style={{ fontSize: 13 }}>{ood.policy ?? '—'}</td></tr>
                <tr><td>Model status</td><td style={{ fontSize: 13 }}>{ood.model_status ?? '—'}</td></tr>
              </tbody>
            </table>
            {Array.isArray(ood.levels) && ood.levels.length > 0 && (
              <>
                <div className="eyebrow" style={{ marginTop: 12 }}>OOD LEVELS</div>
                <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 6 }}>
                  {ood.levels.map((l: string) => <span key={l} className="tag">{l}</span>)}
                </div>
              </>
            )}
          </>
        ) : (
          <p className="dim">No OOD information available.</p>
        )}
      </Section>

      {/* 7. Robustness */}
      <Section title="Robustness"
        sub="Prediction stability under ordinary image processing (JPEG, resize, blur, screenshots…).">
        <p className="dim" style={{ fontSize: 13 }}>
          Measured on synthetic data only. Run the transform suite on any image from the{' '}
          <Link to="/research">Research page</Link> (POST /api/robustness), or read the measured
          results in <code>ROBUSTNESS.md</code>.
        </p>
      </Section>

      {/* 8. Calibration */}
      <Section title="Calibration" sub="Per-model calibration method and status.">
        {calibModels.length === 0 ? (
          <p className="dim">No calibration records available.</p>
        ) : (
          <table className="data">
            <thead><tr><th>Model</th><th>Version</th><th>Method</th><th>Status</th></tr></thead>
            <tbody>
              {calibModels.map((m: any, i: number) => (
                <tr key={i}>
                  <td className="mono">{m.model ?? '—'}</td>
                  <td className="mono">{m.version ?? '—'}</td>
                  <td style={{ fontSize: 13 }}>{m.calibration ?? m.calibration_detail ?? '—'}</td>
                  <td><StatusBadge status={m.status ?? 'UNKNOWN'} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        )}
      </Section>
    </div>
  );
}
