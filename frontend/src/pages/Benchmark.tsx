import { useCallback, useEffect, useState } from 'react';
import { runBenchmark, getBenchmarkResults, getBenchmarks } from '../api';

interface CalBin { n: number; accuracy: number | null; mean_confidence: number | null }
interface Calibration { ece: number | null; n_bins: number; bins: CalBin[] }
interface TaskMetrics {
  n: number; threshold: number | null;
  accuracy: number | null; precision: number | null; recall: number | null;
  f1: number | null; specificity: number | null;
  false_positive_rate: number | null; false_negative_rate: number | null;
  tp: number | null; tn: number | null; fp: number | null; fn: number | null;
  roc_auc: number | null; pr_auc: number | null;
  calibration: Calibration | null;
  note?: string;
}
interface BenchmarkResult {
  benchmark: string; dataset: string; dataset_version: string;
  dataset_fingerprint: string; evaluated_at: string;
  n_images: number; n_failed: number;
  categories: Record<string, number>;
  tasks: Record<'ai_detection' | 'manipulation_detection', {
    overall: TaskMetrics;
    by_category: Record<string, TaskMetrics | { n: number; note: string } | null>;
  }>;
  elapsed_s: number; note?: string;
}

type CategoryEntry = TaskMetrics | { n: number; note: string } | null;
function isFullMetrics(e: CategoryEntry): e is TaskMetrics {
  return !!e && (e as TaskMetrics).accuracy !== undefined && !(e as { note?: string }).note;
}

const fmt = (v: number | null | undefined, digits = 3) =>
  v === null || v === undefined || Number.isNaN(v) ? 'n/a' : Number(v).toFixed(digits);
const fmtPct = (v: number | null | undefined) =>
  v === null || v === undefined || Number.isNaN(v) ? 'n/a' : `${(Number(v) * 100).toFixed(1)}%`;
const fmtDate = (s?: string) => {
  if (!s) return '—';
  const d = new Date(s);
  return isNaN(d.getTime()) ? s : d.toLocaleString();
};
const fpShort = (s?: string) => (s ? s.slice(0, 12) : '—');

const CATEGORY_ORDER = ['authentic', 'ai_generated', 'manipulated', 'screenshots', 'compressed'];
const CATEGORY_LABEL: Record<string, string> = {
  authentic: 'Authentic', ai_generated: 'AI-generated', manipulated: 'Manipulated',
  screenshots: 'Screenshots', compressed: 'Compressed',
};

function MetricCard({ label, value, hint }: { label: string; value: string; hint?: string }) {
  return (
    <div className="panel panel-pad metric" style={{ minWidth: 0 }}>
      <div className="lbl">{label}</div>
      <div className="val" style={{ color: value === 'n/a' ? 'var(--text-faint)' : 'var(--text)' }}>{value}</div>
      {hint && <div className="faint" style={{ fontSize: 11.5, marginTop: 2 }}>{hint}</div>}
    </div>
  );
}

function ConfusionMatrix({ m }: { m: TaskMetrics }) {
  const cells: { key: string; label: string; val: number | null; desc: string }[] = [
    { key: 'tp', label: 'TP', val: m.tp, desc: 'Correctly flagged' },
    { key: 'fn', label: 'FN', val: m.fn, desc: 'Missed' },
    { key: 'fp', label: 'FP', val: m.fp, desc: 'Wrongly flagged' },
    { key: 'tn', label: 'TN', val: m.tn, desc: 'Correctly cleared' },
  ];
  return (
    <div>
      <div style={{ display: 'grid', gridTemplateColumns: '1fr 1fr', gap: 8, maxWidth: 520 }}>
        {cells.map((c) => (
          <div key={c.key} className="panel panel-pad" style={{
            borderLeft: c.key === 'fp' || c.key === 'fn'
              ? '3px solid rgba(248,113,113,0.55)' : '3px solid rgba(52,211,153,0.55)',
          }}>
            <div className="eyebrow">{c.label} <span className="faint">· {c.desc}</span></div>
            <div style={{ fontSize: 26, fontWeight: 800, fontVariantNumeric: 'tabular-nums' }}>
              {c.val === null || c.val === undefined ? 'n/a' : c.val}
            </div>
          </div>
        ))}
      </div>
      <div className="faint" style={{ fontSize: 12, marginTop: 8, maxWidth: 520 }}>
        Rows: actual class (positive = flagged class for this task). Columns: prediction.
        Threshold: {m.threshold === null || m.threshold === undefined ? 'n/a' : Number(m.threshold).toFixed(2)}.
      </div>
    </div>
  );
}

function CalibrationChart({ cal }: { cal: Calibration | null }) {
  if (!cal || !Array.isArray(cal.bins) || cal.bins.length === 0) {
    return <p className="dim">No calibration data.</p>;
  }
  return (
    <div style={{ maxWidth: 640 }}>
      <div className="eyebrow" style={{ marginBottom: 8 }}>
        Calibration reliability — mean predicted confidence vs observed accuracy per bin
      </div>
      <div style={{ display: 'flex', flexDirection: 'column', gap: 6 }}>
        {cal.bins.map((b, i) => {
          const conf = b.mean_confidence ?? 0;
          const acc = b.accuracy ?? 0;
          const known = b.mean_confidence !== null && b.accuracy !== null;
          return (
            <div key={i} style={{ display: 'grid', gridTemplateColumns: '52px 1fr 120px', gap: 10, alignItems: 'center' }}>
              <div className="faint" style={{ fontSize: 11.5, fontVariantNumeric: 'tabular-nums' }}>
                bin {i + 1} · n={b.n}
              </div>
              <div style={{ position: 'relative', height: 22 }}>
                <div style={{
                  position: 'absolute', inset: 0, background: 'var(--accent-dim)',
                  border: '1px solid var(--border-strong)', borderRadius: 4,
                  width: `${Math.max(0, Math.min(1, conf)) * 100}%`,
                }} />
                <div style={{
                  position: 'absolute', inset: 0, background: 'transparent',
                  borderLeft: `3px solid ${known ? (Math.abs(conf - acc) > 0.15 ? '#f87171' : '#34d399') : 'var(--text-faint)'}`,
                  borderRadius: 2, width: `${Math.max(0, Math.min(1, acc)) * 100}%`,
                }} title={`observed accuracy ${(acc * 100).toFixed(1)}%`} />
              </div>
              <div className="faint" style={{ fontSize: 11.5, fontVariantNumeric: 'tabular-nums' }}>
                conf {fmt(b.mean_confidence)} · acc {fmt(b.accuracy)}
              </div>
            </div>
          );
        })}
      </div>
      <div className="faint" style={{ fontSize: 12, marginTop: 10 }}>
        <span style={{ display: 'inline-block', width: 22, height: 10, background: 'var(--accent-dim)', border: '1px solid var(--border-strong)', borderRadius: 3, marginRight: 6 }} />
        predicted confidence &nbsp;&nbsp;
        <span style={{ display: 'inline-block', width: 3, height: 12, background: '#34d399', marginRight: 6, verticalAlign: -1 }} />
        observed accuracy (green = well calibrated, red = gap &gt; 0.15).
        Closer alignment between the bar and the marker means better-calibrated probabilities.
      </div>
    </div>
  );
}

export default function Benchmark() {
  const [dataset, setDataset] = useState('sample');
  const [version, setVersion] = useState('1.0');
  const [results, setResults] = useState<BenchmarkResult | null>(null);
  const [fetching, setFetching] = useState(false);
  const [running, setRunning] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [tab, setTab] = useState<'ai_detection' | 'manipulation_detection'>('ai_detection');

  const fetchResults = useCallback(async (ds: string) => {
    setFetching(true);
    setError(null);
    try {
      const r = await getBenchmarkResults(ds);
      setResults(r as BenchmarkResult);
    } catch (e: any) {
      setResults(null);
      setError(null); // 404 = no results yet → honest empty state, not an error
    } finally {
      setFetching(false);
    }
  }, []);

  useEffect(() => { fetchResults('sample'); }, [fetchResults]);

  const handleRun = async () => {
    if (!dataset.trim()) { setError('Enter a dataset name.'); return; }
    setRunning(true);
    setError(null);
    try {
      const r = await runBenchmark(dataset.trim(), version.trim() || '1.0');
      setResults(r as BenchmarkResult);
    } catch (e: any) {
      setError(e?.message || 'Benchmark run failed.');
    } finally {
      setRunning(false);
    }
  };

  const task = results?.tasks?.[tab];
  const overall = task?.overall;
  const cal = overall?.calibration ?? null;

  const categories = task?.by_category ?? {};
  const orderedCats = [
    ...CATEGORY_ORDER.filter((c) => c in categories),
    ...Object.keys(categories).filter((c) => !CATEGORY_ORDER.includes(c)),
  ];

  return (
    <div className="shell" style={{ paddingTop: 30, paddingBottom: 20 }}>
      <div className="eyebrow">DETECTOR PERFORMANCE · RESEARCH</div>
      <h1 style={{ margin: '0 0 8px', fontSize: 30 }}>Benchmark Dashboard</h1>
      <p className="dim" style={{ maxWidth: 720 }}>
        Measured performance of the detection pipeline on a labelled evaluation set.
        Metrics here describe how the detector <em>performed on this dataset</em> — they are not
        guarantees about any single image.
      </p>

      {/* Controls */}
      <div className="panel panel-pad" style={{ marginBottom: 18 }}>
        <div className="eyebrow">Evaluation control</div>
        <div style={{ display: 'flex', gap: 10, flexWrap: 'wrap', alignItems: 'end' }}>
          <label style={{ display: 'flex', flexDirection: 'column', gap: 6, fontSize: 12.5 }}>
            <span className="faint">Dataset</span>
            <input value={dataset} onChange={(e) => setDataset(e.target.value)}
              placeholder="sample" style={{ padding: '9px 12px', borderRadius: 8, border: '1px solid var(--border-strong)', background: 'var(--bg)', color: 'var(--text)', minWidth: 200 }} />
          </label>
          <label style={{ display: 'flex', flexDirection: 'column', gap: 6, fontSize: 12.5 }}>
            <span className="faint">Version</span>
            <input value={version} onChange={(e) => setVersion(e.target.value)}
              placeholder="1.0" style={{ padding: '9px 12px', borderRadius: 8, border: '1px solid var(--border-strong)', background: 'var(--bg)', color: 'var(--text)', width: 110 }} />
          </label>
          <button className="btn-ghost" onClick={() => fetchResults(dataset.trim() || 'sample')}
            disabled={fetching || running} style={{ alignSelf: 'end' }}>
            {fetching ? 'Loading…' : 'Load results'}
          </button>
          <button className="btn" onClick={handleRun} disabled={running || fetching}
            style={{ alignSelf: 'end', padding: '9px 18px' }}>
            {running ? 'Running benchmark…' : 'Run benchmark'}
          </button>
        </div>
        <p className="faint" style={{ fontSize: 12.5, marginBottom: 0 }}>
          A full run re-analyzes every image in the dataset and can take a while — keep this page open.
          Results are stored per dataset and replace any previous run for the same dataset.
        </p>
        {error && <div className="demo-banner" style={{ marginTop: 12 }}>{error}</div>}
      </div>

      {fetching && !results && <p className="dim">Loading benchmark results…</p>}

      {/* Honest empty state — never placeholder numbers */}
      {!fetching && !running && !results && (
        <div className="panel panel-pad" style={{ textAlign: 'center', padding: '56px 24px' }}>
          <div style={{ fontSize: 40, marginBottom: 12 }}>📊</div>
          <h2 style={{ margin: '0 0 8px' }}>No benchmark data</h2>
          <p className="dim" style={{ maxWidth: 560, margin: '0 auto 18px' }}>
            There are no evaluation results for dataset “{dataset}” yet. Run an evaluation to
            produce metrics. <strong>Never quote performance claims without an actual run.</strong>
          </p>
          <button className="btn" onClick={handleRun} disabled={running}>
            Run benchmark on “{dataset}”
          </button>
        </div>
      )}

      {results && (
        <>
          {/* Caution banner */}
          {results.note && (
            <div className="demo-banner" style={{ marginBottom: 16 }}>
              <strong>⚠ Evaluation caveat:</strong> {results.note}
            </div>
          )}

          {/* Header strip */}
          <div className="panel panel-pad" style={{ marginBottom: 18 }}>
            <div className="eyebrow">Evaluation record</div>
            <dl className="kv">
              <div style={{ display: 'contents' }}><dt>Dataset</dt><dd>{results.dataset} · v{results.dataset_version}</dd></div>
              <div style={{ display: 'contents' }}><dt>Evaluated</dt><dd>{fmtDate(results.evaluated_at)}</dd></div>
              <div style={{ display: 'contents' }}><dt>Images</dt><dd>{results.n_images}{results.n_failed ? ` (${results.n_failed} failed)` : ''}</dd></div>
              <div style={{ display: 'contents' }}><dt>Fingerprint</dt><dd>{fpShort(results.dataset_fingerprint)}</dd></div>
              <div style={{ display: 'contents' }}><dt>Elapsed</dt><dd>{results.elapsed_s?.toFixed(1)}s</dd></div>
            </dl>
            <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginTop: 12 }}>
              {Object.entries(results.categories ?? {}).map(([c, n]) => (
                <span key={c} className="chip">
                  {CATEGORY_LABEL[c] ?? c}: <strong>{n}</strong>
                </span>
              ))}
            </div>
          </div>

          {/* Task tabs */}
          <div className="viewer-tabs">
            {(['ai_detection', 'manipulation_detection'] as const).map((t) => (
              <button key={t} className={tab === t ? 'active' : ''} onClick={() => setTab(t)}
                aria-pressed={tab === t}>
                {t === 'ai_detection' ? 'AI detection' : 'Manipulation detection'}
              </button>
            ))}
          </div>

          {overall ? (
            <>
              {/* Metric cards */}
              <div style={{ display: 'grid', gridTemplateColumns: 'repeat(auto-fill, minmax(160px, 1fr))', gap: 12, marginBottom: 18 }}>
                <MetricCard label="Accuracy" value={fmt(overall.accuracy)} hint={`n=${overall.n}`} />
                <MetricCard label="Precision" value={fmt(overall.precision)} />
                <MetricCard label="Recall" value={fmt(overall.recall)} />
                <MetricCard label="F1 score" value={fmt(overall.f1)} />
                <MetricCard label="ROC-AUC" value={fmt(overall.roc_auc)} />
                <MetricCard label="PR-AUC" value={fmt(overall.pr_auc)} />
                <MetricCard label="False-positive rate" value={fmtPct(overall.false_positive_rate)} />
                <MetricCard label="False-negative rate" value={fmtPct(overall.false_negative_rate)} />
                <MetricCard label="Specificity" value={fmt(overall.specificity)} />
                <MetricCard label="Calibration error (ECE)" value={fmt(cal?.ece, 4)} hint={`${cal?.n_bins ?? '—'} bins`} />
              </div>

              <h2 className="section-title">Confusion matrix</h2>
              <div style={{ marginBottom: 18 }}>
                <ConfusionMatrix m={overall} />
              </div>

              <h2 className="section-title">Per-category performance</h2>
              <p className="section-sub">
                Where the detector fails matters. A model that scores well overall but collapses on
                screenshots is not a reliable detector.
              </p>
              <div className="panel panel-pad" style={{ padding: 0, overflowX: 'auto', marginBottom: 18 }}>
                <table style={{ width: '100%', borderCollapse: 'collapse', fontSize: 13 }}>
                  <thead>
                    <tr style={{ textAlign: 'left', borderBottom: '1px solid var(--border-strong)' }}>
                      {['Category', 'n', 'Accuracy', 'F1', 'ROC-AUC', 'Note'].map((h) => (
                        <th key={h} style={{ padding: '12px 14px', color: 'var(--text-faint)', fontWeight: 600, fontSize: 11, letterSpacing: '0.08em', textTransform: 'uppercase' }}>{h}</th>
                      ))}
                    </tr>
                  </thead>
                  <tbody>
                    {orderedCats.map((c) => {
                      const e = categories[c];
                      return (
                        <tr key={c} style={{ borderBottom: '1px solid var(--border)' }}>
                          <td style={{ padding: '10px 14px', fontWeight: 600 }}>{CATEGORY_LABEL[c] ?? c}</td>
                          {e === null ? (
                            <td colSpan={4} style={{ padding: '10px 14px' }} className="faint">no data</td>
                          ) : isFullMetrics(e) ? (
                            <>
                              <td style={{ padding: '10px 14px', fontVariantNumeric: 'tabular-nums' }}>{e.n}</td>
                              <td style={{ padding: '10px 14px', fontVariantNumeric: 'tabular-nums' }}>{fmt(e.accuracy)}</td>
                              <td style={{ padding: '10px 14px', fontVariantNumeric: 'tabular-nums' }}>{fmt(e.f1)}</td>
                              <td style={{ padding: '10px 14px', fontVariantNumeric: 'tabular-nums' }}>{fmt(e.roc_auc)}</td>
                              <td style={{ padding: '10px 14px' }} className="faint">—</td>
                            </>
                          ) : (
                            <td colSpan={4} style={{ padding: '10px 14px' }} className="faint">
                              n={e.n} — {e.note ?? 'insufficient samples'}
                            </td>
                          )}
                        </tr>
                      );
                    })}
                    {orderedCats.length === 0 && (
                      <tr><td colSpan={6} style={{ padding: 16 }} className="faint">No per-category data.</td></tr>
                    )}
                  </tbody>
                </table>
              </div>

              <h2 className="section-title">Calibration</h2>
              <p className="section-sub">
                A probability of 0.8 should be right about 80% of the time. This chart compares
                predicted confidence against observed accuracy in each probability bin.
              </p>
              <div className="panel panel-pad" style={{ marginBottom: 18 }}>
                <CalibrationChart cal={cal} />
              </div>

              <div className="disclaimer-strip">
                These metrics describe past performance on dataset “{results.dataset}” (fingerprint {fpShort(results.dataset_fingerprint)}).
                They do not guarantee future accuracy on different images, generators, or
                post-processing. Report issues or suggest datasets via the usual channels.
              </div>
            </>
          ) : (
            <p className="dim">No task data for “{tab}”.</p>
          )}
        </>
      )}

      {running && (
        <div className="panel panel-pad" style={{ marginTop: 16 }}>
          <div className="eyebrow">Running…</div>
          <p className="dim" style={{ marginBottom: 0 }}>
            The benchmark is re-analyzing the dataset on the server. This page will show the new
            results when the run completes.
          </p>
        </div>
      )}

      <RunComparison />
    </div>
  );
}

/** Model-to-model / run-to-run comparison (V3, Phase 24). */
function RunComparison() {
  const [runs, setRuns] = useState<any[]>([]);
  const [selA, setSelA] = useState('');
  const [selB, setSelB] = useState('');
  const [dataA, setDataA] = useState<any>(null);
  const [dataB, setDataB] = useState<any>(null);

  useEffect(() => {
    getBenchmarks().then((r) => {
      setRuns(r.benchmarks ?? []);
      if ((r.benchmarks ?? []).length >= 1) setSelA(r.benchmarks[0].dataset);
      if ((r.benchmarks ?? []).length >= 2) setSelB(r.benchmarks[1].dataset);
    }).catch(() => {});
  }, []);

  useEffect(() => {
    if (selA) getBenchmarkResults(selA).then(setDataA).catch(() => setDataA(null));
  }, [selA]);
  useEffect(() => {
    if (selB) getBenchmarkResults(selB).then(setDataB).catch(() => setDataB(null));
  }, [selB]);

  if (runs.length < 2) return null;
  const rows: Array<{ label: string; key: string }> = [
    { label: 'Accuracy', key: 'accuracy' }, { label: 'F1', key: 'f1' },
    { label: 'ROC-AUC', key: 'roc_auc' }, { label: 'PR-AUC', key: 'pr_auc' },
    { label: 'False-positive rate', key: 'false_positive_rate' },
    { label: 'False-negative rate', key: 'false_negative_rate' },
    { label: 'Calibration error', key: 'calibration' },
  ];
  const val = (d: any, k: string) => {
    const o = d?.tasks?.ai_detection?.overall;
    if (!o) return 'n/a';
    if (k === 'calibration') return o.calibration?.ece ?? 'n/a';
    return o[k] ?? 'n/a';
  };

  return (
    <div style={{ marginTop: 24 }}>
      <h2 className="section-title">Compare evaluations</h2>
      <p className="section-sub">Side-by-side metrics from saved benchmark runs (AI-detection task).</p>
      <div className="panel panel-pad">
        <div style={{ display: 'flex', gap: 12, marginBottom: 12, flexWrap: 'wrap' }}>
          <label className="faint" style={{ fontSize: 12 }}>Run A
            <select value={selA} onChange={(e) => setSelA(e.target.value)} style={{ marginLeft: 6 }}>
              {runs.map((r) => <option key={r.dataset} value={r.dataset}>{r.dataset} (v{r.dataset_version})</option>)}
            </select>
          </label>
          <label className="faint" style={{ fontSize: 12 }}>Run B
            <select value={selB} onChange={(e) => setSelB(e.target.value)} style={{ marginLeft: 6 }}>
              {runs.map((r) => <option key={r.dataset} value={r.dataset}>{r.dataset} (v{r.dataset_version})</option>)}
            </select>
          </label>
        </div>
        <table className="data">
          <thead><tr><th>Metric</th><th>{selA}</th><th>{selB}</th></tr></thead>
          <tbody>
            {rows.map(function (r) {
              return (
                <tr key={r.key}>
                  <td>{r.label}</td>
                  <td className="mono">{String(val(dataA, r.key))}</td>
                  <td className="mono">{String(val(dataB, r.key))}</td>
                </tr>
              );
            })}
          </tbody>
        </table>
        <p className="faint" style={{ fontSize: 12, marginTop: 8 }}>
          Compare runs across dataset versions or configurations. Only runs from actual
          evaluations appear here — never hardcoded numbers.
        </p>
      </div>

      <BiasPanel data={dataA} name={selA} />
    </div>
  );
}

/** Bias & limitations analysis display (V3, Phase 25). */
function BiasPanel({ data, name }: { data: any; name: string }) {
  const bias = data?.bias_analysis;
  if (!bias) return null;
  const attrs = Object.entries(bias).filter(([, g]: any) => Object.keys(g).length > 0);
  if (!attrs.length) return null;
  return (
    <div style={{ marginTop: 18 }}>
      <h2 className="section-title">Bias &amp; limitations</h2>
      <p className="section-sub">
        How accuracy varies across image attributes for “{name}”.
        A detector that is strong overall but weak on screenshots has a documented limitation, not a hidden one.
      </p>
      <div className="grid grid-2">
        {attrs.map(([attr, groups]: any) => (
          <div key={attr} className="panel panel-pad">
            <div className="eyebrow">{String(attr).replace(/_/g, ' ')}</div>
            <table className="data" style={{ marginTop: 8 }}>
              <thead><tr><th>Group</th><th>n</th><th>Accuracy</th><th>F1</th></tr></thead>
              <tbody>
                {Object.entries(groups).map(([g, m]: any) => (
                  <tr key={g}>
                    <td>{g}</td><td className="mono">{m.n}</td>
                    <td className="mono">{m.accuracy ?? 'n/a'}</td>
                    <td className="mono">{m.f1 ?? 'n/a'}</td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        ))}
      </div>
    </div>
  );
}
