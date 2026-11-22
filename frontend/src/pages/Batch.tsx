import { useMemo, useState } from 'react';
import UploadZone, { type PickedFile } from '../components/UploadZone';
import { batchAnalyze } from '../api';
import { VERDICT_META, fmtBytes } from '../types';

interface Row {
  filename: string; analysis_id?: string; verdict?: string;
  ai_probability?: number; manipulation_probability?: number;
  confidence?: number; confidence_label?: string; image_quality?: string;
  dimensions?: string; error?: string;
}

export default function Batch() {
  const [files, setFiles] = useState<PickedFile[]>([]);
  const [rows, setRows] = useState<Row[] | null>(null);
  const [loading, setLoading] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [query, setQuery] = useState('');
  const [verdictFilter, setVerdictFilter] = useState('all');
  const [sortKey, setSortKey] = useState<'filename' | 'ai_probability' | 'manipulation_probability' | 'confidence'>('ai_probability');
  const [sortDir, setSortDir] = useState<1 | -1>(-1);

  const run = async () => {
    setLoading(true); setError(null); setRows(null);
    try {
      const r = await batchAnalyze(files.map((f) => f.file));
      setRows(r.results);
    } catch (e: any) { setError(e.message); }
    finally { setLoading(false); }
  };

  const filtered = useMemo(() => {
    if (!rows) return [];
    let out = rows.filter((r) =>
      r.filename.toLowerCase().includes(query.toLowerCase()) &&
      (verdictFilter === 'all' || r.verdict === verdictFilter));
    out = [...out].sort((x, y) => {
      // Stage 1 (L12): missing values sort LAST in either direction and are
      // never coerced into a measurement ('', 0). Valid-number ordering and
      // the chosen sort direction are unchanged; ties keep input order
      // (Array.sort is stable), so the result is deterministic.
      const xv = (x as any)[sortKey]; const yv = (y as any)[sortKey];
      const xn = xv === null || xv === undefined; const yn = yv === null || yv === undefined;
      if (xn && yn) return 0;
      if (xn) return 1;
      if (yn) return -1;
      return (xv > yv ? 1 : xv < yv ? -1 : 0) * sortDir;
    });
    return out;
  }, [rows, query, verdictFilter, sortKey, sortDir]);

  const csvCell = (v: unknown): string => {
    // Missing values render as 'unavailable', never blank (Stage 1
    // null-safety: a blank cell is ambiguous with an empty value).
    // Cells containing a comma, quote, or newline are quoted, with
    // quotes doubled, so a filename containing a comma cannot break
    // the CSV structure or shift later columns.
    if (v === null || v === undefined) return 'unavailable';
    const s = String(v);
    if (s.includes(',') || s.includes('"') || s.includes('\n')) return `"${s.replace(/"/g, '""')}"`;
    return s;
  };

  const downloadCsv = () => {
    if (!rows) return;
    const head = 'filename,verdict,ai_probability,manipulation_probability,confidence,image_quality,dimensions';
    const lines = rows.map((r) => [r.filename, r.verdict, r.ai_probability, r.manipulation_probability, r.confidence, r.image_quality, r.dimensions].map(csvCell).join(','));
    const blob = new Blob([[head, ...lines].join('\n')], { type: 'text/csv' });
    const url = URL.createObjectURL(blob);
    const a = document.createElement('a');
    a.href = url; a.download = 'truthlenz-batch.csv'; a.click();
    URL.revokeObjectURL(url);
  };

  const th = (label: string, key: typeof sortKey) => (
    <th className="sortable" onClick={() => { setSortKey(key); setSortDir(sortKey === key ? (sortDir === 1 ? -1 : 1) : -1); }}>
      {label} {sortKey === key ? (sortDir === 1 ? '▲' : '▼') : ''}
    </th>
  );

  return (
    <div className="shell" style={{ paddingTop: 30, paddingBottom: 20 }}>
      <div className="eyebrow">BULK FORENSICS</div>
      <h1 style={{ margin: '0 0 8px', fontSize: 30 }}>Batch Analysis</h1>
      <p className="section-sub">Analyze up to 8 images at once. Sort, filter, search and export the results table.</p>

      <UploadZone multiple onPick={(f) => setFiles((prev) => [...prev, ...f].slice(0, 8))} compact />

      {files.length > 0 && (
        <div style={{ marginTop: 14 }}>
          <div style={{ display: 'flex', gap: 8, flexWrap: 'wrap', marginBottom: 12 }}>
            {files.map((f, i) => (
              <div key={i} className="file-row" style={{ padding: '6px 10px' }}>
                <img className="thumb" src={f.previewUrl} alt="" style={{ width: 36, height: 36 }} />
                <span style={{ fontSize: 12.5 }}>{f.file.name} <span className="faint">({fmtBytes(f.file.size)})</span></span>
                <button className="btn btn-ghost btn-sm" onClick={() => { URL.revokeObjectURL(files[i].previewUrl); setFiles(files.filter((_, j) => j !== i)); }}>✕</button>
              </div>
            ))}
          </div>
          <button className="btn btn-primary" disabled={loading} onClick={run}>
            {loading ? 'Analyzing batch…' : `Analyze ${files.length} image${files.length > 1 ? 's' : ''}`}
          </button>
        </div>
      )}
      {error && <div className="alert error" style={{ marginTop: 12 }}>{error}</div>}
      {loading && <div className="skeleton" style={{ height: 200, marginTop: 16 }} />}

      {rows && (
        <div style={{ marginTop: 20 }}>
          <div style={{ display: 'flex', gap: 10, marginBottom: 12, flexWrap: 'wrap' }}>
            <input type="search" placeholder="Search filename…" value={query} onChange={(e) => setQuery(e.target.value)} style={{ maxWidth: 260 }} aria-label="Search results" />
            <select value={verdictFilter} onChange={(e) => setVerdictFilter(e.target.value)} style={{ maxWidth: 220 }} aria-label="Filter by verdict">
              <option value="all">All verdicts</option>
              {Object.keys(VERDICT_META).map((v) => <option key={v} value={v}>{VERDICT_META[v as keyof typeof VERDICT_META].label}</option>)}
            </select>
            <button className="btn btn-sm" onClick={downloadCsv}>Download CSV</button>
          </div>
          <div className="panel" style={{ overflowX: 'auto' }}>
            <table className="data">
              <thead><tr>
                {th('Filename', 'filename')}
                <th>Verdict</th>
                {th('AI %', 'ai_probability')}
                {th('Manip %', 'manipulation_probability')}
                {th('Confidence', 'confidence')}
                <th>Quality</th><th>Dimensions</th>
              </tr></thead>
              <tbody>
                {filtered.map((r, i) => {
                  const m = r.verdict ? VERDICT_META[r.verdict as keyof typeof VERDICT_META] : null;
                  return (
                    <tr key={i}>
                      <td style={{ maxWidth: 220, overflow: 'hidden', textOverflow: 'ellipsis' }}>{r.filename}</td>
                      <td>{r.error ? <span className="tag suspicious">error</span> : m && <span className="tag" style={{ color: m.color, borderColor: m.color + '66' }}>{m.label}</span>}</td>
                      <td className="mono">{r.ai_probability?.toFixed(0) ?? '—'}</td>
                      <td className="mono">{r.manipulation_probability?.toFixed(0) ?? '—'}</td>
                      <td className="mono">{r.confidence?.toFixed(0) ?? '—'} <span className="faint">{r.confidence_label}</span></td>
                      <td>{r.image_quality ?? '—'}</td>
                      <td className="mono faint">{r.dimensions ?? '—'}</td>
                    </tr>
                  );
                })}
              </tbody>
            </table>
          </div>
          {filtered.length === 0 && <div className="empty">No results match your filters.</div>}
        </div>
      )}
    </div>
  );
}
