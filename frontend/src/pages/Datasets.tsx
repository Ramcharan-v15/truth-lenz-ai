import { useEffect, useState } from 'react';
import { getDatasetRegistry, getDataset, getDatasetCard, registerDataset } from '../api';

/**
 * Dataset Registry (V5): real-world dataset identities, license states,
 * quality audits, dataset cards. Everything comes from /api/datasets/registry.
 * No real-world datasets registered yet is the honest empty state.
 */

const LICENSE_COLORS: Record<string, string> = {
  LICENSE_VERIFIED: '#22c55e',
  LICENSE_REQUIRES_REVIEW: '#f59e0b',
  RESTRICTED: '#f43f5e',
  UNKNOWN: '#94a3b8',
};

function LicenseBadge({ status }: { status: string }) {
  const s = (status || 'UNKNOWN').toUpperCase();
  const color = LICENSE_COLORS[s] ?? '#94a3b8';
  return (
    <span className="tag" style={{ borderColor: `${color}66`, color, fontWeight: 700 }}>
      {s.replace(/_/g, ' ')}
    </span>
  );
}

function QualityBadge({ status }: { status: string }) {
  const s = (status || 'UNKNOWN').toUpperCase();
  const color = s === 'READY' ? '#22c55e' : s === 'BLOCKED' ? '#f43f5e' : '#94a3b8';
  return (
    <span className="tag" style={{ borderColor: `${color}66`, color, fontWeight: 700 }}>
      {s}
    </span>
  );
}

export default function Datasets() {
  const [registry, setRegistry] = useState<any[]>([]);
  const [selectedId, setSelectedId] = useState<string | null>(null);
  const [detail, setDetail] = useState<any>(null);
  const [card, setCard] = useState<string | null>(null);
  const [loading, setLoading] = useState(true);
  const [error, setError] = useState<string | null>(null);
  const [form, setForm] = useState({ name: '', source: '', license: '', version: '' });
  const [formMsg, setFormMsg] = useState<string | null>(null);
  const [submitting, setSubmitting] = useState(false);

  const loadRegistry = async () => {
    try {
      const r = await getDatasetRegistry().catch(() => ({ datasets: [] }));
      setRegistry(r?.datasets ?? []);
    } catch (e: any) {
      setError(e?.message || 'Failed to load dataset registry.');
    } finally {
      setLoading(false);
    }
  };

  useEffect(() => { loadRegistry(); }, []);

  useEffect(() => {
    if (!selectedId) { setDetail(null); setCard(null); return; }
    (async () => {
      try {
        const [d, c] = await Promise.all([
          getDataset(selectedId).catch(() => null),
          getDatasetCard(selectedId).catch(() => null),
        ]);
        setDetail(d);
        setCard(c);
      } catch { setDetail(null); setCard(null); }
    })();
  }, [selectedId]);

  const submit = async (e: React.FormEvent) => {
    e.preventDefault();
    if (!form.name.trim()) { setFormMsg('Dataset name is required.'); return; }
    setSubmitting(true);
    setFormMsg(null);
    try {
      const r = await registerDataset({
        name: form.name.trim(),
        source: form.source.trim() || undefined,
        license: form.license.trim() || undefined,
        version: form.version.trim() || undefined,
      });
      setFormMsg(`Registered as ${r.dataset_id ?? r.id ?? 'unknown ID'}. License state defaults to REQUIRES_REVIEW until verified.`);
      setForm({ name: '', source: '', license: '', version: '' });
      await loadRegistry();
    } catch (e: any) {
      setFormMsg(`Registration failed: ${e?.message || 'unknown error'}`);
    } finally {
      setSubmitting(false);
    }
  };

  if (loading) return <div className="page"><p className="dim">Loading dataset registry…</p></div>;
  if (error) return <div className="page"><div className="alert">{error}</div></div>;

  const selected = detail;

  return (
    <div className="page">
      <div className="eyebrow">DATASET PROVENANCE</div>
      <h1>Dataset Registry</h1>
      <p className="section-sub" style={{ marginTop: -8 }}>
        Registered real-world datasets with provenance and license states. Synthetic data is
        tracked separately in the V4 manifests — this registry is for real-world datasets only.
      </p>

      {registry.length === 0 ? (
        <div className="alert" style={{ borderColor: '#f59e0b66', background: 'rgba(245,158,11,0.06)' }}>
          <strong>No real-world datasets registered. REAL-WORLD DATASET REQUIRED.</strong>
          <p className="dim" style={{ fontSize: 13, margin: '8px 0 0' }}>
            Real-world validation cannot begin until a dataset with documented provenance and a
            verified license is registered below. Nothing here is a placeholder — the registry
            is genuinely empty.
          </p>
        </div>
      ) : (
        <div className="panel panel-pad" style={{ marginTop: 16 }}>
          <h2 className="section-title" style={{ marginTop: 0 }}>Registered datasets</h2>
          <table className="data">
            <thead><tr><th>ID</th><th>Name</th><th>Version</th><th>License</th><th>Images</th><th>Quality</th></tr></thead>
            <tbody>
              {registry.map((d: any) => (
                <tr key={d.dataset_id}
                    onClick={() => setSelectedId(d.dataset_id === selectedId ? null : d.dataset_id)}
                    style={{ cursor: 'pointer', background: d.dataset_id === selectedId ? 'rgba(56,189,248,0.06)' : undefined }}>
                  <td className="mono">{d.dataset_id}</td>
                  <td>{d.name ?? '—'}</td>
                  <td className="mono">{d.version ?? '—'}</td>
                  <td><LicenseBadge status={d.license_status} /></td>
                  <td className="mono">{d.image_count ?? '—'}</td>
                  <td><QualityBadge status={d.quality_status} /></td>
                </tr>
              ))}
            </tbody>
          </table>
        </div>
      )}

      {selectedId && (
        <div className="panel panel-pad" style={{ marginTop: 16 }}>
          <h2 className="section-title" style={{ marginTop: 0 }}>
            Dataset detail <span className="mono" style={{ fontSize: 14 }}>{selectedId}</span>
          </h2>
          {selected ? (
            <>
              <table className="data">
                <tbody>
                  {Object.entries(selected)
                    .filter(([k]) => !['quality_audit', 'checks'].includes(k))
                    .map(([k, v]) => (
                      <tr key={k}>
                        <td style={{ width: 220 }} className="faint">{k.replace(/_/g, ' ')}</td>
                        <td style={{ fontSize: 13 }}>{typeof v === 'object' ? JSON.stringify(v) : String(v ?? '—')}</td>
                      </tr>
                    ))}
                </tbody>
              </table>
              {selected.quality_audit && (
                <>
                  <div className="eyebrow" style={{ marginTop: 16 }}>QUALITY AUDIT</div>
                  <p><QualityBadge status={selected.quality_audit.status} /></p>
                  {Array.isArray(selected.quality_audit.checks) && (
                    <table className="data">
                      <thead><tr><th>Check</th><th>Result</th></tr></thead>
                      <tbody>
                        {selected.quality_audit.checks.map((c: any, i: number) => (
                          <tr key={i}>
                            <td style={{ fontSize: 13 }}>{c.name ?? c.check ?? `check ${i + 1}`}</td>
                            <td style={{ fontSize: 13 }}>{String(c.passed ?? c.status ?? c.result ?? '—')}</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  )}
                  {Array.isArray(selected.quality_audit.reasons) && selected.quality_audit.reasons.length > 0 && (
                    <div className="alert" style={{ marginTop: 8 }}>
                      <strong>Blocked reasons</strong>
                      <ul style={{ margin: '8px 0 0', paddingLeft: 18, fontSize: 13 }}>
                        {selected.quality_audit.reasons.map((r: string, i: number) => <li key={i}>{r}</li>)}
                      </ul>
                    </div>
                  )}
                </>
              )}
              {card && (
                <>
                  <div className="eyebrow" style={{ marginTop: 16 }}>DATASET CARD</div>
                  <pre className="faint" style={{ fontSize: 12, whiteSpace: 'pre-wrap', background: 'var(--bg)', padding: 12, borderRadius: 8, border: '1px solid var(--border)' }}>
                    {card}
                  </pre>
                </>
              )}
            </>
          ) : (
            <p className="dim">Could not load dataset detail.</p>
          )}
        </div>
      )}

      <div className="panel panel-pad" style={{ marginTop: 16 }}>
        <h2 className="section-title" style={{ marginTop: 0 }}>Register a dataset</h2>
        <p className="dim" style={{ fontSize: 13 }}>
          Registers dataset metadata and provenance only — image content is never uploaded here.
          New registrations default to license state REQUIRES_REVIEW until verified.
        </p>
        <form onSubmit={submit} style={{ display: 'grid', gap: 10, maxWidth: 520, marginTop: 12 }}>
          <label className="faint" style={{ fontSize: 13 }}>Name *
            <input type="text" value={form.name} onChange={(e) => setForm({ ...form, name: e.target.value })} placeholder="e.g. Camera photos — batch 1" style={{ width: '100%', marginTop: 4 }} />
          </label>
          <label className="faint" style={{ fontSize: 13 }}>Source
            <input type="text" value={form.source} onChange={(e) => setForm({ ...form, source: e.target.value })} placeholder="e.g. contributed by research partner" style={{ width: '100%', marginTop: 4 }} />
          </label>
          <label className="faint" style={{ fontSize: 13 }}>License
            <input type="text" value={form.license} onChange={(e) => setForm({ ...form, license: e.target.value })} placeholder="e.g. CC-BY-4.0, research-only" style={{ width: '100%', marginTop: 4 }} />
          </label>
          <label className="faint" style={{ fontSize: 13 }}>Version
            <input type="text" value={form.version} onChange={(e) => setForm({ ...form, version: e.target.value })} placeholder="e.g. 1.0" style={{ width: '100%', marginTop: 4 }} />
          </label>
          <button className="btn" type="submit" disabled={submitting} style={{ justifySelf: 'start' }}>
            {submitting ? 'Registering…' : 'Register dataset'}
          </button>
          {formMsg && <p className="dim" style={{ fontSize: 13 }}>{formMsg}</p>}
        </form>
      </div>
    </div>
  );
}
