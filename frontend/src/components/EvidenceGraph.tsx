import { useMemo, useState } from 'react';
import type { AnalysisReport, MLDetector, Signal } from '../types';
import { VERDICT_META } from '../types';

interface Node {
  id: string;
  label: string;
  sub: string;
  kind: 'verdict' | 'branch' | 'signal' | 'detector' | 'provenance' | 'ood' | 'calibration' | 'attribution';
  x: number;
  y: number;
  payload?: Signal | MLDetector | any;
}

const W = 960;

/**
 * Visual evidence graph (Phase 14): FINAL VERDICT -> Forensics / ML / Provenance
 * -> individual signals. Click any node for score, confidence, evidence,
 * technical detail and limitations.
 */
export default function EvidenceGraph({ report }: { report: AnalysisReport }) {
  const [selected, setSelected] = useState<string | null>(null);
  const ens = report.ensemble;
  const meta = VERDICT_META[ens.verdict];

  const { nodes, edges } = useMemo(() => {
    const nodes: Node[] = [];
    const edges: [string, string][] = [];
    const signals = report.signals.filter((s) => s.status !== 'unable');
    // Historical reports (pre-Stage-2 suppression flags) carry
    // ml_suppressed_reason while their detector entries still say
    // available=true. A valued entry under a suppression reason is a
    // withheld output, not a ready detector — exclude it from the
    // detector state/value rendering, exactly as the current
    // suppressed flag excludes it. (The ML branch label and the
    // empty-detector leaf already key on the reason.)
    const historicalSuppressed = Boolean((ens as any).ml_suppressed_reason);
    const dets = ens.detectors?.filter(
      (d) => d.available && !(historicalSuppressed && d.probability_ai != null)) ?? [];

    nodes.push({ id: 'verdict', label: 'FINAL VERDICT', sub: meta.label, kind: 'verdict', x: W / 2, y: 46 });

    const branches = [
      { id: 'forensics', label: 'FORENSICS', sub: `weight ${ens.hybrid_weights?.forensics ?? 1}`, kind: 'branch' as const },
      { id: 'ml', label: 'ML MODELS', sub: (ens as any).ml_suppressed_reason ? 'output withheld' : (dets.length ? `${dets.length} ready` : 'not configured'), kind: 'branch' as const },
      { id: 'prov', label: 'PROVENANCE', sub: ens.provenance?.note?.slice(0, 34) ?? '', kind: 'branch' as const },
    ];
    branches.forEach((b, i) => {
      const x = W * (0.18 + 0.32 * i);
      nodes.push({ ...b, x, y: 150 });
      edges.push(['verdict', b.id]);
    });

    // signal leaves under forensics
    const n = Math.min(signals.length, 10);
    signals.slice(0, 10).forEach((s, i) => {
      const x = 60 + (i * (W - 120)) / Math.max(n - 1, 1);
      const id = `sig-${s.name}`;
      nodes.push({ id, label: s.display_name, sub: `AI ${s.ai_score.toFixed(0)} · ${s.confidence.toFixed(0)}%`, kind: 'signal', x, y: 268, payload: s });
      edges.push(['forensics', id]);
    });

    // detector leaves under ML
    dets.forEach((d, i) => {
      const id = `det-${i}`;
      nodes.push({
        id, label: d.detector_name.slice(0, 26), kind: 'detector', x: W * 0.5, y: 268,
        sub: `P(AI) ${d.probability_ai}% · ${d.calibration}`, payload: d,
      });
      edges.push(['ml', id]);
    });

    // V4 nodes: OOD, Calibration, Generator Attribution (spec §23)
    const oodStatus = (ens as any).ml_ood_status ?? 'UNKNOWN';
    nodes.push({
      id: 'ood', label: 'OOD STATUS', sub: oodStatus.replace(/_/g, ' '),
      kind: 'ood', x: W * 0.82, y: 150,
      payload: { ood_status: oodStatus, note: 'Out-of-distribution: input vs training distribution' },
    });
    edges.push(['ml', 'ood']);

    const calMethod = dets[0]?.calibration ?? 'none';
    nodes.push({
      id: 'cal', label: 'CALIBRATION', sub: String(calMethod).slice(0, 30),
      kind: 'calibration', x: W * 0.5, y: 96,
      payload: { method: calMethod, note: 'Raw scores never presented as probabilities' },
    });
    edges.push(['ml', 'cal']);

    const genAttr = (report as any).generator_attribution;
    nodes.push({
      id: 'attr', label: 'GENERATOR', sub: genAttr?.status === 'ATTRIBUTION_UNAVAILABLE' ? 'unavailable' : (genAttr?.label ?? 'unknown'),
      kind: 'attribution', x: W * 0.18, y: 96,
      payload: genAttr ?? { status: 'ATTRIBUTION_UNAVAILABLE' },
    });
    edges.push(['verdict', 'attr']);
    if (!dets.length) {
      // Stage 2 (S2-1): a configured-but-withheld model is not "no model".
      const suppressedReason = (ens as any).ml_suppressed_reason as string | null | undefined;
      nodes.push(suppressedReason
        ? { id: 'det-none', label: 'ML withheld', sub: 'output withheld', kind: 'detector', x: W * 0.5, y: 268, payload: { note: suppressedReason } }
        : { id: 'det-none', label: 'No model', sub: 'MODEL_NOT_CONFIGURED', kind: 'detector', x: W * 0.5, y: 268 });
      edges.push(['ml', 'det-none']);
    }
    nodes.push({
      id: 'prov-leaf', label: 'Provenance detail', kind: 'provenance', x: W * 0.82, y: 268,
      sub: `AI-lean ${ens.provenance?.ai_score ?? 50}/100`,
      payload: ens.provenance,
    });
    edges.push(['prov', 'prov-leaf']);
    return { nodes, edges };
  }, [report, ens, meta.label]);

  const byId = useMemo(() => Object.fromEntries(nodes.map((n) => [n.id, n])), [nodes]);
  const sel = selected ? byId[selected] : null;

  const nodeColor = (k: Node['kind']) =>
    k === 'verdict' ? meta.color : k === 'branch' ? '#38bdf8' : k === 'detector' ? '#a78bfa' : k === 'provenance' ? '#f59e0b' : '#64748b';

  return (
    <div style={{ border: '1px solid var(--border)', borderRadius: 12, background: 'var(--card)', padding: 16 }}>
      <div style={{ fontWeight: 700, marginBottom: 4 }}>Evidence graph</div>
      <div style={{ fontSize: 12, color: 'var(--muted)', marginBottom: 8 }}>
        How the verdict was assembled — click any node for its score, confidence, evidence and limitations.
      </div>
      <svg viewBox={`0 0 ${W} 330`} style={{ width: '100%', height: 'auto', cursor: 'pointer' }} role="img" aria-label="Evidence graph">
        {edges.map(([a, b]) => {
          const A = byId[a]; const B = byId[b];
          const mx = (A.x + B.x) / 2;
          return <path key={`${a}-${b}`} d={`M ${A.x} ${A.y + 22} C ${mx} ${A.y + 22}, ${mx} ${B.y - 22}, ${B.x} ${B.y - 22}`} fill="none" stroke="#475569" strokeWidth={1.5} />;
        })}
        {nodes.map((nd) => {
          const c = nodeColor(nd.kind);
          const isSel = selected === nd.id;
          const w = nd.kind === 'verdict' ? 190 : nd.kind === 'branch' ? 150 : 118;
          return (
            <g key={nd.id} onClick={() => setSelected(isSel ? null : nd.id)} style={{ cursor: 'pointer' }}>
              <rect x={nd.x - w / 2} y={nd.y - 22} width={w} height={44} rx={10}
                fill={isSel ? c : 'var(--card)'} fillOpacity={isSel ? 0.25 : 1}
                stroke={c} strokeWidth={isSel ? 2.5 : 1.5} />
              <text x={nd.x} y={nd.y - 2} textAnchor="middle" fontSize={11} fontWeight={700} fill="var(--text)">
                {nd.label.length > 24 ? nd.label.slice(0, 23) + '…' : nd.label}
              </text>
              <text x={nd.x} y={nd.y + 13} textAnchor="middle" fontSize={9} fill="var(--muted)">
                {nd.sub.length > 30 ? nd.sub.slice(0, 29) + '…' : nd.sub}
              </text>
            </g>
          );
        })}
      </svg>
      {sel && <NodeDetail node={sel} />}
    </div>
  );
}

function NodeDetail({ node }: { node: Node }) {
  const box: React.CSSProperties = {
    marginTop: 12, border: '1px solid var(--border)', borderRadius: 10,
    padding: 12, background: 'var(--bg)', fontSize: 13,
  };
  if (node.kind === 'verdict') {
    const m = VERDICT_META[node.label as keyof typeof VERDICT_META];
    return <div style={box}><b>Final verdict node.</b> Aggregates all branches below with hybrid weights.</div>;
  }
  if (node.kind === 'branch') {
    return <div style={box}><b>{node.label}</b> — {node.sub}. Branch weight from detector_config.json → ensemble_v2.</div>;
  }
  if (node.kind === 'provenance') {
    const p = node.payload;
    return (
      <div style={box}>
        <b>Provenance</b>
        <div>AI-leaning score: {p?.ai_score}/100 (confidence {p?.confidence}%)</div>
        <div style={{ color: 'var(--muted)' }}>{p?.note}</div>
        <div style={{ color: 'var(--muted)', marginTop: 4 }}>
          Limitation: provenance is claimed context, not pixel evidence. Missing provenance proves nothing.
        </div>
      </div>
    );
  }
  if (node.kind === 'detector') {
    const d = node.payload as MLDetector | undefined;
    if (!d) return <div style={box}><b>No ML model configured.</b> The verdict used forensic signals only.</div>;
    return (
      <div style={box}>
        <b>{d.detector_name}</b> <span style={{ color: 'var(--muted)' }}>v{d.detector_version} · {d.model_type} · {d.model_status}</span>
        <div style={{ marginTop: 6 }}>Raw score: <b>{d.raw_score}</b> → calibrated P(AI): <b>{d.probability_ai}%</b></div>
        <div>Calibration: {d.calibration} · self-reported confidence {d.confidence}% · inference {d.inference_time_ms} ms</div>
        {d.generator_label && <div>Generator label: {d.generator_label}</div>}
        <div style={{ marginTop: 6, color: 'var(--muted)' }}>Limitations: {d.limitations}</div>
      </div>
    );
  }
  if (node.kind === 'ood') {
    return (
      <div style={box}>
        <b>Out-of-distribution status</b>
        <div style={{ marginTop: 4 }}>Status: <b>{node.payload?.ood_status?.replace(/_/g, ' ')}</b></div>
        <div style={{ color: 'var(--muted)', marginTop: 4 }}>
          Measures whether this input resembles the ML model's training distribution.
          OOD does not mean AI — it means treat the ML probability with extra caution.
        </div>
      </div>
    );
  }
  if (node.kind === 'calibration') {
    return (
      <div style={box}>
        <b>Calibration</b>
        <div style={{ marginTop: 4 }}>Method: {node.payload?.method}</div>
        <div style={{ color: 'var(--muted)', marginTop: 4 }}>
          Raw model scores are never presented as probabilities. Calibration maps
          raw outputs to probabilities on held-out data.
        </div>
      </div>
    );
  }
  if (node.kind === 'attribution') {
    const g = node.payload;
    return (
      <div style={box}>
        <b>Generator attribution</b>
        <div style={{ marginTop: 4 }}>Status: <b>{g?.status}</b></div>
        <div style={{ color: 'var(--muted)', marginTop: 4 }}>
          {g?.detail ?? 'Attribution requires a dedicated evaluated model. AI detection never implies a specific generator.'}
        </div>
      </div>
    );
  }
  const s = node.payload as Signal;
  return (
    <div style={box}>
      <b>{s.display_name}</b>
      <span style={{ color: 'var(--muted)' }}> — status {s.status}, {s.duration_ms.toFixed(0)} ms</span>
      <div style={{ marginTop: 6 }}>AI score <b>{s.ai_score.toFixed(0)}/100</b> · manipulation <b>{s.manip_score.toFixed(0)}/100</b> · confidence <b>{s.confidence.toFixed(0)}%</b></div>
      <div style={{ marginTop: 6 }}>
        {s.evidence.slice(0, 5).map((e, i) => (
          <div key={i} style={{ marginTop: 2 }}>
            <span style={{ color: 'var(--muted)' }}>[{e.direction}]</span> {e.text}
          </div>
        ))}
      </div>
      {s.warnings.length > 0 && (
        <div style={{ marginTop: 6, color: '#f59e0b' }}>Warnings: {s.warnings.join('; ')}</div>
      )}
      <details style={{ marginTop: 6 }}>
        <summary style={{ cursor: 'pointer', color: 'var(--muted)' }}>Technical detail</summary>
        <pre style={{ fontSize: 11, overflowX: 'auto', marginTop: 4 }}>{JSON.stringify(s.technical, null, 1).slice(0, 1500)}</pre>
      </details>
    </div>
  );
}
