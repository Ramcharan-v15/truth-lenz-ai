import { useEffect, useState } from 'react';
import type { AnalysisReport } from '../types';
import { getModelStatus, type ModelRegistryStatus } from '../api';

/**
 * Model status strip (Phase 23):
 *   FORENSIC ENGINE  ● Active
 *   ML DETECTOR      ● Configured / ○ Not configured
 *   C2PA             ● Available / ○ Missing
 * Never hides the absence of a trained model.
 */
export default function ModelStatus({ report }: { report?: AnalysisReport | null }) {
  const [reg, setReg] = useState<ModelRegistryStatus | null>(null);

  useEffect(() => {
    getModelStatus().then(setReg).catch(() => setReg(null));
  }, []);

  const md = report?.signals.find((s) => s.name === 'metadata');
  const c2pa = md?.technical?.c2pa_status as string | undefined;
  const c2paOk = c2pa === 'unvalidated' || c2pa === 'verified';
  const mlOn = reg?.ml_available ?? (report?.ensemble.ml_available ?? false);
  const demoModel = (reg?.models ?? []).find((m: any) =>
    /demonstration/i.test(m.limitations ?? '') || m.config?.demonstration);

  const dot = (on: boolean, color = '#22c55e') => (
    <span
      style={{
        display: 'inline-block', width: 9, height: 9, borderRadius: '50%',
        background: on ? color : 'transparent',
        border: `2px solid ${on ? color : '#64748b'}`,
        marginRight: 8, flexShrink: 0,
      }}
    />
  );

  const item = (label: string, on: boolean, sub: string, color?: string) => (
    <div style={{ display: 'flex', alignItems: 'flex-start', gap: 4, minWidth: 0 }}>
      <div style={{ marginTop: 3 }}>{dot(on, color)}</div>
      <div style={{ minWidth: 0 }}>
        <div style={{ fontSize: 12, fontWeight: 700, letterSpacing: '0.06em' }}>{label}</div>
        <div style={{ fontSize: 11, color: 'var(--muted)', marginTop: 2 }}>{sub}</div>
      </div>
    </div>
  );

  return (
    <div
      style={{
        border: '1px solid var(--border)', borderRadius: 12, padding: '12px 16px',
        background: 'var(--card)', display: 'flex', gap: 28, flexWrap: 'wrap',
        alignItems: 'flex-start',
      }}
    >
      {item('FORENSIC ENGINE', true, 'Active — 10 local analyzers')}
      {item('ML DETECTOR', mlOn,
        mlOn
          ? `${reg?.ready ?? 0} model${(reg?.ready ?? 0) === 1 ? '' : 's'} ready`
          : 'Not configured — forensic analysis only',
        mlOn ? '#38bdf8' : undefined)}
      {item('C2PA', c2paOk,
        c2pa ? `Content Credentials: ${c2pa}` : 'No report loaded — status unknown',
        c2paOk ? '#a78bfa' : undefined)}
      {!mlOn && (
        <div style={{ fontSize: 11, color: 'var(--muted)', flexBasis: '100%', marginTop: 2 }}>
          Currently operating with forensic analysis only. A trained model can be registered
          in <code>models/registry.json</code> — it will join the ensemble as one signal, never the verdict.
        </div>
      )}
      {mlOn && demoModel && (
        <div style={{ fontSize: 11, color: '#f59e0b', flexBasis: '100%', marginTop: 2 }}>
          ⚠ Demonstration model active ({demoModel.name} v{demoModel.version}): trained on synthetic
          data, not validated on real imagery. Advisory signal only — reduced ensemble weight, no veto power.
        </div>
      )}
    </div>
  );
}
