import type { AnalysisReport } from '../types';

const STAGE_LABELS: Record<string, string> = {
  metadata: 'Metadata Analysis',
  statistics: 'Image Inspection',
  compression: 'Compression Analysis',
  noise: 'Pixel Forensics',
  frequency: 'Frequency Analysis',
  pixel: 'Pixel Forensics',
  ai_artifacts: 'AI Artifact Analysis',
  manipulation: 'Manipulation Check',
  quality: 'Quality Assessment',
  consistency: 'Consistency Check',
};

/**
 * Animated pipeline. During analysis, stages light up in forensic order;
 * once the real result arrives, each stage shows its ACTUAL status and
 * measured duration from the backend - no fake spinners.
 */
export default function Pipeline({
  report, activeIndex, done,
}: {
  report?: AnalysisReport;
  activeIndex?: number;
  done?: boolean;
}) {
  const stages = report?.stages ?? Object.keys(STAGE_LABELS).map((s) => ({ stage: s, display: STAGE_LABELS[s], status: 'pending', duration_ms: 0 }));

  return (
    <div className="pipeline" role="list" aria-label="Analysis pipeline">
      {stages.map((s, i) => {
        let cls = '';
        let icon = '·';
        if (done || report) {
          if (s.status === 'ok') { cls = 'done'; icon = '✓'; }
          else if (s.status === 'warning') { cls = 'warn'; icon = '!'; }
          else if (s.status === 'unable') { cls = 'failed'; icon = '×'; }
          else { cls = 'done'; icon = '✓'; }
        } else if (activeIndex !== undefined) {
          if (i < activeIndex) { cls = 'done'; icon = '✓'; }
          else if (i === activeIndex) { cls = 'running'; icon = '●'; }
        }
        return (
          <div key={s.stage} className={`stage ${cls}`} role="listitem">
            <span className="dot" aria-hidden="true">{icon}</span>
            <span>{i + 1}. {s.display}</span>
            <span className="meta">
              {report ? `${s.duration_ms.toFixed(0)} ms · ${s.status}` : cls === 'running' ? 'analyzing…' : cls === 'done' ? 'completed' : ''}
            </span>
          </div>
        );
      })}
    </div>
  );
}
