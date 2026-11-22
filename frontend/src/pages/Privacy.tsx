import { useState } from 'react';
import { deleteAllHistory } from '../api';

const ROWS: [string, string][] = [
  ['Are images uploaded to a server?', 'Yes — to this application’s own analysis endpoint, so the forensic engine can process them.'],
  ['Are uploaded images stored?', 'No. Images are decoded and analyzed in memory and never written to disk — unless you explicitly enable “Save to history”.'],
  ['When are temporary files deleted?', 'There are no temporary files: processing is fully in-memory. Request memory is released when analysis completes.'],
  ['Is analysis data retained?', 'Only if you opt into history. Then a thumbnail, filename, scores and verdict are kept in a local database — never the full image.'],
  ['Do third-party AI APIs receive the image?', 'No. All analysis runs locally in this application. No external detection API is called; no trained-model service receives your image.'],
  ['What data is collected?', 'Nothing beyond what you choose to save. No accounts, no tracking, no analytics.'],
];

export default function Privacy() {
  const [deleted, setDeleted] = useState(false);

  const wipe = async () => {
    if (window.confirm('Delete all saved analyses and thumbnails?')) {
      await deleteAllHistory();
      setDeleted(true);
    }
  };

  return (
    <div className="shell" style={{ paddingTop: 30, paddingBottom: 20 }}>
      <div className="eyebrow">PRIVACY CENTER</div>
      <h1 style={{ margin: '0 0 8px', fontSize: 30 }}>Privacy</h1>
      <p className="section-sub">
        Truth Lenz AI is privacy-first by architecture: in-memory processing, opt-in history,
        no third-party image sharing, no accounts, no tracking.
      </p>

      <div className="panel panel-pad">
        {ROWS.map(([q, a]) => (
          <div key={q} style={{ padding: '12px 0', borderBottom: '1px solid var(--border)' }}>
            <div style={{ fontWeight: 700, fontSize: 14 }}>{q}</div>
            <div className="dim" style={{ fontSize: 13.5, marginTop: 4 }}>{a}</div>
          </div>
        ))}
      </div>

      <h2 className="section-title">Your data, your control</h2>
      <div className="panel panel-pad">
        <p className="dim" style={{ fontSize: 13.5, marginTop: 0 }}>
          Saved analyses live in this application’s local database. You can remove a single
          analysis from the History page, or wipe everything here.
        </p>
        <button className="btn btn-danger" onClick={wipe}>Delete all history</button>
        {deleted && <div className="alert" style={{ marginTop: 12 }}>All saved analyses have been deleted.</div>}
      </div>

      <div className="disclaimer-strip" style={{ marginTop: 20 }}>
        Even with these protections, avoid uploading images containing sensitive personal
        information unless you are running Truth Lenz AI on infrastructure you control.
      </div>
    </div>
  );
}
