import { Link } from 'react-router-dom';

const SIGNALS = [
  { name: 'Metadata Forensics', desc: 'EXIF, generator signatures, editing-software tags, C2PA scan.' },
  { name: 'Noise Analysis', desc: 'Sensor-noise estimation in flat regions; consistency mapping.' },
  { name: 'Frequency Analysis', desc: 'FFT spectrum, spectral slope, generative upsampling fingerprints.' },
  { name: 'Pixel Forensics', desc: 'Error Level Analysis, micro-texture autocorrelation.' },
  { name: 'AI Artifacts', desc: 'Heuristic synthetic-texture, symmetry and repetition indicators.' },
  { name: 'Manipulation Detection', desc: 'Copy-move block matching, splicing noise anomalies, heatmaps.' },
  { name: 'Compression', desc: 'Quantization tables, block structure, recompression traces.' },
  { name: 'Consistency', desc: 'Internal metadata contradictions and provenance checks.' },
];

const STEPS = ['Upload', 'Inspect', 'Extract signals', 'Analyze', 'Cross-check', 'Ensemble scoring', 'Explain', 'Report'];

export default function Home() {
  return (
    <div className="shell">
      <section className="hero">
        <div className="scan-line" aria-hidden="true" />
        <div className="eyebrow">AI IMAGE AUTHENTICITY & DIGITAL FORENSICS</div>
        <h1>Can you trust<br />this <span className="accent">image?</span></h1>
        <p className="sub">
          Analyze images for AI generation, manipulation and authenticity using
          multiple independent forensic signals — not a single black-box detector.
        </p>
        <div className="hero-ctas">
          <Link to="/analyze" className="btn btn-primary">Analyze an Image →</Link>
          <Link to="/how-it-works" className="btn">See How It Works</Link>
        </div>
        <p className="faint" style={{ fontSize: 12.5, marginTop: 18 }}>
          Probabilistic evidence, never 100% certainty · Privacy-first: images are processed in memory
        </p>
      </section>

      <section>
        <div className="eyebrow">MULTI-SIGNAL ARCHITECTURE</div>
        <h2 className="section-title" style={{ marginTop: 0 }}>One detector can be fooled. Ten signals cross-check each other.</h2>
        <p className="section-sub">
          Each analyzer returns structured evidence with its own confidence. A weighted ensemble
          combines them — and when signals genuinely disagree, the verdict is <em>inconclusive</em>{' '}
          rather than forced.
        </p>
        <div className="grid grid-4">
          {SIGNALS.map((s) => (
            <div key={s.name} className="card hoverable">
              <h3 style={{ fontSize: 14 }}>{s.name}</h3>
              <p className="dim" style={{ fontSize: 13, margin: 0 }}>{s.desc}</p>
            </div>
          ))}
        </div>
      </section>

      <section>
        <h2 className="section-title">From upload to explained verdict</h2>
        <div className="panel panel-pad" style={{ marginTop: 12 }}>
          <div style={{ display: 'flex', flexWrap: 'wrap', gap: 8, alignItems: 'center' }}>
            {STEPS.map((s, i) => (
              <span key={s} style={{ display: 'flex', alignItems: 'center', gap: 8 }}>
                <span className="tag" style={{ color: 'var(--accent)', borderColor: 'var(--accent)' }}>{i + 1}. {s}</span>
                {i < STEPS.length - 1 && <span className="faint">→</span>}
              </span>
            ))}
          </div>
        </div>
      </section>

      <section>
        <h2 className="section-title">Built for people who need evidence</h2>
        <div className="grid grid-3" style={{ marginTop: 12 }}>
          {[
            { t: 'Researchers & students', d: 'Every technique ships with an educational explainer — what it is, why it matters, and its limits.' },
            { t: 'Journalists & investigators', d: 'Exportable forensic reports (PDF/JSON/CSV) with evidence trails, timestamps and analysis IDs.' },
            { t: 'Analysts & the curious', d: 'Compare two images, batch-analyze sets, inspect pixels, ELA maps and frequency spectra.' },
          ].map((c) => (
            <div key={c.t} className="card"><h3>{c.t}</h3><p className="dim" style={{ fontSize: 13.5, margin: 0 }}>{c.d}</p></div>
          ))}
        </div>
      </section>

      <section>
        <div className="disclaimer-strip" style={{ marginTop: 34 }}>
          <strong>Responsible AI:</strong> AI image detection is probabilistic. No detector can guarantee
          whether an image is AI-generated or authentic. Compression, editing, screenshots, metadata
          removal and new generation techniques all affect results. Do not use Truth Lenz AI as the
          sole basis for legal, academic, employment, financial or other high-stakes decisions.
        </div>
        <div style={{ textAlign: 'center', marginTop: 26 }}>
          <Link to="/analyze" className="btn btn-primary">Start Analyzing →</Link>
        </div>
      </section>
    </div>
  );
}
