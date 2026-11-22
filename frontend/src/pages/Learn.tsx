const TOPICS = [
  {
    t: 'Metadata forensics', what: 'Metadata is data embedded in the file: camera model, timestamps, software tags, GPS.',
    why: 'It gives provenance context — which camera or software touched the file, and when.',
    show: 'A “Canon EOS R5” tag with exposure settings supports a camera-capture story; a “Stable Diffusion” tag is direct AI evidence.',
    limits: 'It can be removed, stripped by apps, or completely fabricated. Absence proves nothing.',
  },
  {
    t: 'Noise analysis', what: 'Every camera sensor adds subtle, consistent noise. Truth Lenz estimates it only in flat regions, where texture cannot bias the measurement.',
    why: 'Synthetic images are often implausibly clean, or show noise that varies unnaturally between regions.',
    show: 'Even noise across flat areas supports authentic capture; hot or dead patches flag further inspection.',
    limits: 'Denoising, heavy compression and high ISO change noise dramatically.',
  },
  {
    t: 'Frequency analysis', what: 'The 2D Fourier transform reveals repeating patterns invisible to the eye.',
    why: 'Some generators leave periodic fingerprints from upsampling; natural images decay smoothly (~1/f).',
    show: 'Sharp isolated spectral peaks, or an abnormally steep spectral slope, are worth a closer look.',
    limits: 'Newer generators are cleaner; regular real-world textures (fabrics, screens) can mimic peaks.',
  },
  {
    t: 'Error Level Analysis (ELA)', what: 'The image is re-saved at a fixed JPEG quality and the difference is amplified.',
    why: 'Regions with different compression histories — e.g. spliced content — respond differently.',
    show: 'Bright ELA regions are potentially suspicious regions for investigation.',
    limits: 'High-contrast edges light up innocently. ELA never proves manipulation.',
  },
  {
    t: 'Compression analysis', what: 'JPEG quantization tables describe how aggressively each 8×8 block was compressed.',
    why: 'Non-standard tables hint the image was re-saved at a different quality, as happens after editing.',
    show: 'Standard single-compression tables vs unusual ones; block-boundary discontinuity measurement.',
    limits: 'Many legitimate pipelines re-encode images (messaging apps, CDNs, social platforms).',
  },
  {
    t: 'Copy-move detection', what: 'The image is tiled into blocks, hashed via DCT, and distant near-duplicates are found and clustered.',
    why: 'Cloning — duplicating part of an image to hide or replicate objects — leaves clustered duplicates.',
    show: 'A heatmap overlay marks duplicated regions on the original image.',
    limits: 'Repetitive natural patterns (windows, fences) can trigger it; small or warped clones may be missed.',
  },
  {
    t: 'Ensemble scoring', what: 'Independent signal scores are combined with opinion-weighting: decisive signals count more, neutral ones less.',
    why: 'No single method is reliable. Weighted evidence with disagreement detection avoids forced verdicts.',
    show: 'The dashboard shows every signal’s contribution, the weights used, and why the verdict was reached.',
    limits: 'Weights are engineering judgment, documented in detector_config.json — not ground truth.',
  },
  {
    t: 'Confidence modelling', what: 'Confidence (trust in the numbers) is modelled separately from probability (the numbers).',
    why: 'Poor quality, heavy compression, screenshots and missing metadata all degrade what detectors can see.',
    show: 'Each confidence penalty is listed explicitly on the results dashboard.',
    limits: 'Confidence itself is an estimate — treat it as guidance, not a guarantee.',
  },
];

export default function Learn() {
  return (
    <div className="shell" style={{ paddingTop: 30, paddingBottom: 20 }}>
      <div className="eyebrow">LEARN FROM FORENSICS</div>
      <h1 style={{ margin: '0 0 8px', fontSize: 30 }}>Learn</h1>
      <p className="section-sub">
        Every technique Truth Lenz AI uses, explained: what it is, why it matters, what it shows,
        and where it breaks down. The same explainers appear inside each analysis (“Learn from this analysis”).
      </p>
      <div className="grid grid-2">
        {TOPICS.map((t) => (
          <div key={t.t} className="card learn-card">
            <h3>{t.t}</h3>
            <p style={{ fontSize: 13.5 }}><strong className="dim">What it is — </strong>{t.what}</p>
            <p style={{ fontSize: 13.5 }}><strong className="dim">Why it matters — </strong>{t.why}</p>
            <p style={{ fontSize: 13.5 }}><strong className="dim">What this image shows — </strong>{t.show}</p>
            <p style={{ fontSize: 13.5, marginBottom: 0 }}><strong style={{ color: 'var(--warn)' }}>Limitations — </strong>{t.limits}</p>
          </div>
        ))}
      </div>
      <div className="disclaimer-strip" style={{ marginTop: 24 }}>
        Forensic literacy in one sentence: <strong>look for converging evidence across independent
        methods, and distrust any single number</strong> — including ours.
      </div>
    </div>
  );
}
