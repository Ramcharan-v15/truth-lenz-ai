import type { Signal } from '../types';

const FIELDS: { key: string; label: string }[] = [
  { key: 'filename', label: 'Filename' },
  { key: 'mime', label: 'MIME type' },
  { key: 'format', label: 'Container format' },
  { key: 'size', label: 'Dimensions' },
  { key: 'mode', label: 'Color model' },
  { key: 'exif_tag_count', label: 'EXIF tags' },
  { key: 'Make', label: 'Camera manufacturer' },
  { key: 'Model', label: 'Camera model' },
  { key: 'LensModel', label: 'Lens' },
  { key: 'ISOSpeedRatings', label: 'ISO' },
  { key: 'ExposureTime', label: 'Shutter speed' },
  { key: 'FNumber', label: 'Aperture' },
  { key: 'FocalLength', label: 'Focal length' },
  { key: 'DateTimeOriginal', label: 'Capture timestamp' },
  { key: 'DateTime', label: 'Modification timestamp' },
  { key: 'Software', label: 'Software' },
  { key: 'ImageDescription', label: 'Description' },
  { key: 'Artist', label: 'Artist' },
  { key: 'Copyright', label: 'Copyright' },
  { key: 'gps_present', label: 'GPS data' },
  { key: 'icc_profile', label: 'ICC profile' },
  { key: 'xmp', label: 'XMP data' },
  { key: 'c2pa_detected', label: 'Content Credentials (C2PA)' },
  { key: 'generation_parameters_present', label: 'AI generation parameters' },
  { key: 'timestamp_mismatch', label: 'Timestamp mismatch' },
  { key: 'bits_per_pixel', label: 'Bits per pixel' },
  { key: 'estimated_jpeg_quality', label: 'Est. JPEG quality' },
  { key: 'quantization_tables', label: 'Quantization tables' },
  { key: 'info_keys', label: 'Embedded info keys' },
];

export default function MetadataTable({ signal, filename, fileSize, mimeType }: {
  signal?: Signal; filename: string; fileSize: number; mimeType: string;
}) {
  const tech = signal?.technical || {};
  const exif = tech.exif_summary || {};

  const get = (key: string): { value: string; tag: 'present' | 'absent' | 'suspicious' | 'unknown' } => {
    if (key === 'filename') return { value: filename, tag: 'present' };
    if (key === 'mime') return { value: mimeType, tag: 'present' };
    if (key === 'size') return { value: tech.size ? `${tech.size[0]} × ${tech.size[1]}` : '—', tag: 'present' };
    if (key === 'format' || key === 'mode') {
      const v = tech[key];
      return { value: v ? String(v) : 'Unknown', tag: v ? 'present' : 'unknown' };
    }
    if (key === 'exif_tag_count') {
      const n = tech.exif_tag_count ?? 0;
      return { value: n > 0 ? `${n} tags` : 'None found', tag: n > 0 ? 'present' : 'absent' };
    }
    if (key in exif) {
      const v = String(exif[key]);
      const low = v.toLowerCase();
      const suspicious = /photoshop|gimp|stable diffusion|midjourney|dall|firefly|ai generat/i.test(low);
      return { value: v, tag: suspicious ? 'suspicious' : 'present' };
    }
    if (['gps_present', 'icc_profile', 'xmp', 'c2pa_detected', 'generation_parameters_present', 'timestamp_mismatch'].includes(key)) {
      const v = tech[key];
      if (v === true) {
        const susp = key === 'generation_parameters_present' || key === 'timestamp_mismatch';
        return { value: 'Yes', tag: susp ? 'suspicious' : 'present' };
      }
      return { value: 'No', tag: 'absent' };
    }
    if (key === 'info_keys') {
      const v = tech.info_keys;
      return { value: Array.isArray(v) && v.length ? v.join(', ') : 'None', tag: Array.isArray(v) && v.length ? 'present' : 'absent' };
    }
    const v = tech[key];
    if (v !== undefined && v !== null && v !== '') return { value: String(v), tag: 'present' };
    return { value: '—', tag: 'absent' };
  };

  return (
    <div>
      <dl className="kv" style={{ gridTemplateColumns: '220px 1fr' }}>
        {FIELDS.map(({ key, label }) => {
          const { value, tag } = get(key);
          return (
            <div key={key} style={{ display: 'contents' }}>
              <dt>{label}</dt>
              <dd style={{ display: 'flex', gap: 10, alignItems: 'center', flexWrap: 'wrap' }}>
                <span style={{ fontFamily: 'var(--font)' }}>{value}</span>
                <span className={`tag ${tag}`}>{tag}</span>
              </dd>
            </div>
          );
        })}
      </dl>
      <div className="disclaimer-strip" style={{ marginTop: 16 }}>
        Metadata can be removed or altered and should not be considered definitive evidence.
        Absence of metadata is <strong>not</strong> evidence of AI generation.
      </div>
    </div>
  );
}
