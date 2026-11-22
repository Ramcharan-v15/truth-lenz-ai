import { useRef, useState } from 'react';
import { fmtBytes } from '../types';

const ACCEPT = '.jpg,.jpeg,.png,.webp,.bmp,.heic,.heif';
const MAX_MB = 25;

export interface PickedFile {
  file: File;
  previewUrl: string;
  dims: string;
}

export function validateImage(f: File): string | null {
  const okTypes = ['image/jpeg', 'image/png', 'image/webp', 'image/bmp', 'image/heic', 'image/heif'];
  const ext = f.name.split('.').pop()?.toLowerCase();
  const okExt = ['jpg', 'jpeg', 'png', 'webp', 'bmp', 'heic', 'heif'];
  if (!okTypes.includes(f.type) && !(ext && okExt.includes(ext)))
    return `Unsupported format "${ext || '?'}". Please use JPG, PNG, WEBP or BMP.`;
  if (f.size > MAX_MB * 1024 * 1024)
    return `File is ${fmtBytes(f.size)} — the limit is ${MAX_MB} MB.`;
  if (f.size === 0) return 'This file is empty.';
  return null;
}

export function loadPreview(f: File): Promise<PickedFile> {
  return new Promise((resolve, reject) => {
    const url = URL.createObjectURL(f);
    const img = new Image();
    img.onload = () => resolve({ file: f, previewUrl: url, dims: `${img.naturalWidth}×${img.naturalHeight}` });
    img.onerror = () => { URL.revokeObjectURL(url); reject(new Error('Could not read this image. It may be corrupted.')); };
    img.src = url;
  });
}

export default function UploadZone({
  onPick, multiple = false, compact = false,
}: {
  onPick: (files: PickedFile[]) => void;
  multiple?: boolean;
  compact?: boolean;
}) {
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);

  const handle = async (list: FileList | null) => {
    setError(null);
    if (!list || !list.length) return;
    const picked: PickedFile[] = [];
    for (const f of Array.from(list)) {
      const err = validateImage(f);
      if (err) { setError(err); continue; }
      try { picked.push(await loadPreview(f)); }
      catch (e: any) { setError(e.message); }
      if (!multiple) break;
    }
    if (picked.length) onPick(picked);
  };

  return (
    <div>
      <div
        className={`dropzone${dragging ? ' dragging' : ''}`}
        style={compact ? { padding: '22px 16px' } : undefined}
        role="button" tabIndex={0}
        aria-label="Upload image. Drag and drop or press Enter to browse."
        onClick={() => inputRef.current?.click()}
        onKeyDown={(e) => { if (e.key === 'Enter' || e.key === ' ') { e.preventDefault(); inputRef.current?.click(); } }}
        onDragOver={(e) => { e.preventDefault(); setDragging(true); }}
        onDragLeave={() => setDragging(false)}
        onDrop={(e) => { e.preventDefault(); setDragging(false); handle(e.dataTransfer.files); }}
      >
        <div style={{ fontSize: compact ? 26 : 38, marginBottom: 8 }} aria-hidden="true">◈</div>
        <div style={{ fontWeight: 700, fontSize: compact ? 14 : 17 }}>
          {dragging ? 'Drop the image here' : 'Drag & drop an image here'}
        </div>
        <div className="dim" style={{ fontSize: 13, marginTop: 6 }}>
          or <span style={{ color: 'var(--accent)', fontWeight: 600 }}>browse files</span> — JPG, PNG, WEBP, BMP{multiple ? ' (multiple allowed)' : ''}
        </div>
        <div className="faint" style={{ fontSize: 12, marginTop: 8 }}>
          Max {MAX_MB} MB · processed in memory · never stored unless you choose to save
        </div>
        <input
          ref={inputRef} type="file" accept={ACCEPT} multiple={multiple}
          style={{ display: 'none' }} aria-hidden="true"
          onChange={(e) => { handle(e.target.files); e.target.value = ''; }}
        />
      </div>
      {error && <div className="alert error" role="alert">{error}</div>}
    </div>
  );
}
