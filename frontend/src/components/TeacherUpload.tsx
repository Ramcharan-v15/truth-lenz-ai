import { useEffect, useRef, useState } from 'react';
import { fmtBytes } from '../types';
import {
  detectTeacherFormat,
  TEACHER_MAX_MB,
  validateTeacherFileMeta,
} from '../teacherDemo';

export interface TeacherPickedFile {
  file: File;
  previewUrl: string;
  dims: string;
  width: number;
  height: number;
  detectedFormat: string;
}

async function readFirstBytes(file: File, n = 12): Promise<Uint8Array> {
  const slice = file.slice(0, n);
  const buf = await slice.arrayBuffer();
  return new Uint8Array(buf);
}

function loadImageInfo(previewUrl: string): Promise<{ dims: string; width: number; height: number }> {
  return new Promise((resolve, reject) => {
    const img = new Image();
    img.onload = () => resolve({ dims: `${img.naturalWidth}×${img.naturalHeight}`, width: img.naturalWidth, height: img.naturalHeight });
    img.onerror = () => reject(new Error('Could not read this image. It may be corrupted. Please re-export it as JPG, PNG, or WebP and try again.'));
    img.src = previewUrl;
  });
}

/**
 * Teacher Demo upload area.
 *
 * Strictly JPG / JPEG / PNG / WebP only (unlike the general UploadZone,
 * which also accepts BMP/HEIC). Checks, in order: empty file, file size
 * (25 MB), extension/MIME metadata, magic bytes (authoritative — the
 * extension alone is never trusted), then image decodability for
 * dimensions/preview. The image is previewed locally in the browser
 * only: it is never sent to the server, never stored, and never sent
 * to a third-party service by this component.
 */
export default function TeacherUpload({
  onPick,
}: {
  onPick: (picked: TeacherPickedFile | null) => void;
}) {
  const [dragging, setDragging] = useState(false);
  const [error, setError] = useState<string | null>(null);
  const [picked, setPicked] = useState<TeacherPickedFile | null>(null);
  const inputRef = useRef<HTMLInputElement>(null);
  const pickedRef = useRef<TeacherPickedFile | null>(null);

  useEffect(() => {
    return () => {
      if (pickedRef.current) URL.revokeObjectURL(pickedRef.current.previewUrl);
    };
  }, []);

  const clearPicked = () => {
    if (pickedRef.current) URL.revokeObjectURL(pickedRef.current.previewUrl);
    pickedRef.current = null;
    setPicked(null);
    onPick(null);
  };

  const handle = async (list: FileList | null) => {
    setError(null);
    if (!list || list.length === 0) return;
    const file = list[0];
    const metaError = validateTeacherFileMeta(file);
    if (metaError) {
      setError(metaError);
      return;
    }
    let detected: ReturnType<typeof detectTeacherFormat>;
    try {
      detected = detectTeacherFormat(await readFirstBytes(file));
    } catch {
      setError('Could not read this file. Please try again with a JPG, PNG, or WebP image.');
      return;
    }
    if (!detected) {
      setError('Unsupported or unrecognized image format. The file contents do not match JPG, PNG, or WebP, even if the file name suggests otherwise. Please use a real JPG, PNG, or WebP file.');
      return;
    }
    const previewUrl = URL.createObjectURL(file);
    let info: { dims: string; width: number; height: number };
    try {
      info = await loadImageInfo(previewUrl);
    } catch (e: any) {
      URL.revokeObjectURL(previewUrl);
      setError(e.message || 'Could not read this image. It may be corrupted.');
      return;
    }
    if (pickedRef.current) URL.revokeObjectURL(pickedRef.current.previewUrl);
    const next: TeacherPickedFile = { file, previewUrl, dims: info.dims, width: info.width, height: info.height, detectedFormat: detected };
    pickedRef.current = next;
    setPicked(next);
    onPick(next);
  };

  return (
    <div>
      {!picked && (
        <div
          className={`dropzone${dragging ? ' dragging' : ''}`}
          role="button"
          tabIndex={0}
          aria-label="Upload image. Drag and drop or press Enter to browse. JPG, PNG, or WebP, up to 25 MB."
          onClick={() => inputRef.current?.click()}
          onKeyDown={(e) => {
            if (e.key === 'Enter' || e.key === ' ') {
              e.preventDefault();
              inputRef.current?.click();
            }
          }}
          onDragOver={(e) => {
            e.preventDefault();
            setDragging(true);
          }}
          onDragLeave={() => setDragging(false)}
          onDrop={(e) => {
            e.preventDefault();
            setDragging(false);
            handle(e.dataTransfer.files);
          }}
        >
          <div style={{ fontSize: 38, marginBottom: 8 }} aria-hidden="true">◈</div>
          <div style={{ fontWeight: 700, fontSize: 17 }}>
            {dragging ? 'Drop the image here' : 'Drag & drop an image here'}
          </div>
          <div className="dim" style={{ fontSize: 13, marginTop: 6 }}>
            or <span style={{ color: 'var(--accent)', fontWeight: 600 }}>browse files</span> — JPG, JPEG, PNG, WebP
          </div>
          <div className="faint" style={{ fontSize: 12, marginTop: 8 }}>
            Max {TEACHER_MAX_MB} MB · previewed in your browser only · not uploaded, not stored, not sent to any third party
          </div>
          <input
            ref={inputRef}
            type="file"
            accept=".jpg,.jpeg,.png,.webp"
            style={{ display: 'none' }}
            aria-hidden="true"
            onChange={(e) => {
              handle(e.target.files);
              e.target.value = '';
            }}
          />
        </div>
      )}

      {picked && (
        <div data-testid="teacher-preview">
          <div style={{ border: '1px solid var(--border)', borderRadius: 12, background: '#000', padding: 12, textAlign: 'center' }}>
            <img
              src={picked.previewUrl}
              alt={`Preview of ${picked.file.name}`}
              style={{ display: 'block', maxWidth: '100%', maxHeight: 320, margin: '0 auto', objectFit: 'contain', borderRadius: 6 }}
            />
          </div>
          <div className="file-row" style={{ marginTop: 12 }}>
            <div style={{ flex: 1, minWidth: 0 }}>
              <div style={{ fontWeight: 600, fontSize: 14, overflowWrap: 'break-word' }}>{picked.file.name}</div>
              <div className="faint mono" style={{ fontSize: 12 }}>
                {fmtBytes(picked.file.size)} · {picked.dims} · Detected: {picked.detectedFormat}
              </div>
              <div className="faint" style={{ fontSize: 12, marginTop: 4 }}>
                Preview only — this image has not been sent to the server.
              </div>
            </div>
            <button className="btn btn-ghost btn-sm" onClick={clearPicked} aria-label="Remove selected image">
              ✕ Remove
            </button>
          </div>
        </div>
      )}

      {error && (
        <div className="alert error" role="alert" data-testid="teacher-error">
          {error}
        </div>
      )}
    </div>
  );
}
