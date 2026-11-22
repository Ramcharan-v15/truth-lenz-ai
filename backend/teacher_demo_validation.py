"""Teacher Demo upload validation — TRUTH LENZ AI.

Pure, non-inference helpers for the teacher demonstration page
(frontend/src/pages/TeacherDemo.tsx). These helpers validate file
bytes only. They never run a detector, never load a model for
inference, never open pilot images, and never write a file to disk.

Rules (mirroring backend/api/routes.py::_detect_mime, restricted to
the teacher demo's set):
- Allowed formats: JPG/JPEG, PNG, WebP only. No BMP, no HEIC.
- Magic bytes are authoritative. The file extension and the claimed
  MIME type are never trusted alone.
- Empty files are rejected. Oversized files are rejected.
- A magic-byte pass is necessary, not sufficient, for decodability:
  a truncated or corrupt image can still carry a valid magic prefix.
  Full decodability is checked separately by the browser image
  decoder in the frontend preview, and by the pipeline only if an
  authorized inference path is ever activated in the future.

Proposed API contract (NOT activated — documentation only):
    POST /api/teacher-demo/validate
      multipart field: file
      success: 200 {"detected_format": "image/png", "size_bytes": N}
      errors:  400 empty / malformed, 413 oversized, 415 unsupported
    This endpoint is deliberately NOT implemented in this task:
    the teacher demo previews images locally in the browser and does
    not send them to the server at all, so no server upload, temp
    file, persistence, or cleanup path is needed, and no image data
    leaves the user's machine. If a server-side validation endpoint
    is ever added, it must follow this contract, enforce the same
    limits, process in memory only, clean up any temp file on every
    path, never log image contents, and never call the detector.

Existing API contract actually used by the teacher demo:
    GET /api/models
      Returns registry metadata only (name, version, status,
      lifecycle_status, validation_status, limitations). No image
      is sent, no inference runs, no prediction is returned.
    GET /api/health (optional, for connectivity only)
    POST /api/analyze is an existing forensic pipeline endpoint.
    It is NOT an authorized real-world classification endpoint for
    this demo, and the teacher demo deliberately does not call it.
"""

from __future__ import annotations

TEACHER_MAX_MB = 25
TEACHER_MAX_BYTES = TEACHER_MAX_MB * 1024 * 1024

ALLOWED_MIME_TYPES = ("image/jpeg", "image/png", "image/webp")
ALLOWED_EXTENSIONS = ("jpg", "jpeg", "png", "webp")


def detect_teacher_format(data: bytes) -> str | None:
    """Return the detected MIME type from magic bytes, or None."""
    if not data:
        return None
    if data[:3] == b"\xff\xd8\xff":
        return "image/jpeg"
    if data[:8] == b"\x89PNG\r\n\x1a\n":
        return "image/png"
    if len(data) >= 12 and data[:4] == b"RIFF" and data[8:12] == b"WEBP":
        return "image/webp"
    return None


def validate_teacher_upload_bytes(
    data: bytes,
    filename: str = "upload",
    max_bytes: int = TEACHER_MAX_BYTES,
) -> str:
    """Validate upload bytes. Return detected MIME type.

    Raises ValueError with a user-safe message (no paths, no internal
    details) for empty, oversized, or unsupported data. The filename
    is used only for a secondary extension hint; magic bytes decide.
    """
    if not data:
        raise ValueError("Empty file received.")
    if len(data) > max_bytes:
        raise ValueError(f"File too large. Maximum is {TEACHER_MAX_MB} MB.")
    detected = detect_teacher_format(data)
    if detected is None:
        raise ValueError(
            "Unsupported or unrecognized image format. "
            "Supported: JPG, PNG, WEBP."
        )
    return detected
