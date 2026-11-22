"""Optional analysis history. Privacy-first: nothing is persisted unless the
caller explicitly opts in (save=true). Thumbnails only - never full images."""
from __future__ import annotations

import json
import os
import sqlite3
import time
from typing import Any, Dict, List, Optional

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "history.db")


def _conn() -> sqlite3.Connection:
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    c = sqlite3.connect(DB_PATH)
    c.execute(
        """CREATE TABLE IF NOT EXISTS analyses (
            id TEXT PRIMARY KEY,
            filename TEXT,
            created_at TEXT,
            verdict TEXT,
            ai_probability REAL,
            manipulation_probability REAL,
            authenticity_probability REAL,
            confidence REAL,
            image_quality TEXT,
            width INTEGER,
            height INTEGER,
            thumbnail TEXT,
            full_json TEXT
        )"""
    )
    return c


def save_analysis(report: Dict[str, Any]) -> str:
    # V5.1.8.62: serialize strictly BEFORE any database write, so a
    # report containing NaN/Infinity can never be persisted (not even
    # partially): json.dumps with allow_nan=False raises ValueError
    # before _conn() is touched and before any row is written.
    full_json = json.dumps(report, allow_nan=False)
    ens = report["ensemble"]
    quality = next((s for s in report["signals"] if s["name"] == "quality"), {})
    c = _conn()
    c.execute(
        "INSERT OR REPLACE INTO analyses VALUES (?,?,?,?,?,?,?,?,?,?,?,?,?)",
        (report["analysis_id"], report["filename"], report["analyzed_at"],
         ens["verdict"], ens["ai_probability"], ens["manipulation_probability"],
         ens["authenticity_probability"], ens["confidence"],
         (quality.get("technical") or {}).get("quality_tier", "Unknown"),
         report["width"], report["height"], report["preview"],
         full_json),
    )
    c.commit()
    c.close()
    return report["analysis_id"]


def list_analyses(limit: int = 50) -> List[Dict[str, Any]]:
    c = _conn()
    rows = c.execute(
        "SELECT id, filename, created_at, verdict, ai_probability, manipulation_probability,"
        " authenticity_probability, confidence, image_quality, width, height, thumbnail"
        " FROM analyses ORDER BY created_at DESC LIMIT ?", (limit,)
    ).fetchall()
    c.close()
    cols = ["id", "filename", "created_at", "verdict", "ai_probability",
            "manipulation_probability", "authenticity_probability", "confidence",
            "image_quality", "width", "height", "thumbnail"]
    return [dict(zip(cols, r)) for r in rows]


def get_analysis(analysis_id: str) -> Optional[Dict[str, Any]]:
    c = _conn()
    row = c.execute("SELECT full_json FROM analyses WHERE id=?", (analysis_id,)).fetchone()
    c.close()
    return json.loads(row[0]) if row else None


def delete_analysis(analysis_id: str) -> bool:
    c = _conn()
    cur = c.execute("DELETE FROM analyses WHERE id=?", (analysis_id,))
    c.commit()
    c.close()
    return cur.rowcount > 0


def delete_all() -> int:
    c = _conn()
    cur = c.execute("DELETE FROM analyses")
    c.commit()
    n = cur.rowcount
    c.close()
    return n


def stats() -> Dict[str, Any]:
    c = _conn()
    total = c.execute("SELECT COUNT(*) FROM analyses").fetchone()[0]
    by_verdict = dict(c.execute("SELECT verdict, COUNT(*) FROM analyses GROUP BY verdict").fetchall())
    # Stage 1 (L11): AVG over zero rows is NULL, not a measurement. Report
    # None (rendered 'unavailable') instead of fabricating a 0% average.
    # A genuine 0.0 average (e.g. only fallback rows, confidence 0.0) is kept.
    raw_avg = c.execute("SELECT AVG(confidence) FROM analyses").fetchone()[0]
    avg_conf = round(raw_avg, 1) if (total > 0 and raw_avg is not None) else None
    c.close()
    return {"total": total, "by_verdict": by_verdict, "avg_confidence": avg_conf}
