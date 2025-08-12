"""Simple SQLite-backed cache for analytical API results.

Keys: dataset + process + params_hash + dataset_fingerprint (+ optional trace_id)
Stores JSON payloads for reuse when inputs haven't changed.
"""

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, Optional
from datetime import datetime
import hashlib
import uuid


DB_PATH = Path("processed_data") / "cache.db"


def _get_conn() -> sqlite3.Connection:
    DB_PATH.parent.mkdir(parents=True, exist_ok=True)
    conn = sqlite3.connect(str(DB_PATH))
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS results_cache (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            trace_id TEXT,
            dataset TEXT NOT NULL,
            process TEXT NOT NULL,
            params_hash TEXT NOT NULL,
            dataset_fingerprint TEXT NOT NULL,
            created_at TEXT NOT NULL,
            payload TEXT NOT NULL
        );
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_results_cache_keys
        ON results_cache (dataset, process, params_hash, dataset_fingerprint);
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_results_cache_trace
        ON results_cache (trace_id);
        """
    )
    # Reports table for storing JSON reports
    conn.execute(
        """
        CREATE TABLE IF NOT EXISTS reports (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            dataset TEXT NOT NULL,
            report_type TEXT NOT NULL,
            created_at TEXT NOT NULL,
            payload TEXT NOT NULL
        );
        """
    )
    conn.execute(
        """
        CREATE INDEX IF NOT EXISTS idx_reports_dataset_type
        ON reports (dataset, report_type, created_at DESC);
        """
    )
    return conn


def compute_params_hash(params: Dict[str, Any]) -> str:
    try:
        normalized = json.dumps(params, sort_keys=True, separators=(",", ":"))
    except Exception:
        normalized = str(params)
    return hashlib.sha256(normalized.encode("utf-8")).hexdigest()


def ensure_trace_id(trace_id: Optional[str]) -> str:
    return trace_id or uuid.uuid4().hex


def put_result(
    *,
    trace_id: Optional[str],
    dataset: str,
    process: str,
    params_hash: str,
    dataset_fingerprint: str,
    payload: Dict[str, Any],
) -> str:
    trace = ensure_trace_id(trace_id)
    row = (
        trace,
        dataset,
        process,
        params_hash,
        dataset_fingerprint,
        datetime.utcnow().isoformat(),
        json.dumps(payload, default=str),
    )
    with _get_conn() as conn:
        conn.execute(
            """
            INSERT INTO results_cache
            (trace_id, dataset, process, params_hash, dataset_fingerprint, created_at, payload)
            VALUES (?, ?, ?, ?, ?, ?, ?)
            """,
            row,
        )
    return trace


def get_result(
    *,
    dataset: str,
    process: str,
    params_hash: str,
    dataset_fingerprint: str,
    trace_id: Optional[str] = None,
) -> Optional[Dict[str, Any]]:
    query = (
        "SELECT payload, trace_id FROM results_cache WHERE dataset=? AND process=? AND params_hash=? AND dataset_fingerprint=?"
    )
    params = [dataset, process, params_hash, dataset_fingerprint]
    if trace_id:
        query += " AND trace_id=?"
        params.append(trace_id)
    query += " ORDER BY id DESC LIMIT 1"
    with _get_conn() as conn:
        cur = conn.execute(query, params)
        row = cur.fetchone()
        if not row:
            return None
        payload_str, trace = row
        try:
            payload = json.loads(payload_str)
        except Exception:
            return None
        payload["trace_id"] = trace
        payload["cached"] = True
        return payload


# ---------- JSON report helpers ----------
def add_report(*, dataset: str, report_type: str, payload: Dict[str, Any]) -> int:
    row = (
        dataset,
        report_type,
        datetime.utcnow().isoformat(),
        json.dumps(payload, default=str),
    )
    with _get_conn() as conn:
        cur = conn.execute(
            """
            INSERT INTO reports (dataset, report_type, created_at, payload)
            VALUES (?, ?, ?, ?)
            """,
            row,
        )
        return int(cur.lastrowid)


def list_reports_json(*, limit: int = 50) -> list[dict[str, Any]]:
    with _get_conn() as conn:
        cur = conn.execute(
            "SELECT id, dataset, report_type, created_at FROM reports ORDER BY id DESC LIMIT ?",
            (int(limit),),
        )
        items = []
        for rid, dataset, rtype, created in cur.fetchall():
            items.append({
                "id": int(rid),
                "dataset": dataset,
                "report_type": rtype,
                "created_at": created,
            })
        return items


def get_report_json(report_id: int) -> Optional[Dict[str, Any]]:
    with _get_conn() as conn:
        cur = conn.execute("SELECT id, dataset, report_type, created_at, payload FROM reports WHERE id=?", (int(report_id),))
        row = cur.fetchone()
        if not row:
            return None
        rid, dataset, rtype, created, payload_str = row
        try:
            payload = json.loads(payload_str)
        except Exception:
            payload = {"raw": payload_str}
        return {
            "id": int(rid),
            "dataset": dataset,
            "report_type": rtype,
            "created_at": created,
            "payload": payload,
        }


