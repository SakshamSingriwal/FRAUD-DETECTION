"""
runs.py — Run/experiment manager.

A *run* is one whole pipeline (data → preprocessing → training → prediction).
Each run is a folder under ``runs/`` holding a manifest (lightweight, for the
runs list) and a joblib snapshot of the pipeline state, so a run can be resumed
later from wherever the user left off, and cleaned up when no longer needed.
"""
from __future__ import annotations

import os
import json
import time
import uuid
import shutil

import joblib

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RUNS_DIR = os.path.join(ROOT, "runs")

# Session-state keys that together make up a run's pipeline. Model *objects* live
# inside ``results`` / ``unsup_results``; best/selected models are reconstructed
# from their names on load, so we never pickle them twice.
STATE_KEYS = [
    "raw_df", "meta", "target_col", "problem_type", "prep",
    "results", "unsup_results", "best_model_name", "selected_model_name",
    "scaler", "feature_cols", "current_stage", "max_stage",
]


def _ensure() -> None:
    os.makedirs(RUNS_DIR, exist_ok=True)


def _run_path(run_id: str) -> str:
    return os.path.join(RUNS_DIR, run_id)


def _manifest_path(run_id: str) -> str:
    return os.path.join(_run_path(run_id), "manifest.json")


def _state_path(run_id: str) -> str:
    return os.path.join(_run_path(run_id), "state.joblib")


def _read_manifest(run_id: str) -> dict | None:
    p = _manifest_path(run_id)
    if os.path.isfile(p):
        try:
            with open(p, encoding="utf-8") as fh:
                return json.load(fh)
        except Exception:
            return None
    return None


def _write_manifest(run_id: str, mf: dict) -> None:
    with open(_manifest_path(run_id), "w", encoding="utf-8") as fh:
        json.dump(mf, fh, indent=2, default=str)


# ── Public API ───────────────────────────────────────────────────────────────
def list_runs() -> list[dict]:
    """All runs, most-recently-updated first."""
    _ensure()
    out = []
    for d in os.listdir(RUNS_DIR):
        mf = _read_manifest(d)
        if mf:
            out.append(mf)
    return sorted(out, key=lambda m: m.get("updated", 0), reverse=True)


def new_run(name: str) -> str:
    """Create an empty run, return its id."""
    _ensure()
    run_id = time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:4]
    os.makedirs(_run_path(run_id), exist_ok=True)
    _write_manifest(run_id, {
        "id": run_id, "name": (name or "Untitled run").strip()[:60],
        "created": time.time(), "updated": time.time(),
        "current_stage": 0, "max_stage": 0, "has_data": False,
        "target_col": None, "problem_type": None,
        "best_model_name": None, "n_models": 0,
    })
    return run_id


def _strip_unpicklable(state: dict) -> dict:
    """Drop AutoML model objects (FLAML/AutoGluon are not joblib-safe). Their
    metrics stay, so the run still summarises correctly."""
    s = dict(state)
    res = s.get("results")
    if isinstance(res, dict):
        clean = {}
        for name, r in res.items():
            r2 = dict(r) if isinstance(r, dict) else r
            if isinstance(r2, dict) and r2.get("is_automl"):
                r2.pop("model", None)
            clean[name] = r2
        s["results"] = clean
    return s


def save_run(run_id: str, session_state) -> None:
    """Persist the pipeline snapshot + refresh the manifest summary."""
    if not run_id:
        return
    state = {k: session_state.get(k) for k in STATE_KEYS}
    state = _strip_unpicklable(state)
    try:
        joblib.dump(state, _state_path(run_id))
    except Exception:
        # Last resort: drop every model object, keep metrics so the run still loads.
        for r in (state.get("results") or {}).values():
            if isinstance(r, dict):
                r.pop("model", None)
        try:
            joblib.dump(state, _state_path(run_id))
        except Exception:
            return

    mf = _read_manifest(run_id) or {"id": run_id, "name": "Untitled run",
                                    "created": time.time()}
    mf.update({
        "updated": time.time(),
        "current_stage": int(session_state.get("current_stage", mf.get("current_stage", 0)) or 0),
        "max_stage": int(max(session_state.get("max_stage", 0) or 0, mf.get("max_stage", 0) or 0)),
        "has_data": session_state.get("raw_df") is not None,
        "target_col": session_state.get("target_col"),
        "problem_type": session_state.get("problem_type"),
        "best_model_name": session_state.get("best_model_name"),
        "n_models": len(session_state.get("results") or {}) + len(session_state.get("unsup_results") or {}),
    })
    _write_manifest(run_id, mf)


def load_run(run_id: str) -> dict:
    """Return the saved pipeline state dict (empty if none)."""
    p = _state_path(run_id)
    if os.path.isfile(p):
        try:
            return joblib.load(p)
        except Exception:
            return {}
    return {}


def rename_run(run_id: str, name: str) -> None:
    mf = _read_manifest(run_id)
    if mf:
        mf["name"] = (name or mf.get("name", "Untitled run")).strip()[:60]
        mf["updated"] = time.time()
        _write_manifest(run_id, mf)


def delete_runs(run_ids) -> int:
    """Delete the given runs (folders and all). Returns how many were removed."""
    n = 0
    for run_id in run_ids:
        d = _run_path(run_id)
        if os.path.isdir(d):
            shutil.rmtree(d, ignore_errors=True)
            n += 1
    return n
