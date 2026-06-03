"""
Shared I/O for ML1.py and DFM.py result snapshots.

Layout (one subdirectory per run):
    <results_dir>/<TS>/meta.json
    <results_dir>/<TS>/<key>.parquet            (DataFrames)
    <results_dir>/<TS>/<key>.npz                (unlabeled numpy bundles)

`save_run` accepts a dict of artifacts; `load_run` returns the same dict.
DataFrames go through parquet, dicts of numpy arrays through .npz, anything
else (scalars, lists, strings) through meta.json.
"""

from __future__ import annotations

import json
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Iterable, Optional

import numpy as np
import pandas as pd


# --------------------------------------------------------------------------- #
# Save
# --------------------------------------------------------------------------- #


def make_run_id(now: Optional[datetime] = None) -> str:
    """`YYYYMMDD_HHMMSS` timestamp suitable for sortable directory names."""
    return (now or datetime.now()).strftime("%Y%m%d_%H%M%S")


def save_run(
    base_dir: str | Path,
    artifacts: Dict[str, Any],
    meta: Optional[Dict[str, Any]] = None,
    run_id: Optional[str] = None,
) -> Path:
    """
    Persist a dict of artifacts to `<base_dir>/<run_id>/`.

    Dispatch by type:
        pd.DataFrame / pd.Series   -> <key>.parquet
        dict[str, np.ndarray]      -> <key>.npz
        np.ndarray (alone)         -> <key>.npz with key 'arr_0'
        anything JSON-serializable -> merged into meta.json under "extras"

    Returns the absolute path of the run directory.
    """
    base = Path(base_dir)
    run_id = run_id or make_run_id()
    run_dir = base / run_id
    run_dir.mkdir(parents=True, exist_ok=True)

    extras: Dict[str, Any] = {}
    for key, value in artifacts.items():
        if isinstance(value, pd.DataFrame):
            value.to_parquet(run_dir / f"{key}.parquet")
        elif isinstance(value, pd.Series):
            value.to_frame(name=value.name or key).to_parquet(
                run_dir / f"{key}.parquet"
            )
        elif isinstance(value, dict) and value and all(
            isinstance(v, np.ndarray) for v in value.values()
        ):
            np.savez(run_dir / f"{key}.npz", **value)
        elif isinstance(value, np.ndarray):
            np.savez(run_dir / f"{key}.npz", arr_0=value)
        else:
            extras[key] = _json_safe(value)

    full_meta = dict(meta or {})
    if extras:
        full_meta.setdefault("extras", {}).update(extras)
    full_meta["run_id"] = run_id
    full_meta["saved_at"] = datetime.now().isoformat(timespec="seconds")
    with open(run_dir / "meta.json", "w") as f:
        json.dump(full_meta, f, indent=2, default=_json_safe)

    return run_dir


# --------------------------------------------------------------------------- #
# Load
# --------------------------------------------------------------------------- #


def list_runs(base_dir: str | Path) -> list[str]:
    """All run_ids under `base_dir`, sorted descending (latest first)."""
    base = Path(base_dir)
    if not base.exists():
        return []
    runs = []
    for p in base.iterdir():
        if not p.is_dir():
            continue
        # accept any directory that contains a meta.json or any parquet
        if (p / "meta.json").exists() or any(p.glob("*.parquet")):
            runs.append(p.name)
    return sorted(runs, reverse=True)


def load_run(
    base_dir: str | Path,
    run_id: Optional[str] = None,
    keys: Optional[Iterable[str]] = None,
) -> Dict[str, Any]:
    """
    Load all artifacts in `<base_dir>/<run_id>/`.

    If `run_id` is None the latest run is used. If `keys` is given, only those
    artifact names are loaded (still returns meta.json).

    The returned dict has the loaded artifacts plus:
        "_run_dir":  Path of the loaded run
        "_run_id":   the loaded run id
        "_meta":     contents of meta.json (or {})
    """
    base = Path(base_dir)
    if run_id is None:
        ids = list_runs(base)
        if not ids:
            raise FileNotFoundError(f"No runs found under {base}")
        run_id = ids[0]
    run_dir = base / run_id
    if not run_dir.is_dir():
        raise FileNotFoundError(run_dir)

    meta: Dict[str, Any] = {}
    if (run_dir / "meta.json").exists():
        with open(run_dir / "meta.json") as f:
            meta = json.load(f)

    out: Dict[str, Any] = {"_run_dir": run_dir, "_run_id": run_id, "_meta": meta}

    keep = set(keys) if keys is not None else None
    for f in sorted(run_dir.iterdir()):
        if f.name == "meta.json":
            continue
        key = f.stem
        if keep is not None and key not in keep:
            continue
        if f.suffix == ".parquet":
            out[key] = pd.read_parquet(f)
        elif f.suffix == ".npz":
            with np.load(f, allow_pickle=False) as npz:
                d = {k: npz[k] for k in npz.files}
            # unwrap single-array bundles saved as 'arr_0'
            out[key] = d["arr_0"] if list(d) == ["arr_0"] else d
    return out


# --------------------------------------------------------------------------- #
# cluster_states <-> long-form parquet
# --------------------------------------------------------------------------- #


def cluster_states_to_long(states: Dict[str, Dict[str, pd.Series]]) -> pd.DataFrame:
    """
    {cluster: {metric: Series}}  ->  long-form DataFrame (cluster, t, metric, value).
    Robust to clusters with non-overlapping time indices.
    """
    rows = []
    for cluster, metrics in states.items():
        for metric, s in metrics.items():
            ss = pd.Series(s).dropna()
            if ss.empty:
                continue
            df = pd.DataFrame(
                {
                    "cluster": cluster,
                    "metric": metric,
                    "t": ss.index.astype("int64", copy=False),
                    "value": ss.values.astype("float64", copy=False),
                }
            )
            rows.append(df)
    if not rows:
        return pd.DataFrame(columns=["cluster", "metric", "t", "value"])
    return pd.concat(rows, ignore_index=True)


def cluster_states_from_long(
    long: pd.DataFrame,
) -> Dict[str, Dict[str, pd.Series]]:
    """Inverse of `cluster_states_to_long`. Returns {cluster: {metric: Series}}."""
    out: Dict[str, Dict[str, pd.Series]] = {}
    if long.empty:
        return out
    for (cluster, metric), grp in long.groupby(["cluster", "metric"], sort=False):
        s = grp.set_index("t")["value"].sort_index()
        s.name = metric
        out.setdefault(cluster, {})[metric] = s
    return out


# --------------------------------------------------------------------------- #
# Internals
# --------------------------------------------------------------------------- #


def _json_safe(obj: Any) -> Any:
    """Recursively coerce obj into JSON-serialisable primitives."""
    if isinstance(obj, (str, bool)) or obj is None:
        return obj
    if isinstance(obj, (int, np.integer)):
        return int(obj)
    if isinstance(obj, (float, np.floating)):
        v = float(obj)
        return v if np.isfinite(v) else None
    if isinstance(obj, (list, tuple)):
        return [_json_safe(x) for x in obj]
    if isinstance(obj, dict):
        return {str(k): _json_safe(v) for k, v in obj.items()}
    if isinstance(obj, np.ndarray):
        return _json_safe(obj.tolist())
    if isinstance(obj, (pd.Series, pd.DataFrame)):
        return _json_safe(obj.to_dict())
    if isinstance(obj, Path):
        return str(obj)
    return str(obj)
