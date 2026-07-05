"""Builds and caches the sector × candidate distance matrix."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

CACHE_DIR = Path(__file__).resolve().parent / "cache"


def haversine_matrix(lat1: np.ndarray, lon1: np.ndarray, lat2: np.ndarray, lon2: np.ndarray) -> np.ndarray:
    """Returns (len(lat1), len(lat2)) matrix of distances in km."""
    R = 6371.0088
    la1 = np.radians(lat1)[:, None]
    lo1 = np.radians(lon1)[:, None]
    la2 = np.radians(lat2)[None, :]
    lo2 = np.radians(lon2)[None, :]
    dlat = la2 - la1
    dlon = lo2 - lo1
    a = np.sin(dlat / 2) ** 2 + np.cos(la1) * np.cos(la2) * np.sin(dlon / 2) ** 2
    return R * 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))


def get_or_build(dataset_hash: str, area_lats: np.ndarray, area_lons: np.ndarray, cand_lats: np.ndarray, cand_lons: np.ndarray) -> np.ndarray:
    """Returns float32 distance matrix (n_areas × n_candidates), building and persisting if needed."""
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    tag = dataset_hash[:16]
    npy_path = CACHE_DIR / f"dist_{tag}.npy"
    meta_path = CACHE_DIR / f"dist_{tag}_meta.json"

    if npy_path.exists() and meta_path.exists():
        meta = json.loads(meta_path.read_text())
        if meta.get("dataset_hash") == dataset_hash:
            return np.load(npy_path)

    dmat = haversine_matrix(area_lats, area_lons, cand_lats, cand_lons).astype(np.float32)
    np.save(npy_path, dmat)
    meta_path.write_text(json.dumps({"dataset_hash": dataset_hash, "shape": list(dmat.shape)}))
    return dmat
