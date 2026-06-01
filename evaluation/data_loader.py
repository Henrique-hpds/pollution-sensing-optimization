"""Loads the three parquets into ProblemData (with cached distance matrix)."""

from __future__ import annotations

import hashlib
from pathlib import Path

import numpy as np
import pandas as pd

from algorithms.base import ProblemData
from evaluation.distance_cache import get_or_build, haversine_matrix

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed_data"

_DEFAULT_WEIGHTS = {"alpha": 1.0, "beta": 0.0, "gamma": 1.0, "delta": 0.0}


def _parquet_hash(*paths: Path) -> str:
    h = hashlib.sha256()
    for p in sorted(paths):
        h.update(p.read_bytes())
    return h.hexdigest()


def _to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(
        series.astype(str).str.replace(",", ".", regex=False), errors="coerce"
    )


def load(
    radius_km: float = 2.0,
    weights: dict | None = None,
    processed_dir: Path | None = None,
) -> ProblemData:
    w = {**_DEFAULT_WEIGHTS, **(weights or {})}
    data_dir = processed_dir or PROCESSED_DATA_DIR

    integrated_path = data_dir / "integrated.parquet"
    ubs_path = data_dir / "ubs.parquet"
    cetesb_path = data_dir / "cetesb.parquet"

    dataset_hash = _parquet_hash(integrated_path, ubs_path, cetesb_path)

    integrated = pd.read_parquet(integrated_path)
    ubs = pd.read_parquet(ubs_path)
    cetesb = pd.read_parquet(cetesb_path)

    # --- clean integrated ---
    integrated["latitude"] = _to_float(integrated["latitude"])
    integrated["longitude"] = _to_float(integrated["longitude"])
    integrated["POPULACAO"] = _to_float(integrated["POPULACAO"]).fillna(0)
    integrated["C_IPVS"] = _to_float(integrated["C_IPVS"]).fillna(0)
    for col in ("saude", "exposicao"):
        if col not in integrated.columns:
            integrated[col] = 0.0
        else:
            integrated[col] = _to_float(integrated[col]).fillna(0)
    integrated = integrated.dropna(subset=["CD_SETOR", "latitude", "longitude"]).reset_index(drop=True)

    # --- clean ubs ---
    ubs["NU_LATITUDE"] = _to_float(ubs["NU_LATITUDE"])
    ubs["NU_LONGITUDE"] = _to_float(ubs["NU_LONGITUDE"])
    ubs = ubs.dropna(subset=["CO_CNES", "NU_LATITUDE", "NU_LONGITUDE"]).reset_index(drop=True)

    # --- clean cetesb ---
    cetesb["LATITUDE"] = _to_float(cetesb["LATITUDE"])
    cetesb["LONGITUDE"] = _to_float(cetesb["LONGITUDE"])
    cetesb = cetesb.dropna(subset=["LATITUDE", "LONGITUDE"]).reset_index(drop=True)

    # --- build dicts ---
    areas: dict = {}
    area_index: dict = {}
    for idx, row in integrated.iterrows():
        cd = str(row["CD_SETOR"])
        areas[cd] = {
            "coord": (float(row["latitude"]), float(row["longitude"])),
            "pop": float(row["POPULACAO"]),
            "saude": float(row["saude"]),
            "ipvs": float(row["C_IPVS"]),
            "exposicao": float(row["exposicao"]),
            "cd_dist": str(row["CD_DIST"]),
        }
        area_index[cd] = int(idx)

    candidates: dict = {}
    cand_index: dict = {}
    for idx, row in ubs.iterrows():
        cid = int(row["CO_CNES"])
        candidates[cid] = (float(row["NU_LATITUDE"]), float(row["NU_LONGITUDE"]))
        cand_index[cid] = int(idx)

    existing: dict = {}
    existing_index: dict = {}
    for idx, row in cetesb.iterrows():
        eid = str(idx)
        existing[eid] = (float(row["LATITUDE"]), float(row["LONGITUDE"]))
        existing_index[eid] = int(idx)

    # --- distance matrices ---
    area_ids = sorted(area_index, key=area_index.__getitem__)
    cand_ids = sorted(cand_index, key=cand_index.__getitem__)
    exist_ids = sorted(existing_index, key=existing_index.__getitem__)

    area_lats = np.array([areas[a]["coord"][0] for a in area_ids], dtype=np.float64)
    area_lons = np.array([areas[a]["coord"][1] for a in area_ids], dtype=np.float64)
    cand_lats = np.array([candidates[j][0] for j in cand_ids], dtype=np.float64)
    cand_lons = np.array([candidates[j][1] for j in cand_ids], dtype=np.float64)

    distance_matrix = get_or_build(
        dataset_hash, area_lats, area_lons, cand_lats, cand_lons
    )

    if exist_ids:
        ext_lats = np.array([existing[e][0] for e in exist_ids], dtype=np.float64)
        ext_lons = np.array([existing[e][1] for e in exist_ids], dtype=np.float64)
        existing_distances = haversine_matrix(
            area_lats, area_lons, ext_lats, ext_lons
        ).astype(np.float32)
    else:
        existing_distances = np.empty((len(area_ids), 0), dtype=np.float32)

    return ProblemData(
        areas=areas,
        candidates=candidates,
        existing=existing,
        distance_matrix=distance_matrix,
        existing_distances=existing_distances,
        area_index=area_index,
        cand_index=cand_index,
        existing_index=existing_index,
        weights=w,
        radius_km=radius_km,
        dataset_hash=dataset_hash,
    )
