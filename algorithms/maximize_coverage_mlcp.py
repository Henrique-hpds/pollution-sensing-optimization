from pathlib import Path

import numpy as np
import pandas as pd
import pulp

from algorithms.base import minmax_norm

BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed_data"

def _load_parquet_by_keyword(keyword: str) -> pd.DataFrame:
    files = sorted(PROCESSED_DATA_DIR.glob(f"*{keyword}*.parquet"))
    if not files:
        raise FileNotFoundError(f"Nenhum parquet encontrado para '{keyword}' em {PROCESSED_DATA_DIR}")
    return pd.read_parquet(files[0])

def _to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")

def _haversine_km(lat: float, lon: float, lat_arr: np.ndarray, lon_arr: np.ndarray) -> np.ndarray:
    earth_radius_km = 6371.0088

    lat1 = np.radians(lat)
    lon1 = np.radians(lon)
    lat2 = np.radians(lat_arr)
    lon2 = np.radians(lon_arr)

    dlat = lat2 - lat1
    dlon = lon2 - lon1

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1) * np.cos(lat2) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))
    return earth_radius_km * c

def _build_coverage_sets(areas: dict, candidates: dict, radius_km: float) -> dict:
    candidate_ids = list(candidates.keys())
    cand_lat = np.array([candidates[j][0] for j in candidate_ids], dtype=float)
    cand_lon = np.array([candidates[j][1] for j in candidate_ids], dtype=float)

    coverage = {}
    for i, data in areas.items():
        lat_i, lon_i = data["coord"]
        distances = _haversine_km(float(lat_i), float(lon_i), cand_lat, cand_lon)
        covered_idx = np.where(distances <= radius_km)[0]
        coverage[i] = [candidate_ids[k] for k in covered_idx]
    return coverage

def load_data() -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    integrated = _load_parquet_by_keyword("integrated")
    ubs = _load_parquet_by_keyword("ubs")
    cetesb = _load_parquet_by_keyword("cetesb")
    return integrated, ubs, cetesb

def build_model(
    p: int = 10,
    alpha: float = 1.0,
    beta: float = 1.0,
    gamma: float = 1.0,
    delta: float = 0.0,
    candidate_radius_km: float = 2.0,
    existing_radius_km: float = 3.0) -> tuple[pulp.LpProblem, dict, dict, dict]:
    
    integrated, ubs, cetesb = load_data()

    integrated["latitude"] = _to_float(integrated["latitude"])
    integrated["longitude"] = _to_float(integrated["longitude"])
    integrated["POPULACAO"] = _to_float(integrated["POPULACAO"]).fillna(0)
    integrated["C_IPVS"] = _to_float(integrated["C_IPVS"]).fillna(0)
    ubs["NU_LATITUDE"] = _to_float(ubs["NU_LATITUDE"])
    ubs["NU_LONGITUDE"] = _to_float(ubs["NU_LONGITUDE"])
    cetesb["LATITUDE"] = _to_float(cetesb["LATITUDE"])
    cetesb["LONGITUDE"] = _to_float(cetesb["LONGITUDE"])

    integrated = integrated.dropna(subset=["CD_SETOR", "latitude", "longitude"]).reset_index(drop=True)
    ubs = ubs.dropna(subset=["CO_CNES", "NU_LATITUDE", "NU_LONGITUDE"]).reset_index(drop=True)
    cetesb = cetesb.dropna(subset=["LATITUDE", "LONGITUDE"]).reset_index(drop=True)

    if integrated.empty or ubs.empty:
        raise ValueError("Dados insuficientes para montar o modelo (integrated ou ubs vazio).")

    if "saude" not in integrated.columns:
        integrated["saude"] = 0.0
    else:
        integrated["saude"] = _to_float(integrated["saude"]).fillna(0)

    if "exposicao" not in integrated.columns:
        integrated["exposicao"] = 0.0
    else:
        integrated["exposicao"] = _to_float(integrated["exposicao"]).fillna(0)

    crit_pop = minmax_norm(_to_float(integrated["POPULACAO"]).fillna(0).to_numpy())
    crit_saude = minmax_norm(integrated["saude"].to_numpy())
    crit_ipvs = minmax_norm(_to_float(integrated["C_IPVS"]).fillna(0).to_numpy())
    crit_exp = minmax_norm(integrated["exposicao"].to_numpy())

    areas = {
        row["CD_SETOR"]: {
            "coord": (row["latitude"], row["longitude"]),
            "pop": row["POPULACAO"],
            "saude": row["saude"],
            "ipvs": row["C_IPVS"],
            "exposicao": row["exposicao"],
        }
        for _, row in integrated.iterrows()
    }

    # criterios normalizados [0,1] -> usados SO no peso (mesma convencao do data_loader)
    criteria = {
        cd: {
            "pop": float(crit_pop[k]),
            "saude": float(crit_saude[k]),
            "ipvs": float(crit_ipvs[k]),
            "exposicao": float(crit_exp[k]),
        }
        for k, cd in enumerate(integrated["CD_SETOR"])
    }

    candidates = {
        int(row["CO_CNES"]): (float(row["NU_LATITUDE"]), float(row["NU_LONGITUDE"]))
        for _, row in ubs.iterrows()
    }

    existing = {
        str(idx): (float(row["LATITUDE"]), float(row["LONGITUDE"]))
        for idx, row in cetesb.iterrows()
    }

    I = list(areas.keys())
    J = list(candidates.keys())

    w = {
        i: (
            alpha * criteria[i]["pop"]
            + beta * criteria[i]["saude"]
            + gamma * criteria[i]["ipvs"]
            + delta * criteria[i]["exposicao"]
        )
        for i in I
    }

    if "cetesb_cobertura_3km" in integrated.columns and existing_radius_km == 3.0:
        c = {
            row["CD_SETOR"]: int(row["cetesb_cobertura_3km"])
            for _, row in integrated[["CD_SETOR", "cetesb_cobertura_3km"]].iterrows()
        }
    else:
        c = {}
        existing_lat = np.array([coord[0] for coord in existing.values()], dtype=float)
        existing_lon = np.array([coord[1] for coord in existing.values()], dtype=float)
        for i in I:
            lat_i, lon_i = areas[i]["coord"]
            if existing_lat.size == 0:
                c[i] = 0
                continue
            distances = _haversine_km(float(lat_i), float(lon_i), existing_lat, existing_lon)
            c[i] = int(np.any(distances <= existing_radius_km))

    coverage_by_candidate = _build_coverage_sets(areas, candidates, candidate_radius_km)

    model = pulp.LpProblem("MLCP", pulp.LpMaximize)

    x = pulp.LpVariable.dicts("x", J, cat="Binary")
    y = pulp.LpVariable.dicts("y", I, cat="Binary")

    model += pulp.lpSum(w[i] * y[i] for i in I)

    for i in I:
        model += y[i] <= c[i] + pulp.lpSum(x[j] for j in coverage_by_candidate[i])

    model += pulp.lpSum(x[j] for j in J) <= p

    return model, x, y, coverage_by_candidate


def main() -> None:
    model, x, y, coverage_by_candidate = build_model()
    model.solve()

    installed = [j for j in x if pulp.value(x[j]) == 1]
    covered_areas = [i for i in y if pulp.value(y[i]) == 1]

    print("Status:", pulp.LpStatus[model.status])
    print("\nEstações instaladas:")
    for j in installed:
        print(f"Candidato {j}")

    print("\nÁreas cobertas:", len(covered_areas))
    print("Valor objetivo:", pulp.value(model.objective))
    print("Média de candidatos por área:", round(np.mean([len(v) for v in coverage_by_candidate.values()]), 2))

if __name__ == "__main__":
    main()