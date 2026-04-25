from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd


BASE_DIR = Path(__file__).resolve().parents[1]
PROCESSED_DATA_DIR = BASE_DIR / "data" / "processed_data"
OUTPUT_FILE = PROCESSED_DATA_DIR / "integrated.parquet"


def _to_float(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series.astype(str).str.replace(",", ".", regex=False), errors="coerce")


def _haversine_matrix_km(
    lat1: np.ndarray,
    lon1: np.ndarray,
    lat2: np.ndarray,
    lon2: np.ndarray,
) -> np.ndarray:
    earth_radius_km = 6371.0088

    lat1_rad = np.radians(lat1)[:, None]
    lon1_rad = np.radians(lon1)[:, None]
    lat2_rad = np.radians(lat2)[None, :]
    lon2_rad = np.radians(lon2)[None, :]

    dlat = lat2_rad - lat1_rad
    dlon = lon2_rad - lon1_rad

    a = np.sin(dlat / 2.0) ** 2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon / 2.0) ** 2
    c = 2.0 * np.arctan2(np.sqrt(a), np.sqrt(1.0 - a))

    return earth_radius_km * c


def _nearest_reference(
    source_lat: pd.Series,
    source_lon: pd.Series,
    ref_lat: pd.Series,
    ref_lon: pd.Series,
) -> tuple[np.ndarray, np.ndarray]:
    src_lat = source_lat.to_numpy(dtype=float)
    src_lon = source_lon.to_numpy(dtype=float)
    dst_lat = ref_lat.to_numpy(dtype=float)
    dst_lon = ref_lon.to_numpy(dtype=float)

    dist_matrix = _haversine_matrix_km(src_lat, src_lon, dst_lat, dst_lon)
    nearest_idx = dist_matrix.argmin(axis=1)
    nearest_dist = dist_matrix[np.arange(dist_matrix.shape[0]), nearest_idx]

    return nearest_idx, nearest_dist


def integrate_parquets() -> pd.DataFrame:
    ipvs_file = PROCESSED_DATA_DIR / "ipvs.parquet"
    ubs_file = PROCESSED_DATA_DIR / "ubs.parquet"
    cetesb_file = PROCESSED_DATA_DIR / "cetesb.parquet"

    if not ipvs_file.exists() or not ubs_file.exists() or not cetesb_file.exists():
        missing = [str(p.name) for p in [ipvs_file, ubs_file, cetesb_file] if not p.exists()]
        raise FileNotFoundError(f"Arquivos ausentes em {PROCESSED_DATA_DIR}: {', '.join(missing)}")

    ipvs = pd.read_parquet(ipvs_file).copy()
    ubs = pd.read_parquet(ubs_file).copy()
    cetesb = pd.read_parquet(cetesb_file).copy()

    ipvs["latitude"] = _to_float(ipvs["latitude"])
    ipvs["longitude"] = _to_float(ipvs["longitude"])
    ubs["NU_LATITUDE"] = _to_float(ubs["NU_LATITUDE"])
    ubs["NU_LONGITUDE"] = _to_float(ubs["NU_LONGITUDE"])
    cetesb["LATITUDE"] = _to_float(cetesb["LATITUDE"])
    cetesb["LONGITUDE"] = _to_float(cetesb["LONGITUDE"])

    ipvs = ipvs.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    ubs = ubs.dropna(subset=["NU_LATITUDE", "NU_LONGITUDE"]).reset_index(drop=True)
    cetesb = cetesb.dropna(subset=["LATITUDE", "LONGITUDE"]).reset_index(drop=True)

    if ipvs.empty or ubs.empty or cetesb.empty:
        raise ValueError("Um dos dataframes ficou vazio apos limpeza de coordenadas.")

    ubs_idx, ubs_dist = _nearest_reference(
        ipvs["latitude"],
        ipvs["longitude"],
        ubs["NU_LATITUDE"],
        ubs["NU_LONGITUDE"],
    )
    cetesb_idx, cetesb_dist = _nearest_reference(
        ipvs["latitude"],
        ipvs["longitude"],
        cetesb["LATITUDE"],
        cetesb["LONGITUDE"],
    )

    ubs_nearest = ubs.loc[ubs_idx, ["CO_UNIDADE", "CO_CNES", "NO_FANTASIA", "NO_BAIRRO"]].reset_index(drop=True)
    cetesb_nearest = cetesb.loc[cetesb_idx, ["Nome Estacao", "TIPO"]].reset_index(drop=True)

    integrated = ipvs.reset_index(drop=True).copy()
    integrated["ubs_mais_proxima_km"] = np.round(ubs_dist, 4)
    integrated["cetesb_mais_proxima_km"] = np.round(cetesb_dist, 4)
    integrated["ubs_cobertura_2km"] = (integrated["ubs_mais_proxima_km"] <= 2.0).astype(int)
    integrated["cetesb_cobertura_3km"] = (integrated["cetesb_mais_proxima_km"] <= 3.0).astype(int)

    integrated = pd.concat(
        [
            integrated,
            ubs_nearest.rename(
                columns={
                    "CO_UNIDADE": "ubs_co_unidade_mais_proxima",
                    "CO_CNES": "ubs_co_cnes_mais_proxima",
                    "NO_FANTASIA": "ubs_nome_mais_proxima",
                    "NO_BAIRRO": "ubs_bairro_mais_proxima",
                }
            ),
            cetesb_nearest.rename(
                columns={
                    "Nome Estacao": "cetesb_nome_estacao_mais_proxima",
                    "TIPO": "cetesb_tipo_estacao_mais_proxima",
                }
            ),
        ],
        axis=1,
    )

    return integrated


def main() -> None:
    integrated = integrate_parquets()
    OUTPUT_FILE.parent.mkdir(parents=True, exist_ok=True)
    integrated.to_parquet(OUTPUT_FILE, index=False)
    print(f"Arquivo integrado salvo em: {OUTPUT_FILE}")
    print(f"Linhas: {integrated.shape[0]} | Colunas: {integrated.shape[1]}")


if __name__ == "__main__":
    main()