"""Gera a variavel de EXPOSICAO (malha viaria) por setor censitario.

Script unico e oficial da exposicao. Roda no .venv (osmnx + geopandas).
Faz tudo: baixa a malha viaria principal do OpenStreetMap (com cache),
calcula as duas metricas por setor e salva em processed_data/exposicao.parquet.

Metricas (calculadas em EPSG:31983, metros):
  - densidade_viaria_500m_km : km de vias principais dentro de 500 m do centroide
                               (metrica usada como 'exposicao' no pipeline)
  - dist_via_principal_m     : distancia (m) ate a via principal mais proxima

Uso:
    .venv/bin/python data/scripts/gerar_exposicao.py [--force] [--sample N] [--no-save]
      --force    : rebaixa a malha viaria mesmo se houver cache
      --sample N : calcula so N setores (teste rapido), nao salva
      --no-save  : nao escreve o parquet (so imprime validacao)
"""
from __future__ import annotations

import sys
import time
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

import geopandas as gpd
import osmnx as ox
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED = BASE_DIR / "data" / "processed_data"
ROADS_CACHE = BASE_DIR / "data" / "raw_data" / "vias_principais_sp.parquet"
OUT_FILE = PROCESSED / "exposicao.parquet"

# tipos de via "principais" (maior fluxo -> maior exposicao); ruas locais ficam de fora
HIGHWAY_TAGS = {"highway": ["motorway", "trunk", "primary", "secondary"]}
PLACE = "São Paulo, São Paulo, Brazil"

CRS_M = 31983   # SIRGAS 2000 / UTM 23S (metros)
CRS_LL = 4326   # WGS84 (graus)
BUFFER_M = 500.0


def _arg_int(flag: str, default: int | None) -> int | None:
    if flag in sys.argv:
        return int(sys.argv[sys.argv.index(flag) + 1])
    return default


# ---------------------------------------------------------------------------
# 1. Coleta da malha viaria (OSM, com cache)
# ---------------------------------------------------------------------------
def coletar_vias(force: bool = False) -> gpd.GeoDataFrame:
    if ROADS_CACHE.exists() and not force:
        print(f">> malha viaria em cache: {ROADS_CACHE}")
        return gpd.read_parquet(ROADS_CACHE)

    print(f">> baixando malha viaria principal de: {PLACE}")
    gdf = ox.features.features_from_place(PLACE, HIGHWAY_TAGS)
    gdf = gdf[gdf.geometry.type.isin(["LineString", "MultiLineString"])].copy()
    keep = [c for c in ["highway", "name", "ref", "lanes", "maxspeed"] if c in gdf.columns]
    gdf = gdf.reset_index()[keep + ["geometry"]]

    ROADS_CACHE.parent.mkdir(parents=True, exist_ok=True)
    gdf.to_parquet(ROADS_CACHE)
    print(f">> salvo cache: {ROADS_CACHE} | segmentos: {len(gdf)}")
    print(">> por tipo:", gdf["highway"].astype(str).value_counts().to_dict())
    return gdf


# ---------------------------------------------------------------------------
# 2. Metricas de exposicao por setor
# ---------------------------------------------------------------------------
def calcular_exposicao(roads_ll: gpd.GeoDataFrame, sample: int | None = None) -> pd.DataFrame:
    integrated = pd.read_parquet(PROCESSED / "integrated.parquet")
    integrated = integrated.dropna(subset=["latitude", "longitude"]).reset_index(drop=True)
    if sample:
        integrated = integrated.sample(sample, random_state=42).reset_index(drop=True)

    pts = gpd.GeoDataFrame(
        integrated[["CD_SETOR"]].copy(),
        geometry=gpd.points_from_xy(integrated["longitude"], integrated["latitude"]),
        crs=CRS_LL,
    ).to_crs(CRS_M)

    roads = roads_ll.to_crs(CRS_M)
    roads = roads[roads.geometry.type.isin(["LineString", "MultiLineString"])].reset_index(drop=True)
    print(f">> setores: {len(pts)} | segmentos de via: {len(roads)}")

    # Metrica 2: distancia a via mais proxima (ponto->linha)
    t0 = time.time()
    near = gpd.sjoin_nearest(pts[["CD_SETOR", "geometry"]], roads[["geometry"]], distance_col="_d")
    dist_m = near.groupby("CD_SETOR")["_d"].min()
    print(f">> dist via mais proxima: {time.time()-t0:.1f}s")

    # Metrica 1: comprimento de via dentro do buffer de 500 m
    t0 = time.time()
    buffers = gpd.GeoDataFrame(
        {"CD_SETOR": pts["CD_SETOR"].values},
        geometry=pts.geometry.buffer(BUFFER_M),
        crs=CRS_M,
    )
    pairs = gpd.sjoin(roads[["geometry"]], buffers, predicate="intersects")
    bgeom = buffers.set_index("CD_SETOR").geometry
    pairs = pairs.merge(bgeom.rename("bgeom"), left_on="CD_SETOR", right_index=True)
    clipped = pairs.geometry.intersection(gpd.GeoSeries(pairs["bgeom"].values, crs=CRS_M, index=pairs.index))
    pairs["_len_m"] = clipped.length
    dens_km = pairs.groupby("CD_SETOR")["_len_m"].sum() / 1000.0
    print(f">> densidade em buffer: {time.time()-t0:.1f}s")

    out = integrated[["CD_SETOR"]].copy()
    out["dist_via_principal_m"] = out["CD_SETOR"].map(dist_m).round(2)
    out["densidade_viaria_500m_km"] = out["CD_SETOR"].map(dens_km).fillna(0.0).round(4)
    return out


def main() -> None:
    force = "--force" in sys.argv
    sample = _arg_int("--sample", None)
    save = "--no-save" not in sys.argv and sample is None

    roads = coletar_vias(force=force)
    out = calcular_exposicao(roads, sample=sample)

    print("\n=== VALIDACAO ===")
    print("linhas:", len(out), "| CD_SETOR unicos:", out["CD_SETOR"].nunique())
    print("nulos dist_via:", int(out["dist_via_principal_m"].isna().sum()))
    print(out[["dist_via_principal_m", "densidade_viaria_500m_km"]].describe().round(2).to_string())
    print("setores sem via em 500m (densidade=0):", int((out["densidade_viaria_500m_km"] == 0).sum()))

    if save:
        OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(OUT_FILE, index=False)
        print(f"\n>> salvo: {OUT_FILE}")
    else:
        print("\n>> (nao salvo)")


if __name__ == "__main__":
    main()
