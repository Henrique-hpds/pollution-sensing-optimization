"""Gera a variavel de SAUDE (internacoes respiratorias) por setor censitario.

Script unico e oficial da saude. Roda no .venv-datasus (pysus/pyreaddbc/dbfread
+ pandas + pyarrow). Faz tudo, em 3 etapas internas:

  1. baixa/decodifica o SIH-RD (AIH Reduzida) do FTP legado do DATASUS, mes a mes
     (RD<UF><AAMM>.dbc), e filtra internacoes respiratorias (DIAG_PRINC ~ 'J') de
     residentes da capital (MUNIC_RES == 355030).
  2. constroi (ou reusa) o lookup CEP -> distrito a partir do CNEFE 2022 (IBGE),
     usando o distrito modal por CEP. Salva cep_to_distrito.parquet.
  3. agrega por distrito: taxa = internacoes / populacao * 1000, e distribui a taxa
     para cada setor do distrito. Salva saude.parquet.

Limitacao de grao: a taxa e' de DISTRITO (96), nao de setor — todo setor do mesmo
distrito herda a mesma taxa. Ver data/considerações.md.

Ano padrao: 2022, para casar temporalmente com o Censo IBGE 2022 (populacao),
o IPVS 2022 e o CNEFE 2022 — consistencia metodologica (analise transversal).

Uso:
    .venv-datasus/bin/python data/scripts/gerar_saude.py [--ano 2022] [--uf SP]
                                                          [--rebuild-lookup] [--no-save]
"""
from __future__ import annotations

import argparse
import ftplib
import warnings
import zipfile
from pathlib import Path

warnings.filterwarnings("ignore")

import pandas as pd
from dbfread import DBF
from pyreaddbc import dbc2dbf

BASE_DIR = Path(__file__).resolve().parents[2]
PROCESSED = BASE_DIR / "data" / "processed_data"
SIH_CACHE = BASE_DIR / "data" / "raw_data" / "sih"                       # .dbc/.dbf baixados
CNEFE_ZIP = BASE_DIR / "data" / "raw_data" / "cnefe" / "3550308_SAO_PAULO.zip"
CNEFE_CSV_NAME = "3550308_SAO_PAULO.csv"
LOOKUP_FILE = PROCESSED / "cep_to_distrito.parquet"
OUT_FILE = PROCESSED / "saude.parquet"

FTP_HOST = "ftp.datasus.gov.br"
FTP_DIR = "/dissemin/publicos/SIHSUS/200801_/Dados"
CAPITAL = ["355030", "3550308"]


# ---------------------------------------------------------------------------
# 1. Download + leitura do SIH-RD
# ---------------------------------------------------------------------------
def _download_dbc(uf: str, ym: str) -> Path:
    SIH_CACHE.mkdir(parents=True, exist_ok=True)
    dest = SIH_CACHE / f"RD{uf}{ym}.dbc"
    if dest.exists() and dest.stat().st_size > 0:
        print(f">> cache: {dest.name}")
        return dest
    print(f">> baixando {dest.name} do FTP DATASUS ...")
    ftp = ftplib.FTP(FTP_HOST, timeout=120)
    ftp.login()
    ftp.cwd(FTP_DIR)
    with open(dest, "wb") as f:
        ftp.retrbinary(f"RETR {dest.name}", f.write)
    ftp.quit()
    return dest


def _read_rd(dbc_path: Path) -> pd.DataFrame:
    dbf_path = dbc_path.with_suffix(".dbf")
    if not dbf_path.exists():
        dbc2dbf(str(dbc_path), str(dbf_path))
    return pd.DataFrame(iter(DBF(str(dbf_path), encoding="latin-1", load=True)))


def coletar_internacoes(uf: str, ano: int) -> pd.DataFrame:
    """Internacoes respiratorias de residentes da capital, ano inteiro; retorna [CEP]."""
    frames = []
    for m in range(1, 13):
        ym = f"{str(ano)[2:]}{m:02d}"
        df = _read_rd(_download_dbc(uf, ym))
        j = df["DIAG_PRINC"].astype(str).str.upper().str.startswith("J")
        cap = df["MUNIC_RES"].astype(str).isin(CAPITAL)
        sub = df.loc[j & cap, ["CEP"]].copy()
        sub["CEP"] = sub["CEP"].astype(str).str.replace(r"\D", "", regex=True).str.zfill(8)
        frames.append(sub)
        print(f">> {ym}: {len(sub)} internacoes resp+capital")
    return pd.concat(frames, ignore_index=True)


# ---------------------------------------------------------------------------
# 2. Lookup CEP -> distrito (CNEFE), com cache
# ---------------------------------------------------------------------------
def obter_lookup_cep_distrito(rebuild: bool = False) -> pd.DataFrame:
    if LOOKUP_FILE.exists() and not rebuild:
        print(f">> lookup CEP->distrito em cache: {LOOKUP_FILE.name}")
        return pd.read_parquet(LOOKUP_FILE)

    print(f">> construindo lookup CEP->distrito do CNEFE ({CNEFE_ZIP.name}) ...")
    with zipfile.ZipFile(CNEFE_ZIP) as z, z.open(CNEFE_CSV_NAME) as f:
        df = pd.read_csv(f, sep=";", usecols=["CEP", "COD_DISTRITO"],
                         dtype={"CEP": "string", "COD_DISTRITO": "string"}, encoding="latin-1")
    df = df.dropna(subset=["CEP", "COD_DISTRITO"])
    df["CEP"] = df["CEP"].str.replace(r"\D", "", regex=True).str.zfill(8)
    df = df[df["CEP"].str.len() == 8]

    # distrito modal por CEP
    counts = df.groupby(["CEP", "COD_DISTRITO"]).size().reset_index(name="n")
    counts = counts.sort_values(["CEP", "n"], ascending=[True, False])
    lookup = counts.drop_duplicates("CEP", keep="first")[["CEP", "COD_DISTRITO"]]
    lookup = lookup.rename(columns={"COD_DISTRITO": "CD_DIST"}).reset_index(drop=True)

    puros = (df.groupby("CEP")["COD_DISTRITO"].nunique() == 1).mean()
    print(f">> {len(lookup)} CEPs | {lookup['CD_DIST'].nunique()} distritos | "
          f"{100*puros:.1f}% dos CEPs num unico distrito")

    LOOKUP_FILE.parent.mkdir(parents=True, exist_ok=True)
    lookup.to_parquet(LOOKUP_FILE, index=False)
    print(f">> salvo: {LOOKUP_FILE}")
    return lookup


# ---------------------------------------------------------------------------
# 3. Taxa por distrito -> setor
# ---------------------------------------------------------------------------
def calcular_saude(uf: str, ano: int, rebuild_lookup: bool) -> tuple[pd.DataFrame, pd.DataFrame]:
    adm = coletar_internacoes(uf, ano)
    print(f">> total internacoes resp+capital {ano}: {len(adm)}")

    lookup = obter_lookup_cep_distrito(rebuild=rebuild_lookup)
    adm = adm.merge(lookup, on="CEP", how="left")
    unmatched = adm["CD_DIST"].isna().mean()
    print(f">> CEP nao encontrado no CNEFE: {100*unmatched:.2f}% "
          f"({adm['CD_DIST'].isna().sum()} de {len(adm)})")
    adm = adm.dropna(subset=["CD_DIST"])

    intern = adm.groupby("CD_DIST").size().rename("internacoes_resp").reset_index()

    ipvs = pd.read_parquet(PROCESSED / "ipvs.parquet")
    ipvs["CD_DIST"] = ipvs["CD_DIST"].astype(str)
    pop = ipvs.groupby("CD_DIST")["POPULACAO"].sum().rename("pop_distrito").reset_index()

    dist = pop.merge(intern, on="CD_DIST", how="left")
    dist["internacoes_resp"] = dist["internacoes_resp"].fillna(0).astype(int)
    dist["taxa_resp_por_1000"] = (dist["internacoes_resp"] / dist["pop_distrito"] * 1000.0).round(4)

    setores = ipvs[["CD_SETOR", "CD_DIST", "NM_DIST", "POPULACAO"]].copy()
    out = setores.merge(dist[["CD_DIST", "internacoes_resp", "taxa_resp_por_1000"]],
                        on="CD_DIST", how="left")
    return out, dist


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--ano", type=int, default=2022)
    ap.add_argument("--uf", default="SP")
    ap.add_argument("--rebuild-lookup", action="store_true", help="reconstroi o lookup CEP->distrito")
    ap.add_argument("--no-save", action="store_true")
    args = ap.parse_args()

    out, dist = calcular_saude(args.uf, args.ano, args.rebuild_lookup)

    print("\n=== VALIDACAO (distrito) ===")
    print("distritos:", len(dist), "| sem internacao:", int((dist["internacoes_resp"] == 0).sum()))
    print(dist[["internacoes_resp", "pop_distrito", "taxa_resp_por_1000"]].describe().round(2).to_string())
    top = dist.merge(out[["CD_DIST", "NM_DIST"]].drop_duplicates(), on="CD_DIST", how="left") \
              .sort_values("taxa_resp_por_1000", ascending=False).head(5)
    print("\ntop 5 distritos por taxa:")
    print(top[["NM_DIST", "internacoes_resp", "pop_distrito", "taxa_resp_por_1000"]].to_string(index=False))
    print("\n=== VALIDACAO (setor) ===")
    print("setores:", len(out), "| sem taxa:", int(out["taxa_resp_por_1000"].isna().sum()))

    if not args.no_save:
        OUT_FILE.parent.mkdir(parents=True, exist_ok=True)
        out.to_parquet(OUT_FILE, index=False)
        print(f"\n>> salvo: {OUT_FILE}")
    else:
        print("\n>> (nao salvo)")


if __name__ == "__main__":
    main()
