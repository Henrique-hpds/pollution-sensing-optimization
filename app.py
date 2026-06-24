"""
Interface Streamlit para visualização e comparação de cenários do modelo MLCP
de otimização de pontos de instalação de sensores de qualidade do ar em SP.

Roda com:  streamlit run app.py
"""
from __future__ import annotations

import json
import sys
import time
from copy import deepcopy
from pathlib import Path

import branca.colormap as cm
import folium
import geopandas as gpd
import numpy as np
import pandas as pd
import pulp
import shapely.wkb
import shapely.wkt
import streamlit as st
from folium.plugins import MarkerCluster
from shapely.geometry import mapping
from streamlit_folium import st_folium

BASE_DIR = Path(__file__).resolve().parent
sys.path.insert(0, str(BASE_DIR))

from algorithms.maximize_coverage_mlcp import build_model, load_data  # noqa: E402

st.set_page_config(page_title="MLCP — Sensores de Ar SP", layout="wide")

# ---------------------------------------------------------------------------
# Carregamento de dados (cacheado)
# ---------------------------------------------------------------------------

def _parse_geom(value):
    if value is None:
        return None
    if hasattr(value, "geom_type"):
        return value
    if isinstance(value, (bytes, bytearray, memoryview)):
        try:
            return shapely.wkb.loads(bytes(value))
        except Exception:
            return None
    if isinstance(value, str):
        s = value.strip()
        if not s:
            return None
        try:
            return shapely.wkt.loads(s)
        except Exception:
            try:
                return shapely.wkb.loads(bytes.fromhex(s))
            except Exception:
                return None
    return None


@st.cache_data(show_spinner="Carregando dados...")
def carregar_dados() -> dict:
    integrated, ubs, cetesb = load_data()

    # cast colunas numéricas (mesmo tratamento do build_model, replicado p/ UI)
    for col in ("latitude", "longitude", "C_IPVS", "POPULACAO"):
        if col in integrated.columns:
            integrated[col] = pd.to_numeric(
                integrated[col].astype(str).str.replace(",", ".", regex=False),
                errors="coerce",
            )
    if "saude" not in integrated.columns:
        integrated["saude"] = 0.0
    if "exposicao" not in integrated.columns:
        integrated["exposicao"] = 0.0

    integrated = integrated.dropna(subset=["CD_SETOR", "latitude", "longitude"]).reset_index(drop=True)

    # parse geometria + simplifica
    geoms = integrated["geometry"].apply(_parse_geom)
    gdf = gpd.GeoDataFrame(
        integrated.drop(columns=["geometry"]),
        geometry=gpd.GeoSeries(geoms, crs="EPSG:4326"),
        crs="EPSG:4326",
    )
    gdf = gdf[~gdf.geometry.is_empty & gdf.geometry.notna()].reset_index(drop=True)
    gdf["geometry"] = gdf.geometry.simplify(tolerance=0.00005, preserve_topology=True)

    ubs["NU_LATITUDE"] = pd.to_numeric(
        ubs["NU_LATITUDE"].astype(str).str.replace(",", ".", regex=False), errors="coerce"
    )
    ubs["NU_LONGITUDE"] = pd.to_numeric(
        ubs["NU_LONGITUDE"].astype(str).str.replace(",", ".", regex=False), errors="coerce"
    )
    ubs = ubs.dropna(subset=["CO_CNES", "NU_LATITUDE", "NU_LONGITUDE"]).reset_index(drop=True)
    ubs["CO_CNES"] = ubs["CO_CNES"].astype(int)

    cetesb["LATITUDE"] = pd.to_numeric(
        cetesb["LATITUDE"].astype(str).str.replace(",", ".", regex=False), errors="coerce"
    )
    cetesb["LONGITUDE"] = pd.to_numeric(
        cetesb["LONGITUDE"].astype(str).str.replace(",", ".", regex=False), errors="coerce"
    )
    cetesb = cetesb.dropna(subset=["LATITUDE", "LONGITUDE"]).reset_index(drop=True)

    return {"setores": gdf, "ubs": ubs, "cetesb": cetesb}


# ---------------------------------------------------------------------------
# Solver
# ---------------------------------------------------------------------------

def resolver(params: dict) -> dict:
    """Roda build_model + solve, devolve um dicionário-resultado serializável."""
    t0 = time.perf_counter()
    model, x, y, cov_by_cand = build_model(
        p=int(params["P"]),
        alpha=float(params["alpha"]),
        beta=float(params["beta"]),
        gamma=float(params["gamma"]),
        delta=float(params["delta"]),
        candidate_radius_km=float(params["candidate_radius_km"]),
        existing_radius_km=float(params["existing_radius_km"]),
    )
    solver = pulp.PULP_CBC_CMD(msg=False, timeLimit=60)
    status_code = model.solve(solver)
    elapsed = time.perf_counter() - t0

    status = pulp.LpStatus[status_code]

    if status not in ("Optimal", "Not Solved", "Undefined"):
        return {
            "status": status,
            "tempo_s": elapsed,
            "params": params,
            "instalados": [],
            "cobertos": [],
            "objetivo": None,
        }

    instalados = [int(j) for j in x if pulp.value(x[j]) is not None and pulp.value(x[j]) > 0.5]
    cobertos = [i for i in y if pulp.value(y[i]) is not None and pulp.value(y[i]) > 0.5]
    objetivo = pulp.value(model.objective)

    return {
        "status": status,
        "tempo_s": elapsed,
        "params": params,
        "instalados": instalados,
        "cobertos": cobertos,
        "objetivo": float(objetivo) if objetivo is not None else None,
    }


def calcular_pesos(setores: pd.DataFrame, params: dict) -> pd.Series:
    return (
        params["alpha"] * setores["POPULACAO"].fillna(0)
        + params["beta"] * setores.get("saude", 0).fillna(0)
        + params["gamma"] * setores["C_IPVS"].fillna(0)
        + params["delta"] * setores.get("exposicao", 0).fillna(0)
    )


def calcular_metricas(resultado: dict, dados: dict) -> dict:
    setores = dados["setores"]
    params = resultado["params"]
    w = calcular_pesos(setores, params)
    cobertos_set = set(resultado["cobertos"])
    mask_cob = setores["CD_SETOR"].isin(cobertos_set)

    pop_total = float(setores["POPULACAO"].fillna(0).sum())
    pop_coberta = float(setores.loc[mask_cob, "POPULACAO"].fillna(0).sum())
    soma_w = float(w.sum())

    ipvs_cob = setores.loc[mask_cob, "C_IPVS"].fillna(0)
    ipvs_ncob = setores.loc[~mask_cob, "C_IPVS"].fillna(0)

    # decomposição da contribuição de cada termo na função objetivo
    contrib = {
        "alpha·população": float(params["alpha"] * setores.loc[mask_cob, "POPULACAO"].fillna(0).sum()),
        "beta·saúde": float(params["beta"] * setores.loc[mask_cob, "saude"].fillna(0).sum())
        if "saude" in setores.columns else 0.0,
        "gamma·IPVS": float(params["gamma"] * setores.loc[mask_cob, "C_IPVS"].fillna(0).sum()),
        "delta·exposição": float(params["delta"] * setores.loc[mask_cob, "exposicao"].fillna(0).sum())
        if "exposicao" in setores.columns else 0.0,
    }

    return {
        "objetivo": resultado["objetivo"],
        "n_cobertos": int(mask_cob.sum()),
        "n_total": int(len(setores)),
        "pop_coberta_pct": (pop_coberta / pop_total * 100) if pop_total else 0.0,
        "cobertura_ponderada_pct": (resultado["objetivo"] / soma_w * 100) if (soma_w and resultado["objetivo"]) else 0.0,
        "ipvs_medio_cob": float(ipvs_cob.mean()) if len(ipvs_cob) else 0.0,
        "ipvs_medio_ncob": float(ipvs_ncob.mean()) if len(ipvs_ncob) else 0.0,
        "contrib": contrib,
        "pop_coberta": pop_coberta,
        "pop_total": pop_total,
    }


# ---------------------------------------------------------------------------
# Mapa
# ---------------------------------------------------------------------------

def construir_mapa(resultado: dict | None, dados: dict, params: dict) -> folium.Map:
    setores = dados["setores"]
    ubs = dados["ubs"]
    cetesb = dados["cetesb"]

    m = folium.Map(location=[-23.55, -46.63], zoom_start=11, tiles="cartodbpositron")

    cobertos_set = set(resultado["cobertos"]) if resultado else set()
    instalados_set = set(resultado["instalados"]) if resultado else set()

    # ---- setores: pesos e cobertura ----
    # 27k polígonos travam o folium. Mostramos: todos os cobertos + amostra dos não-cobertos.
    w = calcular_pesos(setores, params)
    setores_view = setores.copy()
    setores_view["w"] = w
    if cobertos_set:
        mask_cob = setores_view["CD_SETOR"].isin(cobertos_set)
        amostra_ncob = setores_view[~mask_cob].sample(
            n=min(2000, int((~mask_cob).sum())), random_state=42
        )
        setores_view = pd.concat([setores_view[mask_cob], amostra_ncob])
    else:
        setores_view = setores_view.sample(n=min(3000, len(setores_view)), random_state=42)

    if len(setores_view):
        wmax = float(setores_view["w"].quantile(0.95)) or 1.0
        colormap = cm.linear.YlOrRd_09.scale(0, wmax)
        colormap.caption = "Peso w_i (α·pop + β·saúde + γ·IPVS + δ·exp)"

        def style_setor(feat):
            cd = feat["properties"]["CD_SETOR"]
            wi = feat["properties"]["w"]
            outline = "#2ecc71" if cd in cobertos_set else "#cccccc"
            outline_w = 1.2 if cd in cobertos_set else 0.3
            fill = colormap(min(wi, wmax)) if wi and wi > 0 else "#eeeeee"
            return {
                "fillColor": fill,
                "color": outline,
                "weight": outline_w,
                "fillOpacity": 0.55,
            }

        # GeoJson aceita GeoDataFrame diretamente
        layer_setores = folium.FeatureGroup(name="Setores censitários", show=True)
        folium.GeoJson(
            setores_view[["CD_SETOR", "w", "POPULACAO", "C_IPVS", "geometry"]].to_json(),
            style_function=style_setor,
            tooltip=folium.GeoJsonTooltip(
                fields=["CD_SETOR", "POPULACAO", "C_IPVS", "w"],
                aliases=["Setor", "População", "IPVS", "Peso"],
                localize=True,
            ),
        ).add_to(layer_setores)
        layer_setores.add_to(m)
        m.add_child(colormap)

    # ---- CETESB existentes ----
    layer_cetesb = folium.FeatureGroup(name="Estações CETESB (existentes)", show=True)
    nome_col = "Nome Estacao" if "Nome Estacao" in cetesb.columns else cetesb.columns[0]
    for _, row in cetesb.iterrows():
        folium.CircleMarker(
            location=[row["LATITUDE"], row["LONGITUDE"]],
            radius=5,
            color="#2c3e50",
            fill=True,
            fill_color="#3498db",
            fill_opacity=0.9,
            popup=str(row.get(nome_col, "")),
        ).add_to(layer_cetesb)
        folium.Circle(
            location=[row["LATITUDE"], row["LONGITUDE"]],
            radius=params["existing_radius_km"] * 1000,
            color="#3498db",
            weight=1,
            fill=False,
            opacity=0.4,
        ).add_to(layer_cetesb)
    layer_cetesb.add_to(m)

    # ---- UBSs candidatas não escolhidas ----
    layer_ubs_n = folium.FeatureGroup(name="UBSs candidatas (não escolhidas)", show=False)
    nao_escolhidas = ubs[~ubs["CO_CNES"].isin(instalados_set)] if instalados_set else ubs
    for _, row in nao_escolhidas.iterrows():
        folium.CircleMarker(
            location=[row["NU_LATITUDE"], row["NU_LONGITUDE"]],
            radius=2,
            color="#888",
            fill=True,
            fill_opacity=0.5,
            popup=str(row.get("NO_FANTASIA", row["CO_CNES"])),
        ).add_to(layer_ubs_n)
    layer_ubs_n.add_to(m)

    # ---- UBSs escolhidas ----
    if instalados_set:
        layer_ubs_s = folium.FeatureGroup(name="UBSs escolhidas (sensor instalado)", show=True)
        escolhidas = ubs[ubs["CO_CNES"].isin(instalados_set)]
        for _, row in escolhidas.iterrows():
            folium.Marker(
                location=[row["NU_LATITUDE"], row["NU_LONGITUDE"]],
                icon=folium.Icon(color="red", icon="signal", prefix="fa"),
                popup=f"{row.get('NO_FANTASIA', '')} (CNES {row['CO_CNES']})",
            ).add_to(layer_ubs_s)
            folium.Circle(
                location=[row["NU_LATITUDE"], row["NU_LONGITUDE"]],
                radius=params["candidate_radius_km"] * 1000,
                color="#c0392b",
                weight=1.5,
                fill=True,
                fill_color="#e74c3c",
                fill_opacity=0.1,
            ).add_to(layer_ubs_s)
        layer_ubs_s.add_to(m)

    folium.LayerControl(collapsed=False).add_to(m)
    return m


# ---------------------------------------------------------------------------
# Estado de sessão
# ---------------------------------------------------------------------------

if "scenarios" not in st.session_state:
    st.session_state.scenarios = []  # list of dicts: {nome, params, resultado, metricas}
if "ultimo_resultado" not in st.session_state:
    st.session_state.ultimo_resultado = None
if "ultimas_metricas" not in st.session_state:
    st.session_state.ultimas_metricas = None
if "sensibilidade" not in st.session_state:
    st.session_state.sensibilidade = None
if "ultimo_csv" not in st.session_state:
    st.session_state.ultimo_csv = None  # (bytes, nome_cenario) ou None


# ---------------------------------------------------------------------------
# Layout
# ---------------------------------------------------------------------------

st.title("MLCP — Otimização de Sensores de Qualidade do Ar em São Paulo")
st.caption(
    "Modelo de cobertura máxima ponderada. Decide ONDE instalar P sensores entre "
    "UBSs candidatas, maximizando cobertura de setores censitários por população, "
    "saúde, vulnerabilidade social (IPVS) e exposição."
)

dados = carregar_dados()

tab1, tab2, tab3 = st.tabs(["Cenário Atual", "Comparação de Cenários", "Análise de Sensibilidade"])


# =============================================================================
# Aba 1 — Cenário Atual
# =============================================================================
with tab1:
    col_param, col_mapa, col_metric = st.columns([1, 3, 1])

    with col_param:
        st.subheader("Parâmetros")
        P = st.slider("P (nº sensores)", 1, 100, 10)
        alpha = st.slider("α — peso população", 0.0, 2.0, 1.0, 0.05)
        beta = st.slider("β — peso saúde", 0.0, 2.0, 1.0, 0.05)
        gamma = st.slider("γ — peso IPVS", 0.0, 2.0, 1.0, 0.05)
        delta = st.slider("δ — peso exposição", 0.0, 2.0, 0.0, 0.05)
        cand_r = st.slider("Raio candidato (km)", 0.5, 5.0, 2.0, 0.1)
        exist_r = st.slider("Raio existentes (km)", 0.5, 5.0, 3.0, 0.1)

        params_atuais = {
            "P": P, "alpha": alpha, "beta": beta, "gamma": gamma,
            "delta": delta, "candidate_radius_km": cand_r,
            "existing_radius_km": exist_r,
        }

        if st.button("Resolver", type="primary", width="stretch"):
            with st.spinner("Resolvendo MLCP via CBC..."):
                try:
                    res = resolver(params_atuais)
                    st.session_state.ultimo_resultado = res
                    st.session_state.ultimas_metricas = calcular_metricas(res, dados)
                except Exception as e:
                    st.error(f"Falha ao resolver: {e}")
                    st.session_state.ultimo_resultado = None

        res = st.session_state.ultimo_resultado
        if res is not None:
            if res["status"] == "Optimal":
                st.success(f"Status: **{res['status']}**")
            elif res["status"] == "Infeasible":
                st.error(f"Status: **{res['status']}**")
            else:
                st.warning(f"Status: **{res['status']}**")
            st.metric("Tempo de execução", f"{res['tempo_s']:.2f} s")

        nome_cenario = st.text_input("Nome do cenário", value=f"Cenário {len(st.session_state.scenarios)+1}")
        if st.button("Salvar este cenário", width="stretch", disabled=res is None):
            if res and st.session_state.ultimas_metricas:
                st.session_state.scenarios.append({
                    "nome": nome_cenario,
                    "params": deepcopy(res["params"]),
                    "resultado": deepcopy(res),
                    "metricas": deepcopy(st.session_state.ultimas_metricas),
                })
                st.success(f"Salvo: {nome_cenario}")

                # ---- Gerar CSV com UBSs escolhidas ----
                instalados_set = set(res["instalados"])
                ubs_df = dados["ubs"]
                escolhidas = ubs_df[ubs_df["CO_CNES"].isin(instalados_set)].copy()
                csv_cols = [c for c in ["CO_CNES", "NO_FANTASIA", "NU_LATITUDE", "NU_LONGITUDE"] if c in escolhidas.columns]
                csv_bytes = escolhidas[csv_cols].to_csv(index=False).encode("utf-8")
                st.session_state["ultimo_csv"] = (csv_bytes, nome_cenario)

        # Botão de download persiste entre rerenders
        if st.session_state.ultimo_csv is not None:
            csv_bytes, csv_nome = st.session_state.ultimo_csv
            safe_nome = csv_nome.replace(" ", "_")
            st.download_button(
                label="⬇️ Baixar CSV das UBSs escolhidas",
                data=csv_bytes,
                file_name=f"{safe_nome}_ubs_escolhidas.csv",
                mime="text/csv",
                width=True,
            )

    with col_mapa:
        st.subheader("Mapa")
        # Usa params_atuais p/ render mesmo antes de resolver (apenas pesos visuais)
        mapa = construir_mapa(
            st.session_state.ultimo_resultado,
            dados,
            st.session_state.ultimo_resultado["params"] if st.session_state.ultimo_resultado else params_atuais,
        )
        st_folium(mapa, height=650, width="stretch", returned_objects=[])

    with col_metric:
        st.subheader("Métricas")
        m = st.session_state.ultimas_metricas
        if m is None:
            st.info("Configure parâmetros e clique em **Resolver**.")
        else:
            st.metric("Função objetivo", f"{m['objetivo']:.0f}" if m["objetivo"] is not None else "—")
            st.metric("Setores cobertos", f"{m['n_cobertos']} / {m['n_total']}")
            st.metric("População coberta", f"{m['pop_coberta_pct']:.1f}%")
            st.metric("Cobertura ponderada", f"{m['cobertura_ponderada_pct']:.1f}%")
            st.metric("IPVS médio (cobertos)", f"{m['ipvs_medio_cob']:.2f}")
            st.metric("IPVS médio (não-cobertos)", f"{m['ipvs_medio_ncob']:.2f}")

            st.markdown("**Decomposição do objetivo**")
            df_contrib = pd.DataFrame(
                {"Termo": list(m["contrib"].keys()), "Contribuição": list(m["contrib"].values())}
            )
            st.dataframe(df_contrib, hide_index=True, width="stretch")


# =============================================================================
# Aba 2 — Comparação de Cenários
# =============================================================================
with tab2:
    st.subheader("Cenários Salvos")
    if not st.session_state.scenarios:
        st.info("Nenhum cenário salvo. Vá em **Cenário Atual**, resolva e salve.")
    else:
        # listagem com renomear / deletar
        for i, cen in enumerate(st.session_state.scenarios):
            c1, c2, c3 = st.columns([4, 6, 1])
            with c1:
                novo_nome = st.text_input(
                    "nome", value=cen["nome"], key=f"nome_{i}", label_visibility="collapsed"
                )
                st.session_state.scenarios[i]["nome"] = novo_nome
            with c2:
                p = cen["params"]
                st.caption(
                    f"P={p['P']} | α={p['alpha']:.2f} β={p['beta']:.2f} "
                    f"γ={p['gamma']:.2f} δ={p['delta']:.2f} | "
                    f"r_cand={p['candidate_radius_km']:.1f} r_exist={p['existing_radius_km']:.1f} | "
                    f"obj={cen['metricas']['objetivo']:.0f}"
                )
            with c3:
                if st.button("✕", key=f"del_{i}"):
                    st.session_state.scenarios.pop(i)
                    st.rerun()

        st.divider()

        if len(st.session_state.scenarios) >= 2:
            nomes = [c["nome"] for c in st.session_state.scenarios]
            csel1, csel2 = st.columns(2)
            with csel1:
                sel_a = st.selectbox("Cenário A", nomes, index=0, key="cmp_a")
            with csel2:
                sel_b = st.selectbox("Cenário B", nomes, index=1, key="cmp_b")

            cen_a = next(c for c in st.session_state.scenarios if c["nome"] == sel_a)
            cen_b = next(c for c in st.session_state.scenarios if c["nome"] == sel_b)

            # mapas lado a lado
            cmap1, cmap2 = st.columns(2)
            with cmap1:
                st.markdown(f"**{cen_a['nome']}**")
                m_a = construir_mapa(cen_a["resultado"], dados, cen_a["params"])
                st_folium(m_a, height=500, width="stretch", returned_objects=[], key="map_a")
            with cmap2:
                st.markdown(f"**{cen_b['nome']}**")
                m_b = construir_mapa(cen_b["resultado"], dados, cen_b["params"])
                st_folium(m_b, height=500, width="stretch", returned_objects=[], key="map_b")

            # tabela comparativa
            def linha_metricas(cen):
                m = cen["metricas"]
                return {
                    "Objetivo": m["objetivo"],
                    "Setores cobertos": m["n_cobertos"],
                    "% pop. coberta": m["pop_coberta_pct"],
                    "% cob. ponderada": m["cobertura_ponderada_pct"],
                    "IPVS méd. cob.": m["ipvs_medio_cob"],
                    "IPVS méd. n-cob.": m["ipvs_medio_ncob"],
                    **{f"contrib · {k}": v for k, v in m["contrib"].items()},
                }

            df_cmp = pd.DataFrame({
                cen_a["nome"]: linha_metricas(cen_a),
                cen_b["nome"]: linha_metricas(cen_b),
            })
            df_cmp["Δ (B - A)"] = df_cmp[cen_b["nome"]] - df_cmp[cen_a["nome"]]
            st.dataframe(df_cmp.style.format("{:.2f}"), width="stretch")

            # bar chart das contribuições
            df_contrib_cmp = pd.DataFrame({
                cen_a["nome"]: cen_a["metricas"]["contrib"],
                cen_b["nome"]: cen_b["metricas"]["contrib"],
            })
            st.markdown("**Contribuição de cada termo ao objetivo**")
            st.bar_chart(df_contrib_cmp)
        else:
            st.info("Salve pelo menos 2 cenários para comparar.")


