# Dados do projeto — guia de reprodução

Este documento explica **como rodar o projeto na sua máquina** e **como regenerar
os dados** do zero. Para as *decisões* de tratamento e limitações de cada variável,
veja [`considerações.md`](considerações.md).

---

## TL;DR — só quero rodar o projeto

Os artefatos processados (`data/processed_data/*.parquet`) **já vêm versionados** no
repositório. Então, para usar o app e os algoritmos, você **não precisa baixar nenhum
dado bruto nem regenerar nada**:

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py            # interface
# ou rode os algoritmos via experiments/ e evaluation/
```

Só precisa mexer no resto deste guia se quiser **regenerar** exposição/saúde a partir
das fontes originais.

---

## Os dois ambientes (por quê)

O projeto usa **dois virtualenvs** por causa de um conflito de versão do `pandas`:

| venv | usado para | requirements |
|------|-----------|--------------|
| `.venv` | pipeline principal, app, algoritmos, exposição (OSM), figuras | `requirements.txt` (pandas 3.x, geopandas, osmnx, pulp, streamlit) |
| `.venv-datasus` | **só** baixar/decodificar o SIH-RD do DATASUS | `requirements-datasus.txt` (pysus, pyreaddbc, dbfread, pandas 2.x) |

```bash
python -m venv .venv-datasus && source .venv-datasus/bin/activate
pip install -r requirements-datasus.txt
```

O `.venv-datasus` só é necessário na etapa 1 da regeneração da saúde (baixar o `.dbc`).
Todo o resto roda no `.venv`.

---

## Estrutura de `data/`

```
data/
├── README.md                ← este guia
├── considerações.md         ← decisões de tratamento e limitações (o "porquê")
├── integrate_parquets.py    ← junta os parquets em integrated.parquet (pipeline oficial)
├── scripts/                 ← VERSIONADO: scripts que regeneram os dados (ver abaixo)
├── processed_data/          ← VERSIONADO: artefatos prontos (parquets pequenos)
├── raw_data/                ← NÃO versionado (~4,5 GB): brutos, baixar à parte
└── docs/figuras/            ← VERSIONADO: figuras de decisão (PNG) usadas na doc
```

> `data/tests/` (scratch: notebooks didáticos, auditorias, geração das figuras) **não é
> versionado** — é área de prototipagem local.

### O que tem em `processed_data/` (versionado)

| arquivo | o que é | gerado por |
|---------|---------|-----------|
| `ipvs.parquet` | setores + IPVS + população + geometria + distrito | `process_data.ipynb` |
| `ubs.parquet` | UBS candidatas (CNES) | `process_data.ipynb` |
| `cetesb.parquet` | estações CETESB existentes | `process_data.ipynb` |
| `exposicao.parquet` | densidade viária + dist. à via principal, por setor | `scripts/gerar_exposicao.py` |
| `cep_to_distrito.parquet` | lookup CEP → distrito (CNEFE) | `scripts/gerar_saude.py` (subproduto, cacheado) |
| `saude.parquet` | internações resp. / 1000 hab, por distrito → setor | `scripts/gerar_saude.py` |
| `integrated.parquet` | **tudo junto**, consumido pelos algoritmos | `integrate_parquets.py` |

---

## Fontes dos dados brutos (para regenerar do zero)

Coloque cada arquivo em `data/raw_data/` conforme indicado. **Nenhum vai pro git**
(são grandes / redistribuição restrita).

| fonte | conteúdo | onde obter |
|-------|----------|-----------|
| **SIH-RD** (DATASUS) | internações hospitalares (AIH Reduzida) | FTP `ftp.datasus.gov.br` → `/dissemin/publicos/SIHSUS/200801_/Dados/RD SP AAMM.dbc` — **baixado automaticamente** por `scripts/gerar_saude.py` (cacheia em `data/raw_data/sih/`) |
| **CNEFE 2022** (IBGE) | endereços de SP com CEP + distrito | Portal IBGE, Censo 2022 → CNEFE, arquivo `3550308_SAO_PAULO.zip` → `data/raw_data/cnefe/` |
| **OSM** (OpenStreetMap) | malha viária principal | **baixado automaticamente** por `scripts/gerar_exposicao.py` via `osmnx` |
| **IPVS 2022** (SEADE) | vulnerabilidade social | Geoportal da Fundação SEADE (`.gpkg`) |
| **Censo 2022 setores** (IBGE) | malha de setores + população | Portal IBGE, malhas territoriais 2022 |
| **CNES** (DATASUS) | estabelecimentos de saúde | Base CNES do DATASUS |
| **CETESB** | estações de qualidade do ar | `postos-de-cetesb.xlsx` (planilha CETESB) |

---

Cada variável tem **um único script** que faz tudo (coleta + cálculo + salvar):

## Regenerar a EXPOSIÇÃO (roda no `.venv`)

```bash
source .venv/bin/activate
python data/scripts/gerar_exposicao.py       # baixa a malha viária (cacheia) e salva exposicao.parquet
```

## Regenerar a SAÚDE (roda no `.venv-datasus`)

```bash
source .venv-datasus/bin/activate
python data/scripts/gerar_saude.py --ano 2022   # baixa SIH-RD + lookup CNEFE + taxa → saude.parquet
```

O `gerar_saude.py` roda inteiro num venv só: baixa os 12 meses do SIH-RD (cacheia em
`data/raw_data/sih/`), constrói/reusa o lookup CEP→distrito (`cep_to_distrito.parquet`)
e agrega a taxa por distrito → setor.

## Reintegrar tudo em `integrated.parquet` (roda no `.venv`)

```bash
source .venv/bin/activate
python data/integrate_parquets.py
```

---

## Scripts em `scripts/` (versionados)

**Um script por variável**, cada um autocontido. Convenção do projeto: prototipar e
validar em `data/tests/` (scratch, não versionado) antes de consolidar aqui e integrar no
pipeline oficial (`integrate_parquets.py`, `process_data.ipynb`).

| script | papel | venv |
|--------|-------|------|
| `gerar_exposicao.py` | malha viária OSM → densidade/distância por setor → `exposicao.parquet` | `.venv` |
| `gerar_saude.py` | SIH-RD + CNEFE → taxa de internação por distrito → `saude.parquet` (+ `cep_to_distrito.parquet`) | `.venv-datasus` |

As **figuras** em `docs/figuras/` já vêm versionadas prontas. O helper que as gera
(`data/tests/gerar_figuras_doc.py`) é scratch local, não versionado.
