# Otimização de Sensores de Qualidade do Ar

Sistema geoespacial para otimizar a instalação de sensores de qualidade do ar em Unidades Básicas de Saúde (UBSs) no município de São Paulo. Usa diferentes modelos para selecionar *P* UBSs dentre ~400 que maximizem a cobertura populacional ponderada por vulnerabilidade social (IPVS), indicadores de saúde e exposição à poluição.

Métodos implementados: metaheurísticas (GRASP, Simulated Annealing, Tabu Search, Genetic Algorithm, NSGA-II, PSO), multicritério (WLC, AHP, TOPSIS) e modelos de localização clássicos (MCLP, p-mediana, p-centro).

---

## Índice

- [Setup](#setup)
- [Pré-processamento dos dados](#pré-processamento-dados)
- [Aplicação interativa (Streamlit)](#streamlit)
- [Benchmark experimental](#benchmark)
- [Figuras](#figuras)
- [Validação cruzada espacial](#spatial-cv)
- [Métodos disponíveis](#métodos)
- [Arquivo de configuração](#config)
- [Estrutura do projeto](#estrutura)

---

## Setup

```bash
pip install -r requirements.txt
```

---

## Pré-processamento dos dados

Os dados brutos estão em `data/raw_data/`. O pipeline de processamento tem duas etapas:

**1. Notebook de processamento**

```bash
jupyter notebook data/process_data.ipynb
```

Execute todas as células. Isso gera os parquets intermediários em `data/processed_data/`.

**2. Integração dos parquets**

```bash
python data/integrate_parquets.py
```

Junta os dados de setores censitários, UBSs e estações CETESB em um único arquivo `data/processed_data/integrated.parquet`.

> A função principal (`integrate_parquets()`) calcula a distância haversine de cada setor censitário à UBS e CETESB mais próxima, criando colunas como `ubs_cobertura_2km` e `cetesb_cobertura_3km`.

---

## Aplicação interativa (Streamlit)

```bash
streamlit run app.py
```

Interface com três abas:

- **Cenários**: configura pesos (população, saúde, IPVS, exposição), raio de cobertura e *P* sensores; executa o solver e visualiza resultados em mapa Folium.
- **Comparação**: compara cenários salvos lado a lado.
- **Análise de sensibilidade**: explora como variações nos parâmetros afetam a função-objetivo.

---

## Benchmark

> **Entrypoint:** `experiments/run_comparison.py`
>
> **Config padrão:** `experiments/configs/default.yaml`

Executa todos os métodos × valores de *P* (× sementes, para métodos estocásticos) em paralelo, persiste resultados em parquet e gera figuras.

### Flags

| Flag | Tipo | Padrão | Descrição |
|------|------|--------|-----------|
| `--config` | `str` | `experiments/configs/default.yaml` | Caminho para o arquivo YAML de configuração |
| `--p` | `int n+` | *(do config)* | Sobrescreve a lista de valores de *P* (ex: `--p 5 10 20`) |
| `--methods` | `str n+` | *(do config)* | Sobrescreve a lista de métodos (ex: `--methods mclp greedy`) |
| `--radius` | `float` | *(do config)* | Sobrescreve o raio de cobertura em km (ex: `--radius 3.0`) |
| `--figures-only` | flag | — | Pula o benchmark; só regera figuras de um runs.parquet existente (requer `--runs`) |
| `--runs` | `str` | — | Caminho para um `runs.parquet` existente (usado com `--figures-only`) |
| `--no-figures` | flag | — | Pula a geração de figuras após o benchmark |
| `--spatial-cv` | flag | — | Executa validação cruzada leave-one-district-out |

### Exemplos de uso

```bash
# Benchmark completo com config padrão
python experiments/run_comparison.py

# Benchmark só com MCLP e greedy, P = 10 e 20, raio 3 km
python experiments/run_comparison.py --methods mclp greedy --p 10 20 --radius 3.0

# Regenerar figuras de um resultado existente
python experiments/run_comparison.py --figures-only --runs experiments/results/20250702_123456/runs.parquet

# Benchmark sem figuras
python experiments/run_comparison.py --no-figures

# Benchmark com validação cruzada espacial
python experiments/run_comparison.py --spatial-cv
```

### Fluxo interno (`evaluation/benchmark.py`)

1. Expande o grid `(método, P, seed)` a partir do config
2. Cada combinação roda em um subprocesso (crash isolation)
3. Métodos determinísticos rodam com 1 seed; estocásticos rodam com todas as seeds
4. Resultados: `experiments/results/YYYYMMDD_HHMMSS/runs.parquet` + `meta.json`

---

## Figuras

As figuras são geradas automaticamente ao final do benchmark (a menos que `--no-figures` seja passado). Para gerar apenas as figuras de um resultado existente:

```bash
python experiments/run_comparison.py --figures-only --runs experiments/results/YYYYMMDD_HHMMSS/runs.parquet
```

Ou importando o módulo diretamente:

```python
from experiments.figures import generate_all
generate_all("experiments/results/YYYYMMDD_HHMMSS/runs.parquet", "experiments/results/YYYYMMDD_HHMMSS/figures")
```

---

## Validação cruzada espacial

Avalia a generalização dos modelos deixando um distrito de fora por vez (*leave-one-district-out*):

```bash
# Via run_comparison.py
python experiments/run_comparison.py --spatial-cv

# Ou importando diretamente
python -c "
from evaluation.benchmark import _ALL_SOLVERS
from evaluation.data_loader import load
from evaluation.spatial_cv import run_spatial_cv

data = load(radius_km=2.0)
solver = _ALL_SOLVERS['mclp']
cv_df = run_spatial_cv(solver, data, p=10, timeout_s=60)
print(cv_df)
"
```

Resultado salvo em `experiments/results/YYYYMMDD_HHMMSS/spatial_cv.parquet`.

---

## Métodos disponíveis

| Nome (flag) | Classe | Tipo | Determinístico |
|-------------|--------|------|:---:|
| `mclp` | `MCLPSolver` | PLI (CBC) | ✅ |
| `p_median` | `PMedianSolver` | PLI (CBC) | ✅ |
| `p_center` | `PCenterSolver` | PLI (CBC) | ✅ |
| `greedy` | `GreedyCoverageSolver` | Guloso | ✅ |
| `cetesb_only` | `CetEsbOnlySolver` | Baseline | ✅ |
| `wlc` | `WLCSolver` | Multicritério | ✅ |
| `ahp` | `AHPSolver` | Multicritério | ✅ |
| `topsis` | `TOPSISSolver` | Multicritério | ✅ |
| `tabu_search` | `TabuSearchSolver` | Metaheurística | ✅ |
| `random` | `RandomSolver` | Baseline | ❌ |
| `grasp` | `GRASPSolver` | Metaheurística | ❌ |
| `simulated_annealing` | `SimulatedAnnealingSolver` | Metaheurística | ❌ |
| `genetic_algorithm` | `GeneticAlgorithmSolver` | Metaheurística | ❌ |
| `nsga2` | `NSGA2Solver` | Metaheurística multi-objetivo | ❌ |
| `pso` | `PSOSolver` | Metaheurística (enxame) | ❌ |

---

## Configuração

> **Arquivo:** `experiments/configs/default.yaml`

```yaml
p_values: [5, 10, 20, 50, 100]      # Valores de P a testar
radius_km: 2.0                       # Raio de cobertura em km
weights:
  alpha: 1.0   # população
  beta: 0.0    # saúde
  gamma: 1.0   # IPVS
  delta: 0.0   # exposição
methods:
  - mclp
  - p_median
  - p_center
  - wlc
  - topsis
  - ahp
  - greedy
  - random
  - cetesb_only
  - grasp
  - simulated_annealing
  - tabu_search
  - genetic_algorithm
  - nsga2
  - pso
random_seeds: [42, 43, 44, 45, 46]  # Sementes para métodos estocásticos
timeout_s: 60                        # Timeout do solver MCLP (segundos)
run_spatial_cv: false                # Habilita validação cruzada espacial
max_workers: null                    # Núcleos (null = todos disponíveis)
```

---

## Estrutura do projeto

```
.
├── app.py                                 # Streamlit UI
├── algorithms/                            # Solvers
│   ├── base.py                            #  ProblemData, Solver, SolveResult
│   ├── maximize_coverage_mlcp.py          #  MLCP original (app)
│   ├── mclp.py                            #  MLCP (benchmark)
│   ├── baselines.py                       #  Greedy, Random, CETESB-only
│   ├── p_median.py / p_center.py          #  Modelos clássicos de localização
│   ├── metaheuristics.py                  #  GRASP, SA, TS, GA, NSGA-II, PSO
│   └── multicriteria.py                   #  WLC, AHP, TOPSIS
├── data/
│   ├── process_data.ipynb                 #  Pipeline de processamento
│   ├── integrate_parquets.py              #  Junção dos parquets processados
│   ├── raw_data/                          #  Fontes originais
│   └── processed_data/                    #  Parquets processados + integrados
├── evaluation/
│   ├── benchmark.py                       #  Orquestrador do benchmark
│   ├── data_loader.py                     #  Carrega parquets → ProblemData
│   ├── distance_cache.py                  #  Cache de matrizes de distância
│   ├── metrics.py                         #  Métricas de avaliação
│   ├── pareto.py                          #  Análise Pareto
│   └── spatial_cv.py                      #  Validação cruzada espacial
├── experiments/
│   ├── run_comparison.py                  #  CLI principal
│   ├── configs/default.yaml               #  Configuração padrão
│   ├── figures.py                         #  Geração de figuras
│   └── results/                           #  Resultados dos benchmarks
├── requirements.txt
└── README.md
```
