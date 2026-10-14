# Revisiones: proyectos y papers

Registro de qué se revisó realmente, con qué método se verificó y qué commit se
inspeccionó. Existe para que ninguna afirmación del proyecto descanse en memoria
o en el resumen de un tercero.

**Método de verificación.** Los conteos de stars, licencias y fechas de push se
consultaron contra la **API de GitHub**; los papers contra la **API de arXiv**
parseando el XML. Un primer intento de verificar los IDs de arXiv con `sed` dio
falsos negativos en todos ellos — el parseo con `xml.etree` es el método
correcto y es el que respaldó estos datos.

Fecha de verificación: **2026-10-13**.

---

## 1. Repositorios clonados e inspeccionados en código

Los únicos tres sobre los que se leyó código fuente. Clonados con
`git clone --depth 1`, viven en `reference/` y están gitignorados.

| Repo | Commit inspeccionado | Fecha commit | Rol en el diseño |
|---|---|---|---|
| `AI4Finance-Foundation/FinRobot` | `2717499` | 2026-09-28 | **Mejor referencia de provenance** y de contract gate. Es de donde sale el patrón "numbers are code-calculated, narratives are LLM-assisted" |
| `galleonlabs/hypergrok-trading-desk` | `d59364a` | 2026-10-13 | Modelo de **gobernanza**. Documentación y prompts, no software |
| `HKUDS/AI-Trader` | `d03ff6c` | 2026-06-11 | **No usar como referencia de código** — ver §1.1 |
| `AI4Finance-Foundation/FinRL` | `8511e7c` | 2026-10-14 | Infraestructura de RL seria y bien licenciada. **Su capa de evaluación no sostiene un claim de despliegue** — ver §6 (análisis en cascada) |

### 1.1 Auditoría de HKUDS/AI-Trader: el repo no es el paper

Hallazgo propio, no reportado por la investigación previa. Inspección del clon
local: 29 604 LOC de Python en 74 archivos.

**Trampa de licencia.** El README muestra un badge `License-MIT` **estático y
hardcodeado** que enlaza a `LICENSE`, pero no existe ningún archivo de licencia
en el árbol git y la API de GitHub reporta `license: null`. Sin licencia el
default legal es *all rights reserved*. **Bloqueante** para construir sobre él.

**La capa de evaluación del paper no está en el repo.** Grep sobre los 74
archivos Python: `backtest` → 0 archivos, `walk-forward` → 0, `deflated` → 0,
`point-in-time` → 0, `survivorship` → 0, `slippage` → 0, `transaction_cost` → 0,
`pbo` → 0. Lo que el paper describe como benchmark riguroso con anti-contaminación
temporal no es el código publicado.

**Veredicto: usar el paper (arXiv:2512.10971), no el repositorio.**

---

## 2. Repositorios verificados por API (sin clonar)

Stars, licencia y último push consultados contra la API de GitHub. **No se
inspeccionó su código**, por lo que no hay commit registrado.

### Agentes financieros

| Repo | Stars | Licencia | Últ. push | Veredicto |
|---|---|---|---|---|
| `TauricResearch/TradingAgents` | 110 986 | Apache-2.0 | 2026-10-03 | Referencia del patrón de debate. **No modela costes** (explícito en su código) |
| `AI4Finance-Foundation/FinGPT` | 21 399 | MIT | 2026-09-23 | Componente de análisis, no sistema |
| `Paparusi/Clawtrade` | 37 | MIT | 2026-03-14 | Pre-maduro. Promete 7 exchanges, 14 tools, 5 subagentes, vault AES-256 y GraphQL en 137 commits. Catálogo de integración, no arquitectura |

**Desambiguación necesaria:** `Paparusi/Clawtrade` (Go, 37 stars) **≠**
`TianYuFan0504/ClawTrader` (linaje de rebrand de AI-Trader). Nombres casi
idénticos, proyectos distintos.

### Quant / backtesting / ejecución

| Repo | Stars | Licencia | Nota |
|---|---|---|---|
| `freqtrade/freqtrade` | 55 350 | GPL-3.0 | Cripto, maduro, FreqAI |
| `microsoft/qlib` | 49 346 | MIT | Plataforma ML cuant, foco A-shares |
| `nautechsystems/nautilus_trader` | 29 927 | LGPL-3.0 | Core event-driven en Rust. **Backtest y live comparten el mismo código** |
| `QuantConnect/Lean` | 21 984 | Apache-2.0 | Motor C#/Python |
| `hummingbot/hummingbot` | 20 432 | Apache-2.0 | Capa de ejecución cripto |
| `AI4Finance-Foundation/FinRL` | — | — | **Promovido a §1**: clonado e inspeccionado en código. Ver §6 |
| `tensortrade-org/tensortrade` | 7 196 | Apache-2.0 | RL, mantenimiento lento |
| `AI4Finance-Foundation/FinRL-Meta` | 1 949 | MIT | Entornos y datos para RL |

---

## 3. Papers leídos (27 IDs verificados contra la API de arXiv)

### 3.1 Núcleo de evaluación — los que determinaron la arquitectura

| ID | Título | Aporte |
|---|---|---|
| **2608.27734** | *What survives honest evaluation? Leakage-safe, search-aware assessment* (Gençay, ago-2026) | **El más influyente del proyecto.** Un oráculo con leakage **pasa limpiamente Deflated Sharpe y PBO**. De aquí viene la tesis de que el anti-leakage debe ser estructural, no estadístico, y que el arnés se construye antes que cualquier agente |
| **2605.16895** | *The Alpha Illusion* (Ye et al., Fudan + Imperial, may-2026) | Position paper. Audita 6 sistemas. Protocolos **P1–P6** escalonados por fuerza del claim |
| **2605.19337** | *Agentic Trading: When LLM Agents Meet Financial Markets* (Xia et al., Shenzhen Univ., v2 oct-2026) | Survey. Marco **A-C-A**. **185 estudios, subconjunto empírico de 80** |
| **2510.02209** | *StockBench* (Chen et al., Tsinghua) | Benchmark secuencial. El baseline pasivo queda 12º/14 en retorno pero gana en mercado bajista |
| **2512.10971** | *AI-Trader* (Fan et al., HKUDS) | Benchmark live post-cutoff, 3 mercados, 5 tools MCP |
| **2510.11695** | *When Agents Trade / Agent Market Arena* (Qian et al., 17 autores, incl. Lopez-Lira) | **La arquitectura del agente domina la variación; el backbone importa menos** |

> Corrección a la investigación previa: el survey **2605.19337** reporta
> **185 estudios / 80 empíricos**, no "77/19". El claim defendible es
> **R3 = 0, y solo 2 de 80 alcanzan R2**.

### 3.2 Arquitecturas de agentes

| ID | Título | Patrón estudiado |
|---|---|---|
| 2412.20138 | TradingAgents (UCLA + MIT + Tauric) | Debate bull/bear con árbitro, risk team, LangGraph |
| 2407.06567 | FinCon (NeurIPS 2024) | **Conceptual Verbal Reinforcement**, jerarquía manager-analyst, control CVaR |
| 2311.13743 | FinMem (Stevens) | Memoria por capas (shallow/intermediate/deep) + perfil de carácter |
| 2402.18485 | FinAgent (NTU) | Multimodal (precios + noticias + velas), reflexión en dos niveles |
| 2405.14767 | FinRobot plataforma (AI4Finance) | Arquitectura de 4 capas, Financial CoT, Smart Scheduler |
| 2411.08804 | FinRobot equity research (ICAIF 2024) | 3 agentes: Data / Concept / Thesis |
| 2407.18957 | StockAgent | Simulación ABM de mercado |
| 2405.16334 | *Devil's Advocate* (Wang/Li/Deng/Roth/Li, EMNLP 2024 Findings) | Reflexión anticipatoria, 3 intervenciones introspectivas. WebArena 23.5% success (+3.5pp), −45% trials |

> La investigación previa citaba *Devil's Advocate* como "2024, sin ID". Es
> **arXiv:2405.16334**.

### 3.3 Evidencia crítica y metodología

| ID | Autores | Aporte |
|---|---|---|
| **2309.17322** | Glasserman & Lin | *Assessing Look-Ahead Bias in Stock Return Predictions from LLMs* |
| **2304.07619** | Lopez-Lira & Tang | *Can ChatGPT Forecast Stock Price Movements?* |
| **2502.21206** | Sarkar & Vafa | *Chronologically Consistent LLMs* |
| **2407.17866** | Kim, Muhn & Nikolaev (Chicago Booth) | *Financial Statement Analysis with LLMs* |

### 3.4 Alpha mining, RL y surveys

`2308.00016` Alpha-GPT · `2502.16789` AlphaAgent (alpha decay) ·
`2011.09607` + `2111.09395` FinRL · `2112.06753` + `2304.13174` FinRL-Meta ·
`2408.06361` *LLM Agent in Financial Trading: A Survey* · `2306.06031` FinGPT

### 3.5 Referencias sin ID de arXiv

- **Bailey & López de Prado** — *Probability of Backtest Overfitting* y
  *Deflated Sharpe Ratio*. Base de los gates implementados en `domain/stats/`.
- **Harvey, Liu & Zhu 2016** (NBER w20592) — el umbral es **t > 3**, no 2.
- **Hou, Xue & Zhang 2020** — de 447 anomalías replicadas, la mayoría no sobrevive.
- **The Evolution of RL in Quantitative Finance** (ACM Computing Surveys 2025,
  167 publicaciones).

---

## 4. Omisiones de la investigación previa

Dos papers ausentes que cambiaron el diseño del proyecto, y conviene registrar
por qué importaron:

1. **arXiv:2605.16895 (The Alpha Illusion)**
2. **arXiv:2608.27734 (leakage-safe evaluation)**

Ambos reencuadran al "agente escéptico" de una **persona LLM** a un conjunto de
**gates deterministas**. Es la razón de que `domain/skeptic/` sean 8 criterios en
código y no un prompt, y de que el arnés se construyera antes que los agentes.

---

## 5. Qué se tomó de cada arquitectura

| Fuente | Patrón adoptado | Dónde vive en este repo |
|---|---|---|
| FinRobot | Provenance de cada número + contract gate en la frontera de persistencia | `domain/compute/`, `domain/contract/`, `artifact/` |
| TradingAgents | Debate bull/bear con árbitro | `domain/debate/` |
| HyperGrok | Gobernanza: reglas explícitas y auditables | `domain/skeptic/`, `domain/schedule/` |
| Bailey & López de Prado | DSR y PBO con trial ledger append-only | `domain/stats/`, `domain/ledger/` |
| Gençay (2608.27734) | Anti-leakage **estructural** | `domain/pit/`, `domain/backtest/context.py` |

Lo que **no** se tomó: el modelo de costes de TradingAgents (no tiene) y
cualquier línea de código de AI-Trader (sin licencia).

---

## 6. Análisis en cascada: AI4Finance-Foundation/FinRL

Commit inspeccionado: **`8511e7c`** (2026-10-14). **21 052 LOC de Python en 99
archivos.** Licencia **MIT real** (archivo `LICENSE` presente en el árbol y
confirmado por la API — a diferencia de AI-Trader). 16 638 stars, 3 544 forks,
307 issues abiertos, push activo.

Análisis por capas, de metadata a código, terminando en una medición propia.

### L1 — Arquitectura

Separación de responsabilidades limpia y legible, en tres capas:

```
finrl/agents/       adaptadores a librerias de RL (stablebaselines3, elegantrl,
                    rllib, fasttd3, portfolio_optimization)
finrl/meta/         entornos Gym, procesadores de datos, preprocesadores,
                    paper trading
finrl/applications/ recetas ejecutables (stock_trading, crypto, HFT,
                    portfolio_allocation, imitation_learning)
```

Es **infraestructura de investigación en RL**, no un sistema de inversión. Juzgarlo
como lo segundo sería injusto; pero la pregunta relevante para este proyecto es si
su capa de evaluación sostiene las conclusiones que la gente extrae de él.

### L2 — El grep crítico (idéntico al aplicado a AI-Trader, para que sea comparable)

| Término | Archivos | | Término | Archivos |
|---|---|---|---|---|
| `backtest` | **9** | | `survivorship` | **0** |
| `turbulence` | **33** | | `slippage` | **0** |
| `transaction_cost` | 1 | | `deflated` | **0** |
| `walk_forward` | 0 ⚠️ | | `pbo` | **0** |
| `point_in_time` | **0** | | `look_ahead` / `lookahead` | **0** |

⚠️ **Corrección en favor de FinRL:** el grep de `walk_forward` da 0 porque lo
llaman *rolling window*. `finrl/applications/stock_trading/stock_trading_rolling_window.py`
implementa reentrenamiento rodante real (`rolling_window_length`,
`calc_train_trade_starts_ends_if_rolling`). **El walk-forward existe.** Mi primer
grep lo habría declarado ausente por una cuestión de nomenclatura, y habría sido
un error de auditoría.

A diferencia de AI-Trader (donde `backtest` daba 0), aquí **la capa de backtest
existe de verdad**. Lo que no existe es la capa de control de overfitting.

### L3 — Modelo de costes: lineal, sin impacto de mercado

```python
# finrl/meta/env_stock_trading/env_stocktrading.py
buy_cost_pct: list[float]  # default tipico 1e-3 = 10 bps
sell_cost_pct: list[float]
...
self.cost += price * shares * self.sell_cost_pct[index]
```

El coste es **estrictamente proporcional al nocional**. Consecuencias:

- **No hay impacto de mercado.** El coste por unidad no crece con el tamaño, lo que
  equivale a asumir **capacidad infinita**. Duplicar el capital no cuesta más por
  dólar. Es la aproximación que `domain/costs` de este proyecto existe para corregir,
  con impacto de raíz cuadrada.
- **No hay spread ni ADV.** Un nombre ilíquido cuesta lo mismo que uno líquido. Anoche
  medí lo que eso implica: sin filtro de liquidez, un universo muestreado
  mecánicamente produjo un equiponderado compuesto a **+430% anual** — bid-ask bounce
  sobre papeles de centavos, no retorno.
- No hay borrow para cortos ni comisión mínima.

### L4 — Universo: sesgo de supervivencia hardcodeado

El archivo **más grande del repo** es `finrl/config_tickers.py` (1 480 LOC, 1 414
líneas con tickers literales). Su contenido:

```python
# Dow 30 constituents in 2026/03/20
DOW_30_TICKER = ["AXP", "AMGN", "AMZN", "AAPL", "BA", ...]
```

Una lista de constituyentes **actuales**, fija, usada para backtests sobre periodos
históricos — incluida la aplicación de rolling window (línea 52:
`ticker_list=DOW_30_TICKER`). Es sesgo de supervivencia por construcción: las
empresas que salieron del índice no están.

Esto no es una objeción teórica. **Lo medí en este proyecto:** mis 48 tickers
elegidos de memoria batían a SPY en **+7.47% anual sin ninguna señal**. Al
reemplazarlos por un universo muestreado mecánicamente de la lista completa de la
SEC, el exceso desapareció (**−5.41%**, es decir por debajo del índice, que es la
forma esperada de los retornos accionarios). La ventaja de momentum que parecía
sólida (IR +0.97) se volvió **negativa** (−0.10). El sesgo del universo era toda
la señal.

### L5 — Crédito donde corresponde: la turbulencia **sí** es point-in-time

Su único control de riesgo está bien implementado en el eje temporal:

```python
# finrl/meta/preprocessor/preprocessors.py
hist_price = df_price_pivot[
    (df_price_pivot.index < unique_date[i])  # estrictamente anterior
    & (df_price_pivot.index >= unique_date[i - 252])  # ventana rodante de 1 año
]
cov_temp = filtered_hist_price.cov()
```

Ventana rodante estricta, covarianza solo con datos previos. **Sin look-ahead.** Es
mejor disciplina temporal que la que muchos repos con más estrellas exhiben.

Dos matices menores: el índice en `i` usa el retorno de `i`, así que operar al cierre
de `i` sería ejecución en la misma barra (el motor de este proyecto lo impide con un
retardo de una barra); y el `dropna(axis=1)` filtra silenciosamente los nombres con
huecos, aplicando un filtro de supervivencia por ventana.

### L6 — El hallazgo decisivo: búsqueda sobre Sharpe sin deflactar

```python
# finrl/agents/stablebaselines3/tune_sb3.py
import optuna

n_trials: int = 30  # 30 configuraciones de hiperparametros


def calculate_sharpe(self, df): ...  # el objetivo a maximizar


# "threshold: tolerance for increase in sharpe ratio"
```

FinRL **busca 30 configuraciones maximizando Sharpe** y reporta la mejor. Y el grep
de `deflated|bonferroni|multiple.test|family.wise|false.discovery|data.snoop`
devuelve **0 archivos**: no hay ninguna corrección por testeo múltiple.

Esto no es una crítica abstracta. Lo cuantifiqué con el arnés de este proyecto,
sobre **ruido gaussiano puro** (`mu = 0`, sin señal alguna), 756 días (~3 años, el
horizonte típico de sus ejemplos), eligiendo el mejor de 30 como hace `tune_sb3`:

```
Sharpe anualizado del GANADOR, puro ruido (400 repeticiones):
  mediana 1.141    p90 1.558    max 2.721
```

**Con 30 trials y cero señal, un Sharpe de ~1.14 es el resultado esperado.** No es
un caso extremo: es la mediana.

Una corrida concreta de ruido puro que se vería publicable, pasada por los gates:

```
Sharpe anualizado del ganador de 30 trials:  1.4456
  DSR          0.4215   (umbral 0.95)
  PBO          0.4819
  CI excluye 0 True        <- el CI por si solo NO protege
  CERTIFICA    False
```

El intervalo de confianza bootstrap **excluye el cero incluso para ruido puro** —
confirmación independiente de que un CI aislado no es un control de overfitting.
Solo el DSR con el conteo honesto de trials lo rechaza.

### Veredicto

| Dimensión | Estado |
|---|---|
| Licencia | **MIT real**, verificada en árbol y API. Usable |
| Ingeniería | Sólida. Capas limpias, 99 archivos, entornos Gym reutilizables |
| Backtest | **Existe**, con walk-forward rodante real |
| Disciplina temporal | **Parcialmente buena** (turbulencia PIT), sin garantías estructurales |
| Costes | Lineales. Sin impacto, spread, ADV ni borrow → capacidad infinita implícita |
| Universo | **Sesgo de supervivencia hardcodeado** en el archivo más grande del repo |
| Control de overfitting | **Ausente.** Busca 30 trials sobre Sharpe y no deflacta |

**Conclusión:** FinRL es buena infraestructura para *investigar* RL financiero y una
referencia legítima de entornos y adaptadores. **No es evidencia de que sus
estrategias funcionen**, y su propio tooling de tuning produce, por diseño, el
falso positivo que este proyecto demostró numéricamente. Encaja exactamente en el
diagnóstico de arXiv:2605.19337 (**R3 = 0, solo 2 de 80 estudios en R2**) y en el de
arXiv:2605.16895 (*The Alpha Illusion*).

### Qué tomar y qué no

| Tomar | Por qué |
|---|---|
| Separación agents / meta / applications | Permite cambiar de librería de RL sin tocar entornos |
| Turbulencia con ventana rodante estricta | Control de riesgo PIT correcto, portable |
| `stock_trading_rolling_window.py` | Esqueleto de reentrenamiento walk-forward |
| Procesadores de datos multi-fuente | Yahoo, Alpaca, WRDS, EODHD detrás de una interfaz común |

| No tomar | Por qué |
|---|---|
| `config_tickers.py` | Sesgo de supervivencia embotellado. Usar universo PIT con deslistados |
| `buy_cost_pct` / `sell_cost_pct` | Lineal. Sustituir por impacto de raíz cuadrada con ADV |
| `tune_sb3.py` tal cual | Requiere registrar los 30 trials en un ledger y deflactar el Sharpe |
| Cualquier Sharpe reportado | Sin deflactar ni PBO, no distingue señal de la mediana del ruido |

---

## 7. Cascada al ecosistema AI4Finance (39 repos no-fork)

| Repo | Stars | Licencia | Últ. push | Nota |
|---|---|---|---|---|
| FinGPT | 21 399 | MIT | 2026-09-23 | Jupyter. Componente de análisis |
| FinRL | 16 638 | MIT | 2026-10-13 | §6 |
| FinRobot | 8 212 | Apache-2.0 | 2026-09-28 | §1 |
| **ElegantRL** | 4 368 | **NOASSERTION** ⚠️ | 2026-02-20 | Ver §7.1 |
| FinRL-Trading | 3 815 | Apache-2.0 | 2026-09-18 | No inspeccionado |
| FinRL-Meta | 1 949 | MIT | 2026-10-12 | `slippage` en 8 archivos — más que FinRL |
| FinNLP | 1 492 | MIT | 2024-07-01 | Sin mantenimiento >2 años |
| **FinRL_Crypto** | 204 | MIT | 2025-12-15 | **El único con controles de overfitting.** Ver §7.2 |
| FinML | 181 | **NULL** | 2024-02-27 | Sin licencia |
| Quantifying-ESG-Alpha | 77 | **NULL** | 2021-01-12 | Sin licencia, abandonado |

### 7.1 ElegantRL: licencia contradictoria, y FinRL depende de ella

La API de GitHub reporta `NOASSERTION` / "Other". El motivo está en el propio
archivo `LICENSE`:

```
Copyright 2024 AI4Finance Foundation Inc.
All rights reserved.

Licensed under the Apache License, Version 2.0 (the "License");
```

**"All rights reserved" seguido de Apache-2.0 es contradictorio**, y por eso el
clasificador de GitHub no la resuelve. Probablemente la intención sea Apache-2.0
con una línea de copyright mal copiada, pero la ambigüedad es real — y
**`elegantrl` figura en el `requirements.txt` de FinRL**. Un repo MIT depende de
un paquete cuya licencia no se puede determinar automáticamente. Es un riesgo de
cadena de suministro legal, menor que el de AI-Trader (que no tiene licencia en
absoluto) pero digno de registro. `FinRL_Podracer` tiene el mismo problema.

### 7.2 FinRL_Crypto: los controles existen, pero aislados

Commit `495a428`, 40 archivos Python, MIT. El grep crítico, que daba casi todo
cero en FinRL, aquí da:

| Término | Archivos |
|---|---|
| `pbo` | **4** |
| `k-fold` | **4** |
| `purge` | **2** |
| `embargo` | **2** |
| `deflated` | **1** |
| `walk-forward` | **1** |

Contiene `function_CPCV.py` con `class CombPurgedKFoldCV` — **Combinatorial
Purged K-Fold Cross-Validation con embargo**, el toolkit de López de Prado
implementado de verdad, con su docstring explicando el purgado de muestras
solapadas y el periodo de embargo. Más `function_PBO.py` y un script `5_pbo.py`.
Su README declara explícitamente el objetivo: *"Address Overfitting DRL Agents"*.

**La conclusión del ecosistema es esta asimetría:** AI4Finance *sabe* cómo
evaluar honestamente, y lo demuestra en un repo de **204 stars**. Su flagship de
**16 638 stars** — el que la gente clona, cita y del que extrae conclusiones — no
usa nada de eso, y además incluye un optimizador que busca 30 configuraciones
sobre Sharpe sin deflactar. Los controles existen pero no están donde importan.

Nota sobre purging y embargo para este proyecto: no se incorporaron porque no
aplican. El purgado corrige solapamiento entre muestras de entrenamiento y test
cuando las etiquetas abarcan varios periodos; aquí los rebalanceos son
trimestrales y contiguos, sin solapamiento, y las tiras del CSCV en
`domain/stats/pbo.py` no comparten observaciones. Añadir la maquinaria habría
sido ceremonia sin efecto.

---

## 8. Lo único que se incorporó, y su medición

El usuario impuso la regla correcta: **si se incorpora algo, hay que medirlo de
verdad.** De todo el ecosistema solo un componente pasó el filtro de "vale la
pena probarlo": el **índice de turbulencia** de FinRL, por ser la única pieza con
disciplina temporal correcta y por atacar un problema real de este proyecto (los
drawdowns de −72% a −85% del momentum sobre el universo honesto).

Implementado en `domain/backtest/turbulence.py` con dos mejoras sobre la fuente:
umbral point-in-time (percentil de la propia ventana rodante, no de la muestra
completa) y sin ejecución en la misma barra.

### 8.1 Primer resultado: un bug, no una señal

Aplicado al universo completo, el overlay quedaba en efectivo el **86% de los
rebalanceos, idéntico para todo percentil** (80, 90, 95). Diagnóstico:

```
activos con historia completa (p): 336
observaciones en la ventana  (n): 252
ratio p/n: 1.333                       <- covarianza SINGULAR por construccion
turbulencia 1707.5 vs umbral p90 249.0 -> 6.9x
FinRL usa Dow 30: p/n = 0.119          <- bien condicionado
```

Con más activos que observaciones la covarianza muestral es singular y `pinv`
infla la distancia de cualquier punto fuera del sample. **El índice medía
dimensionalidad, no estrés de mercado.** FinRL no tropieza con esto solo porque
lo aplica al Dow 30.

Reparación: cap de 30 activos, los más líquidos, seleccionados point-in-time.
`p/n` vuelve a 0.119, el ratio baja de 6.9x a 0.45x y los percentiles por fin
discriminan (11%, 21% y 39% de efectivo).

### 8.2 Segundo resultado: mejora aparente que no pasa el control

Universo honesto, out-of-sample 2018-2024:

| Variante | SR net | CAGR | maxDD | % efectivo |
|---|---|---|---|---|
| momentum solo (baseline) | 0.171 | −1.18% | −72.1% | 0% |
| + turbulencia p80 | **0.266** | 3.43% | −63.4% | 39% |
| + turbulencia p90 | 0.264 | 2.85% | **−54.3%** | 21% |
| + turbulencia p95 | 0.241 | 1.76% | −64.2% | 11% |

Leído así es tentador: +0.095 de Sharpe y 18 puntos menos de drawdown. Pero la
pregunta correcta no es "¿mejoró?" sino **"¿mejoró más que reducir exposición sin
mirar nada?"**. Control: estar en efectivo **al azar** en la misma fracción de
rebalanceos, 40 semillas:

```
p80 (cash 39%): turbulencia +0.266  vs azar mediana +0.148  p95 +0.364  -> percentil 82%
p90 (cash 21%): turbulencia +0.264  vs azar mediana +0.181  p95 +0.351  -> percentil 78%
p95 (cash 11%): turbulencia +0.241  vs azar mediana +0.206  p95 +0.295  -> percentil 72%
```

**Ninguno alcanza el percentil 95.** La mejora no es distinguible de simplemente
estar menos invertido. El código queda en el repo con el veredicto escrito en su
docstring y bloqueado en tests —el diagnóstico de `p/n` es un hallazgo real sobre
el método, y el resultado negativo vale conservarlo— pero **no está avalado como
fuente de alfa**.

Regla que deja esta medición, y que aplica a cualquier overlay futuro:
**todo mecanismo que reduzca exposición debe superar el control de abstención
aleatoria a la misma frecuencia antes de poder llamarse habilidad.**
