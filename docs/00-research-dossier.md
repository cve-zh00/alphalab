# Dossier de investigación — Sistema de análisis de inversiones con agentes LLM

> Corte: 2026-10-13. Todas las referencias arXiv fueron verificadas contra la API de arXiv.
> Todos los conteos de stars fueron verificados contra la API de GitHub.
> Lo marcado con ⚠️ proviene de lectura de full-text por agente delegado y no está
> verificado contra la fuente primaria por mí directamente.

---

## 0. Conclusión ejecutiva (leer esto si no lees nada más)

La evidencia 2025-2026 es **convergentemente escéptica** sobre agentes LLM que generan alpha
desplegable. Tres resultados independientes:

1. **StockBench** (2510.02209, Tsinghua): 14 modelos, top-20 DJIA, 82 días. La mayoría bate
   al baseline pasivo en retorno y **todos** lo baten en drawdown — pero **todos** pierden
   contra el baseline en mercado bajista y solo ganan en alcista ⚠️. No modela costes ni
   slippage (los autores lo admiten).
2. **AI-Trader** (2512.10971, HKU): benchmark *live* post-cutoff en 3 mercados. En acciones US
   2 de 6 modelos baten al QQQ; en acciones chinas **ninguno** bate al SSE-50; en cripto solo
   1 de 6 bate al baseline ⚠️. Conclusión de los autores: inteligencia general ≠ buen trading,
   y el control de riesgo es lo que determina robustez entre mercados.
3. **The Alpha Illusion** (2605.16895, Fudan + Imperial): audita FinCon, FinMem, TradingAgents,
   FinAgent, QuantAgent, FLAG-Trader. Al reproducir con fricciones, **TradingAgents cae de
   Sharpe 0.43 → 0.22**, por debajo de buy-and-hold ⚠️. Al cruzar el cutoff del modelo,
   FinMem pasa a −71.85% de retorno ⚠️.

Y el resultado que debería reescribir el diseño del proyecto:

4. **"What survives honest evaluation?"** (2608.27734): construyen un **oráculo con leakage
   deliberado** que alcanza Sharpe 34.7 en diseño / 51.5 en evaluación, y **pasa limpiamente
   tanto Deflated Sharpe (DSR = 1.00) como PBO** ⚠️. Razón: la deflación estadística corrige
   *selección múltiple*, no *contaminación del information set*. Son complementarios, no
   sustitutos. En su certificación sobre 453 acciones US point-in-time + 39 ETFs, **las únicas
   estrategias que sobrevivieron fueron benchmarks pasivos** (SPY B&H: Sharpe 0.85,
   IC [+0.24, +1.53]); **todas las estrategias generadas por LLM fueron rechazadas** ⚠️.

**Implicación de diseño, no de pesimismo:** el leakage debe eliminarse *por construcción*, no
detectarse *a posteriori*. Y el "agente escéptico" no puede ser una persona LLM que debate con
otro LLM — tiene que ser **código determinista con criterios de rechazo verificables**.

Lo que sí tiene evidencia sólida y es donde está el valor real:
- **LLM como extractor/interfaz de información auditable** (filings, earnings calls, noticias)
  — Kim/Muhn/Nikolaev 2407.17866 (Chicago Booth) es el diseño anti-leakage más limpio.
- **Disciplina de provenance**: números por código determinista, narrativa por LLM
  (FinRobot V2 lo implementa y es lo mejor del repo).
- **Journaling y calibración** de decisiones propias — nadie lo construye y es lo que hace
  confiable al sistema.

---

## 1. Revisión de la investigación previa (la de Copilot)

Buena base y bien orientada. Correcciones y complementos tras verificación:

### Errores

| Afirmación de Copilot | Realidad verificada |
|---|---|
| Survey Agentic Trading: "revisa 77 estudios, subconjunto empírico de 19" | **Falso.** El abstract oficial (v2, 2026-10-04) dice **185 estudios seleccionados, subconjunto empírico central de 80 sistemas** en condiciones de ciclo cerrado. |
| "solo 2 reportaban protocolos de división temporal, 1 costes de transacción, 1 sesgo de supervivencia" | **Muy exagerado.** Full-text ⚠️: time split 50/80 (62.5%), costes 47/80 (58.8%), survivorship 48/80 (60.0%). Lo que **sí** es correcto: **R3 = 0 trabajos** alcanzan el nivel máximo de reproducibilidad, y solo **2/80 alcanzan R2**. |
| Devil's Advocate "2024" sin ID | Es **arXiv:2405.16334**, Wang/Li/Deng/Roth/Li, EMNLP 2024 Findings. WebArena 23.5% success (+3.5pp), −45% trials. |
| Clawtrade presentado como referencia relevante | **37 stars, Go, último push 2026-03-14.** Proyecto personal pre-maduro: promete 7 exchanges, 14 tools, 5 subagentes, vault AES-256 y GraphQL en 137 commits ⚠️. Es un catálogo de patrones de integración, no una referencia de arquitectura. Lo único que aporta frente a los otros: aborda ejecución real y riesgo pre-trade. |

### Omisiones importantes

Copilot no incluyó los **dos papers más decisivos para la tesis del agente escéptico**:
**The Alpha Illusion** (2605.16895) y **"What survives honest evaluation?"** (2608.27734).
Son precisamente los que convierten "escéptico = personalidad crítica" en
"escéptico = gates deterministas", que es la diferencia entre un juguete y un sistema.

### Lo que acertó

- Todos los conteos de stars verificados exactos (TradingAgents 110 986, Qlib 49 346,
  Freqtrade 55 350, FinGPT 21 399, LEAN 21 984, Hummingbot 20 432, FinRL 16 637,
  FinRobot 8 209, TensorTrade 7 196, FinRL-Meta 1 949).
- Todos los demás arXiv IDs verificados correctos.
- El orden de lectura propuesto es bueno.
- **El experimento decisivo que propone es el correcto**: medir si el escéptico mejora
  resultados fuera de muestra, no si produce el análisis más convincente. Lo adopto.
- La advertencia sobre interpretar las stars es correcta.

---

## 2. Papers (verificados)

### Núcleo de evaluación — leer primero

| ID | Título | Aporte |
|---|---|---|
| **2605.19337** | Agentic Trading: When LLM Agents Meet Financial Markets (Xia et al., Shenzhen Univ., v2 oct-2026) | Survey. Marco **A-C-A** (Architecture/Capability/Adaptation). Rúbrica de reproducibilidad **R0–R3**: R0=45, R1=33, R2=2, R3=0 ⚠️. Guidelines MR-1–MR-7. |
| **2608.27734** | What survives honest evaluation? Leakage-safe, search-aware assessment (Gençay, ago-2026) | **El más importante.** Protocolo leakage-safe estructural + search-aware. Oráculo con leakage pasa DSR y PBO. Solo pasivos sobreviven. |
| **2605.16895** | The Alpha Illusion (Ye et al., Fudan+Imperial, may-2026) | Position paper. Audita 6 sistemas. Protocolos **P1–P6** escalonados por fuerza del claim. |
| **2510.02209** | StockBench (Chen et al., Tsinghua) | Benchmark secuencial. Baseline pasivo queda 12º/14 en retorno pero gana en bajista. |
| **2512.10971** | AI-Trader (Fan et al., HKUDS) | Benchmark live post-cutoff, 3 mercados, 5 tools MCP. |
| **2510.11695** | When Agents Trade / Agent Market Arena (Qian et al., 17 autores, incl. Lopez-Lira) | **La arquitectura del agente domina la variación; el backbone importa menos.** |

### Arquitecturas de agentes

| ID | Título | Patrón a estudiar |
|---|---|---|
| 2412.20138 | TradingAgents (UCLA+MIT+Tauric) | Debate bull/bear con árbitro, risk team, LangGraph |
| 2407.06567 | FinCon (NeurIPS 2024) | **Conceptual Verbal Reinforcement**, jerarquía manager-analyst, control CVaR |
| 2311.13743 | FinMem (Stevens) | Memoria por capas (shallow/intermediate/deep) + perfil de carácter |
| 2402.18485 | FinAgent (NTU) | Multimodal (precios+noticias+velas), reflexión en dos niveles |
| 2405.14767 | FinRobot plataforma (AI4Finance) | Arquitectura de 4 capas, Financial CoT, Smart Scheduler |
| 2411.08804 | FinRobot equity research (ICAIF 2024) | 3 agentes: Data / Concept / Thesis |
| 2407.18957 | StockAgent | Simulación ABM de mercado |
| 2405.16334 | Devil's Advocate | Reflexión anticipatoria: 3 intervenciones introspectivas |

### Alpha mining, RL y surveys

2308.00016 Alpha-GPT · 2502.16789 AlphaAgent (alpha decay) · 2011.09607 + 2111.09395 FinRL ·
2112.06753 + 2304.13174 FinRL-Meta · 2408.06361 LLM Agent in Financial Trading: A Survey ·
2306.06031 FinGPT · *The Evolution of RL in Quantitative Finance* (ACM Comput. Surveys 2025, 167 pubs)

### Evidencia crítica / metodología

- **2309.17322** Glasserman & Lin — *Assessing Look-Ahead Bias in Stock Return Predictions from
  GPT Sentiment*. Descompone en look-ahead bias (in-sample) y **distraction effect** (persiste
  OOS). Anonimizar entidades como de-biasing.
- **2304.07619** Lopez-Lira & Tang — *Can ChatGPT Forecast Stock Price Movements?* El paper "a
  favor" más citado. Matices críticos: GPT-1/2/BERT no predicen (¿memorización?); el edge está
  en small caps y noticias negativas — donde las fricciones son peores.
- **2502.21206** Sarkar & Vafa — *Chronologically Consistent LLMs*.
- **2407.17866** Kim, Muhn & Nikolaev (Chicago Booth) — *Financial Statement Analysis with LLMs*.
  Diseño anti-leakage con datos anonimizados. La evidencia más limpia a favor del LLM como
  analista.
- Bailey & López de Prado: *Probability of Backtest Overfitting*, *Deflated Sharpe Ratio*,
  *Pseudo-Mathematics and Financial Charlatanism*.
- Harvey, Liu & Zhu 2016 (NBER w20592): umbral **t > 3**, no 2. Hou/Xue/Zhang 2020: de 447
  anomalías, 64% insignificantes al 5%.

---

## 3. Repositorios (stars verificadas 2026-10-13)

### Agentes financieros

| Repo | Stars | Lic. | Últ. push | Veredicto |
|---|---|---|---|---|
| TauricResearch/TradingAgents | 110 986 | Apache-2.0 | 2026-10-03 | Referencia de debate. **No modela costes** (explícito en su código) |
| **HKUDS/AI-Trader** | 22 768 | **NINGUNA** | 2026-06-11 | **No usar como referencia de código.** Ver §3.1 |
| FinGPT | 21 399 | MIT | 2026-09-23 | Componente de análisis, no sistema |
| FinRobot | 8 209 | Apache-2.0 | 2026-09-28 | **Mejor arquitectura de provenance** |
| galleonlabs/hypergrok-trading-desk | 100 | MIT | 2026-10-13 | Documentación/prompts, no software. Excelente modelo de *gobernanza* |
| Paparusi/Clawtrade | 37 | MIT | 2026-03-14 | Pre-maduro. Solo útil como catálogo de integración |

### 3.1 HKUDS/AI-Trader — el repo no es el paper

Inspeccionado en clon local (`reference/ai-trader`, 29 604 LOC Python en 74 archivos).
El paper **arXiv:2512.10971 sí es una referencia central** (§2). El **repositorio es otra cosa**:

**Problema 1 — trampa de licencia.** El README muestra un badge `License-MIT` **estático y
hardcodeado** que enlaza a `LICENSE`, pero **no existe ningún archivo de licencia en el árbol
git** y la API de GitHub reporta `license: null`. Sin licencia, el default legal es *all rights
reserved*: no hay permiso para usar, modificar ni redistribuir. **Bloqueante** para construir
sobre él.

**Problema 2 — la capa de evaluación del paper no está en el repo.** Grep sobre los 74 archivos
Python: `backtest` → 0 archivos. `walk-forward` → 0. `deflated` → 0. `point-in-time` → 0.
`survivorship` → 0. `slippage` → 0. `transaction_cost` → 0. `pbo` → 0. Lo que el paper describe
como benchmark riguroso con anti-contaminación temporal **no es el código publicado aquí**.

**Problema 3 — tampoco hay arquitectura de agentes.** Los agentes son *externos* (Claude Code,
Codex, Cursor, OpenClaw): leen un `SKILL.md` y se registran por REST. La capa de agentes son
**6 archivos SKILL.md + una API**. Está más cerca del modelo de HyperGrok (documentación y
prompts) que de TradingAgents (código de agentes).

**Qué es realmente:** una plataforma SaaS de *social/copy trading* con competencias. Los
archivos más grandes son `challenges.py` (2233 LOC), `market_intel.py` (1911),
`routes_signals.py` (1756), `database.py` (1695), `team_missions.py` (1394). Hay
`challenge_scoring.py`, `manage_leaderboard_exclusions.py`, `rewards.py`,
`monthly_challenges.py`, `experiments.py` (A/B testing sobre prompts de agentes),
`experiment_notifications.py`. Los tests (~20 archivos) cubren auth, permisos de admin,
leaderboards, rewards y challenges — **no la evaluación científica**.

**Linaje.** Rastros de un rebrand: `CLAWTRADER_CORS_ORIGINS` en `.env.example`,
`channels.clawtrader.*` y plugins `@clawtrader/tradesync` / `@clawtrader/copytrade` en los
SKILL.md, y `docs/README_AGENT.md` instruye literalmente
`git clone https://github.com/TianYuFan0504/ClawTrader.git`. Nota de desambiguación:
**`TianYuFan0504/ClawTrader` ≠ `Paparusi/Clawtrade`** (el segundo es Go, 37 stars) — nombres
parecidos, proyectos distintos.

**Otras señales:** datos de precios vía Alpha Vantage con **fallback a yfinance**; `license:
null` con badge MIT; 70 issues abiertas; último push 2026-06-11 (4 meses); archivo temporal
`impeccable.context.tmp` olvidado en la raíz. 22 768 stars y 3 437 forks en menos de un año
desde su creación (2025-10-23), con badge de Trendshift y el claim *"100% Fully-Automated"*.

**Veredicto:** usar el **paper** como una de las referencias de evaluación. **No usar el repo**
como referencia de arquitectura ni como dependencia — sin licencia y sin la capa que lo haría
valioso. Lo único reutilizable como idea es el patrón de onboarding (un agente lee un `SKILL.md`
y se auto-registra), que de todos modos HyperGrok ejecuta mejor y con MIT explícita.

### Quant / backtesting / ejecución

| Repo | Stars | Lic. | Nota |
|---|---|---|---|
| freqtrade | 55 350 | GPL-3.0 | Cripto, maduro, FreqAI |
| microsoft/qlib | 49 346 | MIT | Plataforma ML cuant, foco A-shares |
| nautechsystems/nautilus_trader | 29 927 | LGPL-3.0 | Event-driven core Rust, **backtest = live mismo código** |
| QuantConnect/Lean | 21 984 | Apache-2.0 | Motor C#/Python |
| hummingbot | 20 432 | Apache-2.0 | Capa de ejecución cripto |
| FinRL | 16 637 | MIT | RL financiero |
| tensortrade | 7 196 | Apache-2.0 | RL, mantenimiento lento |
| FinRL-Meta | 1 949 | MIT | Entornos y datos para RL |

---

## 4. Lo que vale la pena robar de cada arquitectura

### FinRobot V2 (`finrobot_desktop`) — el botín principal

Tesis del repo: *"numbers are computed by code, judgment is supplied by the LLM. El modelo
nunca emite una cifra que no se pueda trazar a una llamada en `compute/`."*

Estructura real (verificada en clon local): 9 agentes **definidos como markdown**
(`engine/agents/instructions/*.md` — bull, bear, judge, data, analysis, modeling, synthesis,
report), 7 pipelines tipados (`equity_research`, `dcf`, `ddm`, `lbo`, `comps`,
`earnings_analysis`, `ic_memo`), 27 operadores de cómputo + 7 de auditoría, 176 tests unitarios.

**El patrón clave: Output Contract Gate** (`artifact/contract.py`) ⚠️
- Corre en el **único boundary de persistencia** (`builder → store.save`), viendo el artefacto
  completo ensamblado — donde los gates por fragmento son ciegos.
- 7 cláusulas como **funciones puras** `check(artifact) -> Finding | None`. Ej: C1 exige que el
  ratio target/precio esté en banda calibrada; C1b exige que los métodos de valuación se
  corroboren entre sí; C7 impide que un target retirado "resucite" en slots de fallback.
- **Mecanismo decisivo: VALUE-withhold, no refuse-to-rate.** Una violación nulifica el *número*
  en todos los slots y estampa `[CONTRACT/*]`, pero **preserva el veredicto direccional y el
  análisis**. Nunca lanza excepción (blanquearía el run).
  *"Corrupt or out-of-band data withholds the price, never the judgment."*
- Gate "tonto" por decisión de arquitectura: cero I/O, no importa `compute/`/`data/`/`pipelines`,
  bandas como constantes calibradas. Verifica el resultado sin re-ejecutar el cómputo.

Otros patrones: 6 operadores de auditoría que chequean **narrativa contra números**
(`narrative_numeric_grounding`, `narrative_divergence`, `currency_caliber`, `ev_bridge`,
`sector_sign`, `ttm_period`); degradar a **rango** cuando los métodos divergen; agentes como
markdown editable sin tocar Python.

### TradingAgents

- Debate con **router de turnos y tope duro**, no consenso emergente: `count >= 2*max_rounds`
  → el Research Manager (modelo deep-think) resuelve leyendo el historial ⚠️.
- **Dos tiers de modelo**: quick-think para analistas/researchers/trader, deep-think para los
  árbitros. Directamente aplicable a local vs frontier.
- Cada analista corre como **sub-grafo con historial privado y `output_schema` restringido** →
  paralelismo sin colisión de claves ni contaminación de tool-calls ⚠️.
- `max_tool_rounds` + `wrap_up` contra bucles de tool-calling.
- **Settle-then-reflect**: cada run registra su decisión; un run posterior liquida las decisiones
  vencidas (busca retorno realizado y alpha vs benchmark) y el Reflector genera una lección de
  2-4 frases en prosa. **No es vector store** — y solo el Portfolio Manager lee ese contexto ⚠️.
- **Honestidad arquitectónica a imitar**: su `backtest.py` declara que NO es simulador de
  portfolio y NO modela costes, *"inventar quantity/fill price/cash ledger aquí pondría un
  modelo de ejecución detrás de una herramienta de evaluación"* ⚠️. Separa evaluación de
  decisión de simulación de ejecución.
- Integridad point-in-time: SEC EDGAR **as-filed** (Apple 2008 lee $39.6B aunque se reexpresó
  a $36.2B en 2010).

### HyperGrok

No es software, es un **modelo de gobernanza** — y es su mejor aporte:
- **Un solo escritor**: seis agentes leen, uno envía, una vez por aprobación, y reconcilia por
  client order id.
- **Aprobación por ticket id fuera del chat**: *"un bot que puede leer una aprobación también
  podría escribirla"*. El gate vive en la plataforma, no en el prompt.
- **Sizing sobre stop estresado**: un stop disparado es orden de mercado — paga slippage y taker
  en ambas patas. Dimensiona sobre lo que el stop *costará*, no sobre su precio de disparo.
- **Techos que no se pueden negociar**: tu archivo de límites solo puede *apretar* los caps del
  desk, nunca aflojarlos.
- Wallet trade-only (puede operar, no puede retirar). Testnet primero. Proceso/outcome
  calificados por separado en el journal.
- `scripts/research_dataset.py`: bundles con requests, respuestas capturadas, filas CSV
  observadas y **manifiesto hasheado**; distingue `complete` / `incomplete` / `unavailable` y
  **nunca rellena un hueco con datos sintéticos**.

---

## 5. Síntesis: los 8 patrones con evidencia o mérito arquitectónico

1. **Separación cómputo determinista / narración LLM** (FinRobot). El único patrón que
   simultáneamente mejora precisión, auditabilidad y testabilidad.
2. **Gate de contrato en el boundary de persistencia, con value-withhold** (FinRobot).
3. **Debate con árbitro y tope duro de turnos** (TradingAgents). Ojo: AMA (2510.11695) sugiere
   que la arquitectura del agente domina la variación — el debate importa, pero hay que
   *medirlo*, no asumirlo.
4. **Settle-then-reflect con lecciones en prosa corta** (TradingAgents, FinCon CVRF).
5. **Memoria por capas** (FinMem, TradingGPT). El survey Agentic Trading advierte un gap:
   **look-ahead vía memoria semántica** — la memoria puede reintroducir leakage.
6. **Risk gate determinista con CVaR** (FinCon). AI-Trader concluye que es *el* factor de
   robustez cross-market.
7. **Un solo escritor + aprobación fuera de banda** (HyperGrok).
8. **Evaluación leakage-safe + search-aware** (Gençay). El que nadie implementa.

---

## 6. Stack recomendado (por capa)

| Capa | Elección | Por qué |
|---|---|---|
| Storage | **Parquet + DuckDB** (+ SQLite para estado, LanceDB para vectores) | Analítica single-node sin servidor; DuckDB lee Parquet directo |
| Datos | **SEC EDGAR** (as-filed/XBRL frames) + **FRED** gratis como base; **Tiingo** barato o **Polygon** para US; CCXT para cripto | EDGAR y FRED son fuente primaria oficial. **Evitar yfinance en evaluación**: survivorship bias (solo tickers vivos) |
| Backtest | **NautilusTrader** (event-driven, backtest = live) | Paridad research→producción; evita la clase de bug que hace que el backtest mienta. VectorBT solo para sweeps exploratorios |
| Agentes | **PydanticAI** | Outputs tipados y validados con retry, DI estilo FastAPI, testeable con pytest, provider-agnostic. Encaja con TDD/80% |
| Modelo local | **Qwen3.x 27-32B 4-bit vía MLX** (~16-19 GB) + un 8B para extracción | Apache-2.0, tool-calling fuerte. MLX es el runtime más eficiente en Apple Silicon |
| Scheduling | **launchd** | **Decisivo: cron descarta los jobs perdidos si la Mac durmió; launchd los encola y ejecuta al despertar** |
| Observabilidad | **Langfuse** self-hosted + **DeepEval** (pytest) | Tracing por OTel desde PydanticAI; eval como unit test |

### El argumento real para modelos locales (más allá del costo)

**Reproducibilidad temporal.** Una API frontier se actualiza en silencio: el modelo que usaste
en el backtest de enero no es el de marzo, y su cutoff se mueve hacia adelante — lo que destruye
la higiene temporal y hace el backtest irreproducible. Un **modelo open-weight pinneado con
cutoff conocido** permite exactamente lo que exigen Gençay (2608.27734) y Sarkar & Vafa
(2502.21206): evaluar con un modelo cuyo conocimiento *termina antes* del periodo de test.

Corolario de diseño: **versión del modelo y su cutoff son metadatos de primera clase** de cada
run, igual que el hash del dataset.

### Requisitos de RAM (Apple Silicon)

`RAM ≈ params × bytes_por_param × 1.15 + contexto`

| Precisión | 8B | 27-32B | 70B |
|---|---|---|---|
| 4-bit | ~5 GB | ~16-19 GB | ~40 GB |
| 8-bit | ~8-9 GB | ~27-31 GB | ~70-75 GB |

32 GB → 8-14B cómodo, 27-32B justo. **64 GB → sweet spot** (27-32B holgado). 128 GB → 70B
dense 4-bit. Cuidado con MoE: cargan *todos* los pesos aunque activen una fracción.

---

## 7. Controles metodológicos no negociables

Derivados directamente de Gençay (2608.27734), Alpha Illusion (P1–P6) y el survey (MR-1–MR-7).

**Anti-leakage — estructural, porque la estadística no lo detecta**
1. Point-in-time estricto: cada registro con `as_of` y `observed_at`. Fundamentales **as-filed**,
   nunca reexpresados.
2. Feature registry que hace el look-ahead **inexpresable**, no solo prohibido.
3. Modelo con cutoff **anterior** al periodo de test; cutoff registrado por run.
4. El prompting **no es un control** (2511.14214 ⚠️: los LLM no respetan cronología por
   instrucción).
5. Universo point-in-time con delisted incluidos. Sin esto el backtest está sesgado al alza.
6. Considerar anonimización de entidades (Glasserman & Lin).

**Anti-overfitting — search-aware**
7. **Trial ledger completo**: *todos* los backtests, incluidos los generados por el loop del
   agente. La autonomía infla el trial count silenciosamente.
8. DSR con `SR₀` indexado al trial count **real**, umbral > 0.95. Referencia: tras 100 ensayos,
   un Sharpe crudo de 1.0 da DSR ≈ 0.60 ⚠️; se necesita ≈1.8 para certificar.
9. PBO vía CSCV < 0.5.
10. t > 3, no 2 (Harvey-Liu-Zhu).

**Anti-fricción**
11. Costes como *constraint en cada etapa*, nunca filtro ex-post: spread, slippage, impacto
    no-lineal, comisión mínima, borrow. Siempre reportar net-of-costs junto a gross.
12. Modelar capacity. El edge de los LLM aparece en small caps y noticias negativas — justo
    donde las fricciones son máximas.

**Anti-fallo operacional**
13. Ningún agente ejecuta directo. Capa determinista con límites duros y kill-switch.
14. Todo dato externo es **no confiable**: una noticia es un vector de prompt injection.
15. Aprobación humana fuera del canal del agente.

**Validación honesta**
16. Walk-forward con IC bootstrap que excluya el cero.
17. Benchmark contra pasivo **net-of-costs**.
18. Paper trading forward prolongado; medir la **evaporation curve** (design SR → eval SR → live SR).

### Prueba ácida del harness (antes de construir cualquier agente)

Replicar el experimento de Gençay sobre tu propia infraestructura:

- **Debe certificar** SPY buy-and-hold (Sharpe ≈ 0.85, IC excluyendo cero).
- **Debe rechazar** un oráculo con leakage deliberado que le inyectes a propósito.
- **Debe rechazar** una estrategia de ruido puro tras N ensayos.

Si el harness no pasa estas tres, **nada de lo que mida después significa algo**. Esto se
construye *primero*, antes del primer agente.

---

## 8. Riesgo regulatorio y operacional (verificar jurisdicción)

- Capital propio ≠ asesoría financiera. Gestionar capital de terceros cambia el régimen
  (Investment Adviser en US; en Chile, regulación CMF).
- **PDT rule** FINRA 4210: 4+ day trades en 5 días hábiles en cuenta margin → mínimo US$25k.
  Cuenta cash evita parte del problema.
- Términos de uso de las APIs de datos y del broker: muchas licencias prohíben
  redistribución y uso automatizado.
- Impuestos: wash-sale rule (US), tributación de ganancias de capital en Chile.
- Spoofing y manipulación son ilegales incluso operando capital propio.

---

## 9. Orden de trabajo propuesto

**Fase 0 — Harness de evaluación honesta.** Antes de cualquier agente. Data layer
point-in-time, trial ledger, DSR + PBO + bootstrap IC, modelo de costes. **Gate: pasar la
prueba ácida de §7.**

**Fase 1 — Capa de cómputo determinista.** Operadores puros + contract gate estilo FinRobot.
Sin LLM en el loop. TDD, 80% cobertura.

**Fase 2 — Un agente analista con provenance.** LLM solo narra y extrae; ningún número
proviene del modelo. Baseline a batir.

**Fase 3 — Bull/Bear/Judge + escéptico determinista.** Y **medir** si bate a la Fase 2 fuera de
muestra. Si no mejora drawdown ni evita operaciones sin edge después de costes, se descarta:
es complejidad y tokens sin valor demostrado.

**Fase 4 — Journal y calibración.** Settle-then-reflect, Brier score / ECE sobre las propias
predicciones del sistema. Esta fase es la que lo hace confiable.

**Fase 5 — Ejecución.** Solo si las fases 0-4 se lo ganaron. Un solo escritor, aprobación por
ticket fuera de banda, testnet/paper primero, sizing sobre stop estresado.

---

## Referencias de código local

```
reference/finrobot/finrobot_desktop/   # patrón provenance + contract gate
reference/hypergrok/                   # modelo de gobernanza y aprobación
```

**Disclaimer:** documento de investigación técnica. No es asesoría financiera ni recomendación
de operar. Los futuros perpetuos pueden liquidar una cuenta.

---

## 10. Estado de implementación — Fase 0 completa (2026-10-13)

Harness de evaluación honesta implementado y verificado. `make check`: 206 tests,
95.60% cobertura, ruff + mypy strict verdes. `make acid-test`: 5 passed.

### Resultados de la prueba ácida

| Criterio | Resultado | Valores |
|---|---|---|
| Certifica SPY B&H 2010-2024 (T=3773) | ✅ CERTIFICADO | SR_ann 0.8374 (ref. paper 0.85), DSR 0.9993, PBO 0.0, IC [0.0194, 0.0843] |
| Rechaza oráculo con leakage (estructural) | ✅ RECHAZADO | `LeakageError` desde `domain/pit/resolve.py` |
| Gates estadísticos ciegos al leakage (esperado) | ✅ DOCUMENTADO | SR_ann ~20.2, DSR 1.000000, PBO 0.0 → certifica fraude |
| Rechaza best-of-100 ruido GBM | ✅ RECHAZADO | SR_ann 1.8543, DSR 0.5538 < 0.95 |

### Verificaciones adversariales

**Mutation test del gate de leakage.** Neutralizando el `raise` en `resolve.py`, la
prueba ácida falla (`DID NOT RAISE LeakageError`); restaurado, vuelve a verde. El test
tiene dientes, no es decorativo.

**El trial ledger es load-bearing.** El mismo ruido GBM sin edge (SR_ann 1.8543):

| `n_trials` declarado | DSR | PBO | IC excl. 0 | Veredicto |
|---|---|---|---|---|
| 1 (deshonesto) | 0.9952 | 0.4239 | sí | **CERTIFICA — falso positivo** |
| 10 (subcontado) | 0.8560 | 0.4239 | sí | rechaza |
| 100 (honesto) | 0.5538 | 0.4239 | sí | rechaza |

PBO y el IC bootstrap **pasan en los tres casos**. Solo el DSR alimentado con el conteo
honesto atrapa el overfit. Esto confirma empíricamente por qué el ledger no es
contabilidad decorativa: subcontar ensayos convierte ruido en descubrimiento.

### Limitación conocida (resolver en Fase 2)

`resolve_feature` es el camino sancionado y lanza `LeakageError`, pero **nada fuerza
todavía a una estrategia a usarlo** — `visible_records` filtra en silencio y sigue
accesible. Cuando existan estrategias reales (Fases 2-3) hay que cablear
`resolve_feature` como único acceso a features, idealmente con un test de arquitectura
que lo imponga mecánicamente, igual que el guard de pureza del dominio.

### Commits

```
d350b59  test: prueba acida del harness - certifica pasivo, rechaza leakage y ruido
43483fb  feat: gates estadisticos DSR, PBO via CSCV y bootstrap IC
9d605a6  feat: trial ledger append-only con hash canonico de especificacion
d462af9  feat: modelo de costes con impacto no-lineal y stressed stop
a69e406  feat: capa de datos point-in-time con invariante anti-leakage
64686fb  chore: scaffold alphalab con gates de calidad y test de arquitectura
```

---

## 11. Estado de implementación — Fases 1-5 (2026-10-13)

`make check`: 664 passed + 3 skipped, 96.58% cobertura, mypy strict 99 archivos, ruff limpio.
`make acid-test`: 6 passed. 18 commits.

### Lo construido

| Módulo | Qué hace |
|---|---|
| `domain/compute/` | `ProvenancedValue` como DAG inmutable; DCF, WACC, DDM, múltiplos, EV bridge, consensus; 5 operadores de auditoría |
| `domain/contract/` + `artifact/` | 7 cláusulas puras con **value-withhold**; `save()` es el único camino de escritura |
| `domain/skeptic/` | 8 criterios deterministas S1-S8; evalúa todos sin cortocircuito; nunca lanza |
| `domain/journal/` | settle-then-reflect; Brier, ECE, reliability diagram, sobreconfianza con signo |
| `domain/analyst/` + `adapters/llm.py` | PydanticAI con `ToolOutput`; dos tiers; ningún número del modelo |
| `domain/debate/` | Router de turnos de tope duro; aislamiento de contexto por rol |
| `domain/schedule/` + `obs/` | Cuatro cadencias con launchd; calendario puro sin `datetime.now()` |
| `domain/experiment/` | El A/B decisivo con bootstrap pareado y corrección de Bonferroni |

### Hallazgo: el gate PBO era un lanzamiento de moneda

Bajo la hipótesis nula, el rank out-of-sample de la mejor configuración in-sample es
uniforme, así que **E[PBO] ≈ 0.5 — el umbral era el valor nulo**. Ruido puro sin edge
atravesaba el gate:

| escenario | PBO | IC sup | p-valor | gate viejo | gate nuevo |
|---|---|---|---|---|---|
| 100 GBM independientes (sin edge) | 0.4239 | 0.4316 | 0.3250 | **pasaba** | rechaza |
| familia con factor común (sin edge) | 0.3949 | 0.4030 | 0.2500 | **pasaba** | rechaza |
| señal genuina estable | 0.0197 | 0.0224 | 0.0000 | pasa | pasa |
| SPY pasivo (N=1) | 0.0000 | 0.0000 | 0.0000 | pasa | pasa |

Triple criterio nuevo: punto < 0.25, límite superior del IC bootstrap < 0.25, y
p-valor < 0.05 contra la nula simulada.

### El experimento A/B — resultado negativo, reportado tal cual

n_oos = 20, datos GBM sin edge, dobles deterministas:

| brazo | Sharpe gross | Sharpe net | maxDD | propuestas | ejecutadas | DSR | Brier |
|---|---|---|---|---|---|---|---|
| A: agente único | 0.380 | −0.622 | −0.034 | 20 | 20 | 0.433 | 0.263 |
| B: debate bull/bear/judge | 0.380 | −0.622 | −0.034 | 20 | 20 | 0.433 | 0.263 |
| C: debate + escéptico | 0.000 | 0.000 | 0.000 | 20 | **0** | 1.000 | 0.263 |

- **Debate vs agente único**: Δ Sharpe = 0.0000, IC [0, 0]. **No mejora.**
- **Escéptico**: vetó 20 de 20. Pero de esos, **10 habrían ganado y 10 habrían perdido** —
  no discriminó, simplemente dejó de operar. IC de la diferencia [−0.5371, +0.7076]
  incluye cero.
- **Potencia**: Δ Sharpe mínimo detectable = 1.0236, absurdamente grande. El veredicto es
  **INCONCLUSIVE por falta de potencia**, explícitamente distinguido de
  INDISTINGUISHABLE (empate real).

Corrección de Bonferroni sobre 3 pares (confianza por comparación 0.9833), y las 3
comparaciones registradas como ensayos en el trial ledger.

**Qué NO prueba**: corre con dobles deterministas, no con el 27B real — mide arquitectura,
no habilidad del modelo. Sobre datos sintéticos sin edge. No es evidencia de rentabilidad
desplegable.

### Interpretación honesta

El Δ = 0 entre A y B es un artefacto del diseño: dobles deterministas producen el mismo
veredicto direccional por construcción. Mide el cableado, no el debate. Para un veredicto
real sobre el debate hace falta correr los brazos con el LLM y una muestra mucho mayor —
y a ~80 s por debate de una ronda, eso son horas de cómputo, no un test.

El hallazgo sobre el escéptico sí es sustantivo y va en contra de nuestro propio diseño:
**vetar todo no es escepticismo, es abstinencia.** Un gate que rechaza el 100% de los
candidatos mejora el drawdown trivialmente (no operar nunca tiene drawdown cero) sin
aportar discriminación. Los umbrales S1-S8 están calibrados para datos sin edge, que es
justo lo que el experimento les dio. Antes de concluir sobre el escéptico hay que correrlo
sobre candidatos con edge genuino mezclados con ruido, y medir si separa unos de otros.
