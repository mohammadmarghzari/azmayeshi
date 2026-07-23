# Portfolio360 — Deep Project Context (companion to SKILL.md)

## What this app is
A Persian-language (RTL) investment-portfolio analytics app ported from a 6,743-line
Python/Streamlit original (`app.py`, kept in-repo as migration reference only) to Kotlin
Multiplatform + Compose Multiplatform. 19 tabs today: allocation, risk/return, price chart,
transactions journal, markets, watchlist, style comparison, efficient frontier, advanced
options, Black-Litterman, stress test & Monte Carlo, rebalance, benchmark, live data,
save portfolio, alerts, Iran tools, commodity deposit certificates, Bourse options, IME Live.

## Two mirrored repositories — both must always receive identical Kotlin changes
- `mohammadmarghzari/azmayeshi` → branch `claude/python-to-kotlin-migration-bozko2`
- `mohammadmarghzari/sarmaye-portfolio-tool` → branch `main`
Docs may differ per repo (the user pushes instruction files to sarmaye directly — always
`git pull` it before mirroring). A missed mirror file has broken CI once; verify with
`diff -rq` on the `composeApp/src` trees after copying.

## Module map
| Module | Role |
|---|---|
| `core/` | Pure JVM: PortfolioEngine, EfficientFrontier, BlackScholes+Greeks, BlackLitterman, MonteCarloEngine, StressTest, Rebalancing, RollingMetrics, FactorExposure, Seasonality, IranTools, Stats/LinearAlgebra + 5 Ktor clients (Yahoo, CNN Fear&Greed, RSS news, IME, world commodities) each with TtlCache. 10 unit tests run in CI. |
| `composeApp/` | KMP app. `commonMain` = all UI/state/data-layer; `androidMain` = MainActivity (crash-recovery handler → CrashScreen), tilt sensor, adaptive vector launcher icon; `desktopMain` = entry point. |
| `devpreview/` | Local-only JVM alias of commonMain+desktopMain for offline compile checks (sandbox has no dl.google.com → no AGP). Not in git. `expect/actual` breaks here — banned pattern. |

## State & persistence
`AppState` is a facade: network clients live in `data/MarketRepositories`, the
fetch→optimize pipeline in `state/PortfolioSession`, both constructor-injected. Durable
storage: `data/AppPersistence` — one versioned JSON document (`portfolio360_state_v1`) via
multiplatform-settings 1.2.0 holding journal, favorites, selections, alerts, prefs; restored
before first composition, saved through a `snapshotFlow` that drops its first emission.
Computed market data is intentionally NOT persisted. `SavedPortfolio` is still
session-scoped (metrics types lack serializers) — known debt.

## Known landmines (each cost a real debugging session)
- Java `String.format`: an unescaped `%` in Persian labels crashes at runtime
  (`"CVaR 95%: %.2f%%"` did). Escape as `%%`.
- `RowScope.weight()` must be called unqualified — an explicit import collides with an
  internal symbol.
- try/catch cannot wrap composable invocations (Compose compiler restriction).
- Rotating drawn geometry for border effects sweeps across wide cards (energyRing bug);
  rotate gradient phase instead.
- Reading animated `State` in a composable body recomposes every frame — read in
  drawBehind/drawWithContent/graphicsLayer lambdas.
- Persian RTL flips `TopAppBar` slots (navigationIcon renders right) — expected, not a bug.
- Network: Yahoo/CNN need VPN inside Iran; `api.ime.co.ir` is blocked outside Iran. Error
  states must keep explaining this.
- The IME API key is hardcoded in `ImeClient.kt` (inherited from the Python app) — must move
  out of source before any public release (roadmap step 9).

## Roadmap (approved by the user, impact-ordered)
Done: design system + UiState (P1) · flat dark redesign (P2) · motion polish (P3) ·
data/state split (P4) · journal, markets, watchlist (P5.1–5.3) · durable persistence (A1).
Next: home dashboard (C1) → settings screen (A2) → search/sort (C3) → backtesting engine
(B1) → onboarding (A4) → CSV/PDF export (B2) → goals via Monte Carlo (C2) → secrets +
signed release (A3/D1) → rolling/factor UI (B4) → portfolio comparison (B3) →
notifications (C4) → tests/hygiene (D2/D3) → widget (C5) → performance (D4).
