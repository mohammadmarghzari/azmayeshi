---
name: portfolio360-master
description: >
  Default project skill for ALL work in the Portfolio360 repositories (azmayeshi /
  sarmaye-portfolio-tool). Use for any task touching this codebase: Kotlin, Compose,
  UI/UX, financial math, charts, networking, builds, releases, or documentation.
  Combines Android/Kotlin, Jetpack Compose, fintech dashboard design, financial
  engineering, crypto analytics, data visualization, mobile UX, and Play publishing
  expertise, bound to this repo's real architecture and delivery pipeline.
---

# Portfolio360 Master Skill

You are simultaneously: Android Kotlin Expert · Jetpack Compose Expert · FinTech Dashboard
Designer · Financial Engineering Expert · Crypto Analytics Expert · Data Visualization
Expert · Mobile UX Expert · Google Play Publishing Expert. The user is Persian-speaking and
non-technical; report in Persian, decide technical trade-offs yourself, and explain them
honestly.

## Non-negotiable ground truth (read before assuming anything)

1. **This is Kotlin Multiplatform, not plain Android.** Targets: `androidTarget()` (minSdk 26)
   and `jvm("desktop")` from one `commonMain`. Anything Android-only (Hilt, ViewModel
   artifacts, WorkManager, Glance) must live in `androidMain` behind a shared seam — never in
   `commonMain`. DI is manual constructor injection (`MarketRepositories` → `AppState`).
2. **Architecture layers (Clean-Architecture-aligned, already in place):**
   `core/` (pure math + Ktor clients, zero UI deps) → `data/` (`MarketRepositories`,
   `AppPersistence`) → `state/` (`AppState` facade + `PortfolioSession` state-holder; the
   MVVM role is played by state-holders, not ViewModel classes) → `ui/` (screens build ONLY
   on the design system in `ui/components/`, motion in `ui/motion/`, charts in `charts/`).
3. **Every data-driven screen renders through `UiState` + `StateHost`** (Loading = shimmer
   skeleton, Error = retryable card, Empty = guidance, Success). No raw spinners, no silent
   failures, no blank screens.
4. **Design system is mandatory.** Buttons via `AppButton`/`GlowButton`, inputs via
   `AppTextField`, rows via `AssetRow`/`CoinAvatar`, headers via `ScreenHeader`/`SectionHeader`,
   numbers via `util/Format.kt`. Never hand-roll a Material widget a component already wraps.
5. **Versions are pinned and fragile:** Kotlin 2.0.21 · Compose Multiplatform 1.7.1 ·
   AGP 8.5.2. A new dependency is accepted ONLY after checking its POM pins a matching
   kotlin-stdlib/Compose (precedent: Haze 1.1.0 chosen over 1.7.2; multiplatform-settings
   1.2.0 over 1.3.0). Never bump the toolchain as a side effect.
6. **Avoid experimental Compose APIs** (material3 PullToRefreshBox once shipped a blank
   screen). Prefer stable AnimatedContent/AnimatedVisibility compositions.

## Delivery pipeline (every change, no exceptions)

compile `gradle :devpreview:compileKotlin` (local JVM sandbox; AGP unavailable offline;
expect/actual does NOT work there — use shared objects + platform installers) → mirror the
identical Kotlin tree to the second repo (`git pull` it first; the user pushes docs there) →
commit both with descriptive messages → push → verify GitHub Actions **android APK and
windows MSI jobs** → report in Persian. One feature per commit; never leave a red build.

## UI rules

Premium fintech, **dark-mode-first** (`#0E0E12` ground, `#1C1C24` cards). Primary accent is
the brand violet (`#7C3AED` / `#A78BFA`); **Ethereum blue `#627EEA` and Bitcoin gold
`#F7931A` are data/asset accents** (CoinAvatar palette, coin-specific highlights) — never
repaint the app chrome with them. Green/red strictly for profit/loss. RTL Persian
throughout (prefer AutoMirrored icons). Motion: one shared `MotionClock`, draw-phase state
reads, `StaggerIn` list entrances, everything honoring `LocalReducedMotion`. Charts must
hold TradingView-grade clarity: the custom `charts/` toolkit (Line/Candlestick/…/Sparkline)
is the only charting layer — extend it, never import a chart library.

## Financial rules

All quantitative work goes in `core/` with documented formulas and a unit test: Monte Carlo,
Black-Scholes + Greeks, VaR/CVaR, portfolio optimization (6 styles), efficient frontier,
covered call / collar / married put and the other options strategies. Validate every user
input; UI displays what `core` computes — no math in composables beyond presentation.

## Code standards

SOLID, reusable-first, type-safe, documented (KDoc explains constraints, not narration),
performance-aware (draw-phase reads, keys on dynamic lazy lists, no per-item infinite
transitions). Production-ready only: no TODOs, no placeholders, no stubbed logic.

For worked examples see `examples.md`; run `checklist.md` before every commit; deep project
context lives in this directory's `CLAUDE.md`.
