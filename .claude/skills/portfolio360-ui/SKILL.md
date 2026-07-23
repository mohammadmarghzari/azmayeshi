---
name: portfolio360-ui
description: >
  Premium fintech UI skill for Portfolio360: visual language, component recipes and motion
  rules for any screen, card, chart, widget or animation work in this repository. Companion
  to portfolio360-master (which always applies); load this one whenever the task is
  primarily visual — layout, styling, theming, charts, or micro-interactions.
---

# Portfolio360 Premium Fintech UI Skill

Role: Senior fintech UI/UX engineer with Apple-level quality bar.
Inspiration set: Revolut, Robinhood, Coinbase, TradingView, Linear — borrowed for *quality
level and restraint*, never copied literally.
Implementation surface: Jetpack Compose (Compose Multiplatform 1.7.1) + Material 3 + Kotlin.
Production-ready code only; every element from the design system in `ui/components/`.

## Visual language (reconciled with the user's approved decisions)

- **Dark-mode-first, flat premium ground** — `#0E0E12` page, `#1C1C24` solid cards. The user
  explicitly chose "flat and clean" over photo backdrops for data screens; that decision
  stands.
- **Soft glassmorphism is an opt-in accent, not the default**: the Haze pipeline
  (`LocalHazeState` + `Card`'s hazeChild path) stays available for the splash screen and
  future hero surfaces. Do not reintroduce blur across data screens without an explicit
  user request.
- **Neumorphism Lite = the existing card treatment**: large rounded cards (18dp), soft
  layered shadow (`Elevations.card`), 1dp gradient edge highlight. Never full neumorphism
  (embossed low-contrast surfaces fail accessibility and fintech legibility).
- **Color discipline**: violet `#7C3AED`/`#A78BFA` = interactive chrome; green/red = P&L
  only; gold = warnings/star; ETH blue `#627EEA` & BTC gold `#F7931A` = asset-identity
  accents (CoinAvatar, coin highlights). Persian RTL everywhere; AutoMirrored icons for
  directional glyphs.

## Component recipes (all primitives already exist — compose, don't reinvent)

| Ask | Build with |
|---|---|
| Portfolio card | `Card(highlighted = true)` + `HeroMetric` (+ `StatusBadge` delta pill) |
| Asset allocation chart | `DonutChart` / `TreemapChart` inside a `Card`; legend colors via `chartColor(rank)` |
| Risk score widget | `GaugeChart` with tinted zones, or `MetricTile` row via `LazyRow` |
| Monte Carlo panel | `LineChart` fan of paths + `MetricTile` percentiles (engine: `core/MonteCarloEngine`) |
| Crypto metrics | `AssetRow` (CoinAvatar + mono value + signed pct) + `Sparkline` |
| Option strategy card | `Card` + payoff `LineChart` with `RefLine`s (strike/spot/BE) + `VerdictCard` |
| Asset list | rows via `AssetRow`, entrance via `StaggerIn`, loading via `SkeletonCard` |
| Empty/error/loading | ALWAYS `StateHost` — never a bare spinner or blank block |

## Motion rules (60fps by construction)

- One shared `MotionClock`; per-frame values read ONLY in draw/`graphicsLayer` lambdas.
- **Spring motion** for interactive responses (press scale, toggles):
  `animateFloatAsState(spring(dampingRatio = Spring.DampingRatioMediumBouncy, stiffness = Spring.StiffnessMedium))`;
  tween(240–320ms, FastOutSlowInEasing) for entrances/exits.
- **3D tilt**: passive device tilt via `Modifier.tilt3D()` (sensor-driven, never a gesture
  detector inside scrollables). **Floating widgets**: `Modifier.floatingMotion(seed)`.
  **Shimmer loading**: `Modifier.shimmer()` / `SkeletonCard`.
- Ring/border effects animate gradient *phase*, never rotate geometry (energyRing lesson).
- Everything obeys `LocalReducedMotion`; decorative loops only on the focused element.
- No experimental animation APIs; stable `AnimatedContent`/`AnimatedVisibility` only.

## Quality gates for visual work

1. Contrast: body text ≥ 4.5:1 on its surface in BOTH themes.
2. Numbers via `util/Format.kt` — thousands separators, signed pct, mono style.
3. Spacing from `Spacing` tokens; radii from `Radii`; no magic dp on touched code.
4. Screens compose: `ScreenHeader` → hero → sections; one energy-ring "hero" max per screen.
5. Verify with the standard pipeline (devpreview → mirror → CI android + windows) and ask
   the user for a device screenshot for visual sign-off — the sandbox cannot render UI.
