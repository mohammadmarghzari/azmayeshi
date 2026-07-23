# Pre-commit checklist — run before EVERY commit

## Correctness
- [ ] `cd /home/user/azmayeshi && gradle :devpreview:compileKotlin` is green
- [ ] No composable invoked inside try/catch; no experimental Compose API introduced
- [ ] Every `%` in a `.format()` Persian label is escaped (`%%`)
- [ ] New `when (Destination)` branches added everywhere the enum is matched
- [ ] State reads for per-frame animation happen in draw/graphicsLayer phase only

## Architecture
- [ ] Math/networking in `core/`, not in composables
- [ ] New UI built from design-system components (AppButton, AppTextField, Card, AssetRow,
      ScreenHeader, StatusBadge…) — no hand-rolled duplicates
- [ ] Fetch/compute paths render through `UiState` + `StateHost` (skeleton/error/empty wired)
- [ ] Android-only code confined to `androidMain` behind a shared seam; no expect/actual
      (devpreview breaks on it)
- [ ] New dependency version POM-checked against Kotlin 2.0.21 / Compose 1.7.1

## UX / fintech polish
- [ ] Persian text, RTL-safe (AutoMirrored icons for directional glyphs)
- [ ] Dark theme first; violet accent for chrome; green/red only for P&L;
      ETH blue / BTC gold only as asset accents
- [ ] Destructive actions use `ButtonStyle.Destructive` (red), primary CTAs get haptics
- [ ] Motion respects `LocalReducedMotion`; lists enter via `StaggerIn`
- [ ] Numbers formatted via `util/Format.kt` (thousands separators, signed pct)

## Data safety
- [ ] Anything the user creates by hand is covered by `AppPersistence` (or the report says
      explicitly that it is session-scoped and why)
- [ ] No secrets added to source; the known IME key debt is not expanded

## Delivery
- [ ] Second repo `git pull`ed, Kotlin trees mirrored, `diff -rq` clean
- [ ] Both repos committed (descriptive message) and pushed
- [ ] CI verified: **android APK job AND windows MSI job**, both repos
- [ ] Persian report sent: what changed, what was decided, what's next
