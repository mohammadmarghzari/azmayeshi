# Examples — how this skill is applied

## Example 1: "یه تب جدید اضافه کن" (add a new tab)
1. Add entry to `nav/Destinations.kt` (Rounded/AutoMirrored icon, Persian label).
2. Create `ui/screens/<Name>Screen.kt`: `ScreenHeader` → hero `Card(highlighted=true)` with
   `HeroMetric` when there's a headline number → content in `Card`s, rows via `AssetRow`,
   lists cascading with `StaggerIn`.
3. Any fetch/compute path holds `UiState<T>` rendered through `StateHost` (skeleton via
   `SkeletonCard`, retry wired, Empty explains the next action in Persian).
4. Route it in `App.kt` `ScreenHost` and add the `BackgroundArt.forDestination` branch
   (the `when` is exhaustive — the compiler will remind you).
5. Pipeline: devpreview compile → mirror both repos → commit/push → CI (android + windows) →
   Persian report.

## Example 2: "این فرمول مالی رو اضافه کن" (new financial metric)
1. Implement in `core/src/main/.../math/` — vectorized over DoubleArray, KDoc with the
   formula and assumptions, validated inputs.
2. Add a unit test beside the existing ones (PortfolioEngineTest pattern) with a hand-checked
   fixture.
3. Surface it in UI via `MetricTile`/`HeroMetric` using `util/Format.kt` formatting —
   remember `%%` escaping in Persian label strings.

## Example 3: "یه کتابخانه جدید لازم داریم" (new dependency)
1. `curl` the Maven Central POM of each candidate version; accept only a version whose
   kotlin-stdlib/Compose pins match Kotlin 2.0.21 / Compose 1.7.1.
2. Add to `gradle/libs.versions.toml` + `composeApp/build.gradle.kts` + (locally)
   `devpreview/build.gradle.kts`.
3. devpreview compile proves JVM resolution; the CI android job is the first real Android
   resolution — say so in the report.

## Example 4: "چرا صفحه خالیه؟" (debugging a blank/broken screen)
Suspects in order: an experimental Compose API (remove it), a composable invoked inside
try/catch, a `%` format crash (check CrashScreen screenshot the user sends), a state read
in the wrong phase. Reproduce reasoning from `CLAUDE.md` → "Known landmines".

## Counter-examples (what this skill forbids)
- Adding Hilt / ViewModel / any Android-only artifact to `commonMain`.
- Importing a charting library instead of extending `charts/`.
- A raw `Button(...)`/`OutlinedTextField(...)` where `AppButton`/`AppTextField` exist.
- Pushing to one repo without mirroring and pulling the other.
- "TODO: handle error" — the UiState framework exists precisely so this never appears.
