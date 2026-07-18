# Roadmap

Realistic framing: a full commercial AAA RTS (campaign, ranked multiplayer,
map editor, Steam Workshop, anti-cheat, cloud saves...) is normally a
multi-year effort for a studio of dozens. This roadmap breaks that into
phases so a single-developer project can make continuous, compounding
progress. Each phase should end compiling, tested, and committed before the
next starts.

## Phase 1 — Core Systems Foundation (THIS COMMIT)
- Project scaffold (Assets/Editor/Tests/Docs/CI)
- Service locator + game manager skeleton
- RTS camera (zoom/rotate/edge-scroll/pan/smoothing)
- Input System actions asset
- Selection (single/box/control groups)
- Unit movement (NavMesh), stances (attack-move/patrol/guard/hold), formations, waypoints
- Buildings: placement, construction, production queue, power, upgrades, repair
- Resources: two primary + one strategic, harvester loop
- Combat: health, armor/damage-type matrix, weapons, projectiles, veterancy, abilities (framework)
- Fog of war (grid-based), minimap (framework)
- AI skeleton (economy/combat/base-planner/difficulty)
- Networking skeleton (Mirror-based, server-authoritative structure)
- Save system skeleton (local + pluggable cloud provider interface)
- EditMode tests for pure-logic systems
- Editor tool to auto-build a playable prototype scene + placeholder prefabs
- GitHub Actions CI (Unity test runner)

## Phase 2 — Playable Prototype Loop
- Open in Unity, fix first-compile issues, run the bootstrapped prototype scene
- Wire real prefabs (primitive placeholders) into a 1v1 skirmish-able loop
- Balance pass v0 on the 2 launch factions' baseline units
- Basic in-game HUD (resources, minimap, command card, build queue UI)
- Basic AI opponent that can be beaten and can beat an idle player

## Phase 3 — Multiplayer Alpha
- Mirror transport wired end-to-end: lobby → match → server-authoritative sim
- Client prediction + reconciliation for movement
- Reconnect handling, spectator mode
- Dedicated server build target

## Phase 4 — Content Expansion
- Third faction (Ashfall Remnant, asymmetric mechanics)
- Full unit/building rosters per GDD
- Veterancy visuals, ability VFX hooks, upgrade tech trees
- Campaign mission framework + first mission

## Phase 5 — Meta Systems
- Ranked matchmaking, replays, statistics, achievements
- Cloud save provider (Steamworks.NET or platform-appropriate)
- Map editor (terrain painting, object placement, triggers, export)
- Steam Workshop export pipeline

## Phase 6 — Polish & Performance
- LOD, occlusion culling, GPU instancing pass
- Multithreaded pathfinding/economy ticks (Unity Job System/Burst)
- Full art/audio pass (original assets only)
- Platform builds: Windows, Linux, Android

## Phase 7 — Ship Hardening
- Anti-cheat pass on server-authoritative validation
- Full regression test suite, soak tests for long matches
- Store page, build pipelines, release candidate
