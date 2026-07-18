# Ferrum Protocol

An original, non-commercial-IP Real-Time Strategy game built in Unity 6 (URP),
inspired by the pacing and depth of classic RTS titles - all-new factions,
world, art direction, and code.

**Start here:** [`Documentation/HANDBOOK_FA.md`](Documentation/HANDBOOK_FA.md)
(Persian, zero-assumed-knowledge setup guide) or [`Documentation/ROADMAP.md`](Documentation/ROADMAP.md)
for the phase-by-phase build plan.

## Status

Phase 1 (core systems foundation) - see `Documentation/ROADMAP.md` for what's
implemented and what's next. This is not yet a finished game; it's a real,
compiling (pending first open in Unity), testable foundation every subsequent
phase builds on.

## Layout

```
FerrumProtocol/
  Assets/
    Scripts/Runtime/     gameplay code (Core, CameraSystem, Selection, Units,
                          Buildings, Resources, Combat, FogOfWar, Minimap, AI,
                          Networking, Data, Save, Utils)
    Scripts/Editor/       SceneBootstrapper + PrefabGenerator (one-click prototype)
    Scripts/Tests/EditMode/  NUnit tests for pure-logic systems
    Shaders/              URP shaders (team color, fog of war overlay)
    Scenes/, Prefabs/, Audio/, UI/  (populated by the Editor bootstrap tool)
  Documentation/          GDD, roadmap, architecture, networking notes, handbook
  Packages/manifest.json  package dependencies (Mirror, Input System, URP, ...)
```

## Quick Start (after following the handbook's Unity project setup)

In the Unity Editor: **Ferrum Protocol → Build Prototype Scene**, then press Play.

## License / Originality

All names, factions, art direction, and lore are original to this project.
No assets, code, or names are copied from any existing commercial game.
