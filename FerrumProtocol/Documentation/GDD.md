# FERRUM PROTOCOL — Game Design Document (v0.1, Phase 1)

Original working title: **Ferrum Protocol**. Entirely original IP — no names, art,
lore, sounds, or code derived from any existing commercial RTS. Any resemblance to
real or fictional military hardware is coincidental and generic ("tank", "gunship",
"rifle infantry" are genre-standard nouns, not copied designs).

## 1. Premise

Twenty years after a global energy collapse ("The Blackout"), three power blocs
fight for control of the last reserves of **Voltium**, a synthetic energy crystal
that replaced fossil fuels. The war is fought by corporate-military combines,
salvage militias, and autonomous drone armies across reclaimed industrial
continents.

## 2. Factions (Phase 1 design — art TBD, placeholders only)

### 2.1 Aurora Concord
High-tech directed-energy and drone faction. Fragile units, powerful tech,
strong air/ranged options. Color identity: cyan / white / cobalt.

### 2.2 Kessler Dominion
Heavy industrial armor faction. Slow, tough, hard-hitting. Strong base defense
and artillery. Color identity: rust-orange / gunmetal grey.

### 2.3 Ashfall Remnant (Phase 3+, asymmetric third faction)
Guerrilla / salvage faction. Cheap, disposable, stealthy, improvised weapons,
no conventional base — mobile "camps" instead of fixed structures. Color
identity: sand-yellow / black.

## 3. Economy

- **Ferrite** (Primary resource #1): common ore, base income, gathered from
  Ferrite Deposits by Harvester units, refined at a Refinery building.
- **Voltium** (Primary resource #2): energy crystal, rarer nodes, required for
  tier-2/3 units, upgrades and abilities.
- **Command Cells** (Secondary strategic resource): very rare, fixed, non-replenishing
  map pickups, heavily contested. Required for elite units and superweapons.
  Analogous role: a scarce "victory-point" resource that forces map control fights.

## 4. Unit Roster (Phase 1 baseline, expand in later phases)

Each faction ships with, at minimum:
1. Harvester (economy)
2. Rifle Infantry (basic anti-infantry)
3. Light Vehicle (fast, anti-vehicle/scout)
4. Heavy Vehicle (slow, high armor, main damage dealer)
5. Support Aircraft (recon / light strike)
6. Faction Special Unit (unique mechanic per faction)

## 5. Buildings (Phase 1 baseline)

Command Center, Power Plant, Refinery, Barracks, Vehicle Factory, Airfield,
Defense Turret, Radar/Tech Center, Superweapon (late game, Phase 4+).

## 6. Core Loop

Scout → Expand economy → Tech up → Build army → Attack/Defend map control
points (Command Cell nodes) → Leverage veterancy and abilities → Win by
elimination or objective (Skirmish/Campaign modes may vary).

## 7. Damage / Armor Matrix (Phase 1 baseline, tune later)

Damage types: `Kinetic, Explosive, Energy, Piercing, Incendiary`
Armor types: `Unarmored, LightArmor, HeavyArmor, Structure, Aircraft`

See `ArmorDataSO` / `DamageCalculator.cs` for the live multiplier table — this
document is the design intent, the code is the source of truth once implemented.

## 8. Game Modes (target, built incrementally)

Campaign (per-faction, mission-based), Skirmish (vs AI), LAN, Online
(ranked/custom), Co-op (vs AI), Survival (wave defense), Sandbox (no fail state).

## 9. Non-Goals for Phase 1

No final art, no final audio, no balance pass, no full campaign script. Phase 1
is systems + placeholders so every mechanic is *playable* end-to-end before any
art investment is made.
