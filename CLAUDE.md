# CLAUDE.md — Portfolio360 (azmayeshi)

**Default project skill: `.claude/skills/portfolio360-master/` — it applies to every task in
this repository unless the user explicitly overrides it.** Read its `SKILL.md` before
changing anything; its companion `CLAUDE.md` holds deep context (module map, known
landmines, roadmap), `examples.md` shows worked patterns, and `checklist.md` gates every
commit.

Quick facts the skill expands on:
- Kotlin Multiplatform (Android + Windows/desktop from one `commonMain`) — NOT plain
  Android. No Hilt, no ViewModel artifacts, no expect/actual (the local `devpreview`
  compile sandbox can't handle it).
- Mirrored repo: `mohammadmarghzari/sarmaye-portfolio-tool@main` must receive identical
  Kotlin changes (pull it first — the user pushes docs there directly).
- Verify every change: `gradle :devpreview:compileKotlin` locally, then CI's **android**
  and **windows desktop** jobs on both repos.
- The user is Persian-speaking and non-technical: report in Persian, own the technical
  decisions, state trade-offs honestly.
