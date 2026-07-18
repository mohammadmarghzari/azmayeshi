# Networking Notes (Mirror)

Phase 1 ships the *shape* of the networking layer, not a finished multiplayer
game (that's Phase 3). Goals for this phase: get the authority model and
message contracts right early, since retrofitting server authority onto a
singleplayer-first codebase is painful.

## Authority Model
- The server/host simulates the match. Period.
- Clients never move their own units locally "for responsiveness" without the
  server also validating the result — that is exactly the kind of client-trust
  bug that becomes a cheat vector later (see `NetworkedUnit.cs` comments).
- Orders flow: Client input → `[Command]` RPC to server → server validates
  (ownership, resource cost, legality) → server mutates state → server
  replicates via `SyncVar`/`ClientRpc`.

## Prediction (Phase 3 target, hooks present now)
`NetworkedUnit.cs` exposes `predictedPosition` fields so a client-side
prediction/reconciliation layer can be added later without changing the
public API: predict locally, reconcile against the authoritative `SyncVar`
position when it arrives, smooth-correct instead of snapping.

## Lobby
`LobbyManager.cs` is intentionally UI-agnostic (pure state: player slots,
ready state, faction pick, map choice) so it can be driven by any UI later
and unit-tested without a network transport running.

## Anti-Cheat Posture
Because the server is authoritative and clients only ever send *intent*
messages (`CommandMessages.cs`), the server can reject anything that doesn't
match its own simulation (e.g. "you don't own that unit", "you can't afford
that building", "that's out of range"). This is the foundation anti-cheat
validation is built on in Phase 7 — it needs to exist from day one, not be
bolted on.
