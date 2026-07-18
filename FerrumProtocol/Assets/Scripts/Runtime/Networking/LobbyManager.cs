using System.Collections.Generic;

namespace FerrumProtocol.Networking
{
    public enum LobbyPhase
    {
        WaitingForPlayers,
        AllReady,
        Starting
    }

    public class LobbySlot
    {
        public int PlayerId;
        public string DisplayName;
        public int FactionIndex = -1;
        public bool IsReady;
        public bool IsAi;
    }

    /// <summary>
    /// Pure C# lobby state machine - deliberately has zero Mirror/UI dependencies so it can be
    /// driven by any transport or UI and unit-tested without a network session running. A
    /// networked wrapper (Phase 3) reads/writes this via PlayerSession SyncVars.
    /// </summary>
    public class LobbyManager
    {
        private readonly Dictionary<int, LobbySlot> _slots = new Dictionary<int, LobbySlot>();
        public LobbyPhase Phase { get; private set; } = LobbyPhase.WaitingForPlayers;
        public string SelectedMapId { get; private set; } = "prototype";

        public IReadOnlyCollection<LobbySlot> Slots => _slots.Values;

        public LobbySlot AddPlayer(int playerId, string displayName, bool isAi = false)
        {
            var slot = new LobbySlot { PlayerId = playerId, DisplayName = displayName, IsAi = isAi };
            _slots[playerId] = slot;
            Phase = LobbyPhase.WaitingForPlayers;
            return slot;
        }

        public void RemovePlayer(int playerId)
        {
            _slots.Remove(playerId);
            Phase = LobbyPhase.WaitingForPlayers;
        }

        public void SetFaction(int playerId, int factionIndex)
        {
            if (_slots.TryGetValue(playerId, out var slot))
            {
                slot.FactionIndex = factionIndex;
            }
        }

        public void SetReady(int playerId, bool ready)
        {
            if (!_slots.TryGetValue(playerId, out var slot))
            {
                return;
            }

            slot.IsReady = ready;
            RecomputePhase();
        }

        public void SetMap(string mapId) => SelectedMapId = mapId;

        private void RecomputePhase()
        {
            if (_slots.Count == 0)
            {
                Phase = LobbyPhase.WaitingForPlayers;
                return;
            }

            bool allReady = true;
            foreach (var slot in _slots.Values)
            {
                if (!slot.IsReady && !slot.IsAi)
                {
                    allReady = false;
                    break;
                }
            }

            Phase = allReady ? LobbyPhase.AllReady : LobbyPhase.WaitingForPlayers;
        }

        public bool TryBeginStarting()
        {
            if (Phase != LobbyPhase.AllReady)
            {
                return false;
            }

            Phase = LobbyPhase.Starting;
            return true;
        }
    }
}
