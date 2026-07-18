using System;
using System.Collections.Generic;

namespace FerrumProtocol.Save
{
    [Serializable]
    public class SaveData
    {
        public string saveVersion = "0.1.0";
        public string savedAtIso8601;
        public string mapId;
        public List<PlayerSaveEntry> players = new List<PlayerSaveEntry>();
    }

    [Serializable]
    public class PlayerSaveEntry
    {
        public int playerId;
        public int factionIndex;
        public int ferrite;
        public int voltium;
        public int commandCells;
    }

    [Serializable]
    public class MatchStatistics
    {
        public int unitsKilled;
        public int unitsLost;
        public int buildingsLost;
        public int ferriteGathered;
        public int voltiumGathered;
        public float matchDurationSeconds;
    }
}
