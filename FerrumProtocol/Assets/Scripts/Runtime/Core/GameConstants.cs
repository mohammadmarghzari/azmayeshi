namespace FerrumProtocol.Core
{
    /// <summary>Tunable-but-rarely-changed constants shared across systems. Balance numbers belong
    /// in ScriptableObject data assets (Data/*.cs), not here — this file is for structural constants.</summary>
    public static class GameConstants
    {
        public const int MaxPlayers = 8;
        public const int MaxControlGroups = 10; // keys 0-9
        public const float DefaultVisionRadius = 15f;
        public const int FogGridCellsPerWorldUnit = 1; // 1 fog cell per world unit on X/Z
        public const string DefaultSaveSlot = "autosave";
    }
}
