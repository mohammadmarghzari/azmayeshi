using System.Collections.Generic;

namespace FerrumProtocol.Buildings
{
    /// <summary>
    /// Tracks aggregate power produced vs. consumed per player. Buildings register their
    /// produced/consumed values on construction complete and unregister on destruction.
    /// When consumption exceeds production ("brownout"), construction/production speed
    /// should be penalized - see <see cref="ProductionQueue"/> and <see cref="ConstructionSite"/>.
    /// </summary>
    public static class PowerSystem
    {
        private class PlayerPower
        {
            public int Produced;
            public int Consumed;
        }

        private static readonly Dictionary<int, PlayerPower> PerPlayer = new Dictionary<int, PlayerPower>();

        private static PlayerPower Get(int playerId)
        {
            if (!PerPlayer.TryGetValue(playerId, out var power))
            {
                power = new PlayerPower();
                PerPlayer[playerId] = power;
            }
            return power;
        }

        public static void Register(int playerId, int produced, int consumed)
        {
            var power = Get(playerId);
            power.Produced += produced;
            power.Consumed += consumed;
        }

        public static void Unregister(int playerId, int produced, int consumed)
        {
            var power = Get(playerId);
            power.Produced -= produced;
            power.Consumed -= consumed;
        }

        public static bool IsBrownout(int playerId)
        {
            var power = Get(playerId);
            return power.Consumed > power.Produced;
        }

        public static (int produced, int consumed) GetBalance(int playerId)
        {
            var power = Get(playerId);
            return (power.Produced, power.Consumed);
        }

        /// <summary>Speed multiplier applied to construction/production while in brownout.</summary>
        public static float GetSpeedMultiplier(int playerId) => IsBrownout(playerId) ? 0.5f : 1f;
    }
}
