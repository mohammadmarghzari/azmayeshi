using System.Collections.Generic;
using UnityEngine;

namespace FerrumProtocol.Resources
{
    /// <summary>Lightweight registry so harvesters can find the nearest dropoff building without a per-frame scene scan.</summary>
    public static class ResourceDropoffRegistry
    {
        private static readonly List<IResourceDropoff> Dropoffs = new List<IResourceDropoff>();

        public static void Register(IResourceDropoff dropoff) => Dropoffs.Add(dropoff);
        public static void Unregister(IResourceDropoff dropoff) => Dropoffs.Remove(dropoff);

        public static IResourceDropoff FindNearest(Vector3 fromPosition, int ownerPlayerId)
        {
            IResourceDropoff best = null;
            float bestDistSqr = float.MaxValue;

            foreach (var dropoff in Dropoffs)
            {
                if (dropoff.OwnerPlayerId != ownerPlayerId || dropoff.Transform == null)
                {
                    continue;
                }

                float distSqr = (dropoff.Transform.position - fromPosition).sqrMagnitude;
                if (distSqr < bestDistSqr)
                {
                    bestDistSqr = distSqr;
                    best = dropoff;
                }
            }

            return best;
        }
    }
}
