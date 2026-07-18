using System.Linq;
using FerrumProtocol.Selection;
using UnityEngine;

namespace FerrumProtocol.AI
{
    /// <summary>
    /// Picks build-site locations (ring-sampling around the AI's own base, first
    /// unobstructed point wins) and attack targets (nearest known enemy structure).
    /// A full implementation would track scouted enemy positions under fog of war
    /// rather than reading true positions - flagged here as a Phase 4 fidelity gap.
    /// </summary>
    public class BasePlanner
    {
        [System.Serializable]
        public struct Config
        {
            public LayerMask obstructionLayers;
            public float ringStartRadius;
            public float ringStep;
            public int ringSamples;
            public int maxRings;
        }

        private readonly Config _config;

        public BasePlanner(Config config)
        {
            _config = config;
        }

        public static Config DefaultConfig() => new Config
        {
            obstructionLayers = ~0,
            ringStartRadius = 6f,
            ringStep = 4f,
            ringSamples = 12,
            maxRings = 6
        };

        /// <summary>Ring-samples outward from <paramref name="center"/> for the first clear spot big enough for a building footprint.</summary>
        public Vector3? FindBuildSite(Vector3 center, float footprintRadius = 3f)
        {
            for (int ring = 0; ring < _config.maxRings; ring++)
            {
                float radius = _config.ringStartRadius + ring * _config.ringStep;

                for (int i = 0; i < _config.ringSamples; i++)
                {
                    float angle = (360f / _config.ringSamples) * i * Mathf.Deg2Rad;
                    Vector3 candidate = center + new Vector3(Mathf.Cos(angle), 0f, Mathf.Sin(angle)) * radius;

                    if (!Physics.CheckSphere(candidate + Vector3.up, footprintRadius, _config.obstructionLayers))
                    {
                        return candidate;
                    }
                }
            }

            return null;
        }

        public Vector3? ChooseAttackTarget(int ownPlayerId)
        {
            if (SelectionManager.Instance == null)
            {
                return null;
            }

            // Phase 1 baseline: read true enemy positions. Replace with a scouted/last-known-position
            // model once fog-of-war-aware AI memory is built (Phase 4).
            var enemyBuilding = FindAllSelectables()
                .Where(s => s.OwnerPlayerId != ownPlayerId && s.OwnerPlayerId >= 0)
                .OrderBy(s => System.Guid.NewGuid()) // pick a pseudo-random enemy asset each decision to vary attack targets
                .FirstOrDefault();

            return enemyBuilding != null ? enemyBuilding.Transform.position : (Vector3?)null;
        }

        private System.Collections.Generic.IEnumerable<Selectable> FindAllSelectables()
        {
            for (int i = 0; i < Core.GameConstants.MaxPlayers; i++)
            {
                foreach (var s in SelectionManager.Instance.GetAllOwnedBy(i))
                {
                    yield return s;
                }
            }
        }
    }
}
