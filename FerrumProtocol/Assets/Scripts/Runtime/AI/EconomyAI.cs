using System.Linq;
using FerrumProtocol.Buildings;
using FerrumProtocol.Data;
using FerrumProtocol.Resources;
using FerrumProtocol.Selection;
using UnityEngine;

namespace FerrumProtocol.AI
{
    /// <summary>
    /// Heuristic economy management for one AI player: keeps harvester count topped up to the
    /// difficulty's target, and sends freshly built harvesters to the nearest un-depleted
    /// resource node automatically. This is intentionally simple (Phase 1 baseline) - smarter
    /// build-order/expansion timing decisions belong in <see cref="BasePlanner"/>/Phase 4.
    /// </summary>
    public class EconomyAI
    {
        private readonly int _playerId;
        private readonly AIDifficultyDataSO _difficulty;

        public EconomyAI(int playerId, AIDifficultyDataSO difficulty)
        {
            _playerId = playerId;
            _difficulty = difficulty;
        }

        public void Tick(PlayerEconomy economy)
        {
            if (economy == null || SelectionManager.Instance == null)
            {
                return;
            }

            var owned = SelectionManager.Instance.GetAllOwnedBy(_playerId).ToList();
            int harvesterCount = owned.Count(s => s.GetComponent<HarvesterUnit>() != null);

            AssignIdleHarvesters(owned);

            if (harvesterCount >= _difficulty.targetHarvesterCount)
            {
                return;
            }

            var productionQueue = owned
                .Select(s => s.GetComponent<BuildingController>())
                .Where(b => b != null && b.IsConstructed)
                .Select(b => b.GetComponent<ProductionQueue>())
                .FirstOrDefault(q => q != null);

            var harvesterData = owned
                .Select(s => s.GetComponent<BuildingController>())
                .FirstOrDefault(b => b != null && b.Data != null && b.Data.producibleUnits != null)
                ?.Data.producibleUnits.FirstOrDefault(u => u != null && u.isHarvester);

            if (productionQueue != null && harvesterData != null)
            {
                productionQueue.TryEnqueue(harvesterData, economy);
            }
        }

        private void AssignIdleHarvesters(System.Collections.Generic.List<Selectable> owned)
        {
            ResourceNode fallbackNode = null;

            foreach (var s in owned)
            {
                var harvester = s.GetComponent<HarvesterUnit>();
                if (harvester == null || harvester.State != HarvesterState.Idle)
                {
                    continue;
                }

                if (fallbackNode == null)
                {
                    fallbackNode = Object.FindObjectsByType<ResourceNode>(FindObjectsSortMode.None)
                        .FirstOrDefault(n => !n.IsDepleted);
                }

                if (fallbackNode != null)
                {
                    harvester.CommandHarvest(fallbackNode.transform);
                }
            }
        }
    }
}
