using System.Collections.Generic;
using FerrumProtocol.Data;
using FerrumProtocol.Resources;
using UnityEngine;

namespace FerrumProtocol.Buildings
{
    /// <summary>
    /// Per-player research state. Combat/production code should query
    /// <see cref="IsResearched"/> to apply an upgrade's multipliers (e.g. WeaponSystem could
    /// check this once per shot) - kept as a single source of truth rather than pushing
    /// stat changes out to every affected unit instance when research completes.
    /// </summary>
    public class UpgradeManager : MonoBehaviour
    {
        [SerializeField] private int playerId;

        private readonly HashSet<UpgradeDataSO> _researched = new HashSet<UpgradeDataSO>();
        private readonly Dictionary<UpgradeDataSO, float> _inProgress = new Dictionary<UpgradeDataSO, float>();

        public event System.Action<UpgradeDataSO> OnResearchCompleted;

        public bool IsResearched(UpgradeDataSO upgrade) => upgrade != null && _researched.Contains(upgrade);

        public bool TryBeginResearch(UpgradeDataSO upgrade, PlayerEconomy economy)
        {
            if (upgrade == null || economy == null || IsResearched(upgrade) || _inProgress.ContainsKey(upgrade))
            {
                return false;
            }

            if (upgrade.prerequisites != null)
            {
                foreach (var prereq in upgrade.prerequisites)
                {
                    if (!IsResearched(prereq))
                    {
                        return false;
                    }
                }
            }

            if (!economy.TrySpend(upgrade.ferriteCost, upgrade.voltiumCost))
            {
                return false;
            }

            _inProgress[upgrade] = 0f;
            return true;
        }

        private void Update()
        {
            if (_inProgress.Count == 0)
            {
                return;
            }

            List<UpgradeDataSO> completed = null;

            foreach (var kvp in _inProgress)
            {
                float elapsed = kvp.Value + Time.deltaTime * PowerSystem.GetSpeedMultiplier(playerId);
                _inProgress[kvp.Key] = elapsed;

                if (elapsed >= kvp.Key.researchTimeSeconds)
                {
                    (completed ??= new List<UpgradeDataSO>()).Add(kvp.Key);
                }
            }

            if (completed == null)
            {
                return;
            }

            foreach (var upgrade in completed)
            {
                _inProgress.Remove(upgrade);
                _researched.Add(upgrade);
                OnResearchCompleted?.Invoke(upgrade);
            }
        }
    }
}
