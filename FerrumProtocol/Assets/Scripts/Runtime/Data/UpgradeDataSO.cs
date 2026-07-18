using UnityEngine;

namespace FerrumProtocol.Data
{
    [CreateAssetMenu(menuName = "Ferrum Protocol/Tech/Upgrade", fileName = "NewUpgrade")]
    public class UpgradeDataSO : ScriptableObject
    {
        public string upgradeName = "New Upgrade";
        public Sprite icon;
        public int ferriteCost = 500;
        public int voltiumCost = 200;
        public float researchTimeSeconds = 30f;
        public UpgradeDataSO[] prerequisites;

        [Header("Effect (applied by gameplay code reading UpgradeManager.IsResearched - see ARCHITECTURE.md)")]
        public UnitDataSO[] affectedUnits;
        public float healthMultiplier = 1f;
        public float damageMultiplier = 1f;
    }
}
