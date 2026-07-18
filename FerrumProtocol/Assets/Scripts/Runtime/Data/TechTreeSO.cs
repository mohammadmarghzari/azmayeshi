using UnityEngine;

namespace FerrumProtocol.Data
{
    [CreateAssetMenu(menuName = "Ferrum Protocol/Tech/Tech Tree", fileName = "NewTechTree")]
    public class TechTreeSO : ScriptableObject
    {
        public FactionDataSO faction;
        public UpgradeDataSO[] upgrades;
    }
}
