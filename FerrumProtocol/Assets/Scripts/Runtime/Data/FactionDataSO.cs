using UnityEngine;

namespace FerrumProtocol.Data
{
    [CreateAssetMenu(menuName = "Ferrum Protocol/Faction", fileName = "NewFaction")]
    public class FactionDataSO : ScriptableObject
    {
        public string factionName = "New Faction";
        [TextArea] public string description;
        public Color primaryColor = Color.white;
        public Color secondaryColor = Color.gray;
        public Sprite emblem;

        public UnitDataSO[] availableUnits;
        public BuildingDataSO[] availableBuildings;
    }
}
