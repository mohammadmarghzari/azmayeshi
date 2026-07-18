using FerrumProtocol.Combat;
using UnityEngine;

namespace FerrumProtocol.Data
{
    [CreateAssetMenu(menuName = "Ferrum Protocol/Building", fileName = "NewBuilding")]
    public class BuildingDataSO : ScriptableObject
    {
        [Header("Identity")]
        public string buildingName = "New Building";
        public Sprite icon;
        public GameObject prefab;

        [Header("Cost")]
        public int ferriteCost = 500;
        public int voltiumCost = 0;
        public float buildTimeSeconds = 20f;

        [Header("Combat")]
        public float maxHealth = 800f;
        public ArmorType armorType = ArmorType.Structure;

        [Header("Power")]
        public int powerProduced = 0;
        public int powerConsumed = 0;

        [Header("Economy")]
        public bool isResourceDropoff = false;

        [Header("Production")]
        public UnitDataSO[] producibleUnits;
        public BuildingDataSO[] prerequisites;
    }
}
