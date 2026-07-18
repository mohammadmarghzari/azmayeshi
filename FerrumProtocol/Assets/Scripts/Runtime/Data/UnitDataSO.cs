using FerrumProtocol.Combat;
using FerrumProtocol.Units;
using UnityEngine;

namespace FerrumProtocol.Data
{
    public enum UnitCategory
    {
        Infantry,
        Vehicle,
        Aircraft
    }

    [CreateAssetMenu(menuName = "Ferrum Protocol/Unit", fileName = "NewUnit")]
    public class UnitDataSO : ScriptableObject
    {
        [Header("Identity")]
        public string unitName = "New Unit";
        public UnitCategory category = UnitCategory.Infantry;
        public Sprite icon;
        public GameObject prefab;
        [Tooltip("Marks this as a worker/harvester type for AI economy logic and UI grouping.")]
        public bool isHarvester = false;

        [Header("Cost")]
        public int ferriteCost = 100;
        public int voltiumCost = 0;
        public float buildTimeSeconds = 8f;
        public int populationCost = 1;

        [Header("Combat")]
        public float maxHealth = 100f;
        public ArmorType armorType = ArmorType.Unarmored;
        public WeaponDataSO weapon;

        [Header("Movement")]
        public float moveSpeed = 3.5f;
        public float rotationSpeed = 360f;
        public float sightRadius = 15f;

        [Header("Veterancy")]
        public VeterancyRankSO[] veterancyRanks;
    }
}
