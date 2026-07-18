using System.Collections.Generic;
using UnityEngine;

namespace FerrumProtocol.Combat
{
    /// <summary>
    /// Designer-tunable damage-type vs armor-type multiplier table. If none is assigned,
    /// <see cref="DamageCalculator"/> falls back to a sane built-in default so the combat
    /// math is fully unit-testable without needing a ScriptableObject asset in a project.
    /// </summary>
    [CreateAssetMenu(menuName = "Ferrum Protocol/Combat/Damage Matrix", fileName = "DamageMatrix")]
    public class DamageMatrixSO : ScriptableObject
    {
        [System.Serializable]
        public struct Entry
        {
            public ArmorType armorType;
            public DamageType damageType;
            [Min(0f)] public float multiplier;
        }

        [SerializeField] private List<Entry> entries = new List<Entry>();

        public float GetMultiplier(ArmorType armor, DamageType damage)
        {
            foreach (var entry in entries)
            {
                if (entry.armorType == armor && entry.damageType == damage)
                {
                    return entry.multiplier;
                }
            }

            return DamageCalculator.DefaultMultiplier(armor, damage);
        }
    }
}
