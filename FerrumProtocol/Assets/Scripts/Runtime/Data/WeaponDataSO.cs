using FerrumProtocol.Combat;
using UnityEngine;

namespace FerrumProtocol.Data
{
    [CreateAssetMenu(menuName = "Ferrum Protocol/Combat/Weapon", fileName = "NewWeapon")]
    public class WeaponDataSO : ScriptableObject
    {
        public string weaponName = "New Weapon";
        public float damage = 10f;
        public DamageType damageType = DamageType.Kinetic;
        public float rateOfFire = 1f; // shots per second
        public float range = 12f;
        public float splashRadius = 0f; // 0 = single target
        public bool isHitscan = true;
        [Tooltip("Only used when isHitscan is false.")]
        public GameObject projectilePrefab;
        [Tooltip("Only used when isHitscan is false.")]
        public float projectileSpeed = 25f;
    }
}
