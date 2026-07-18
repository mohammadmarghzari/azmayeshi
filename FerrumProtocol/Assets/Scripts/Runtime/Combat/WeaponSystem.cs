using FerrumProtocol.Core;
using FerrumProtocol.Data;
using UnityEngine;

namespace FerrumProtocol.Combat
{
    /// <summary>
    /// Drives a single weapon mounted on a unit/building: finds/holds a target in range,
    /// respects rate of fire, and either applies hitscan damage instantly or spawns a
    /// <see cref="Projectile"/> for travel-time weapons.
    /// </summary>
    public class WeaponSystem : MonoBehaviour
    {
        [SerializeField] private WeaponDataSO weaponData;
        [SerializeField] private Transform muzzle;
        [SerializeField] private LayerMask targetableLayers = ~0;

        private float _cooldownRemaining;
        private Transform _currentTarget;
        private float _rateOfFireMultiplier = 1f;

        public WeaponDataSO WeaponData => weaponData;
        public bool HasTarget => _currentTarget != null;

        public void SetTarget(Transform target) => _currentTarget = target;

        /// <summary>Instance-local multiplier (e.g. from a temporary ability buff). Never mutate
        /// the shared WeaponDataSO asset itself - it is referenced by every unit of this type.</summary>
        public void SetRateOfFireMultiplier(float multiplier) => _rateOfFireMultiplier = multiplier;

        private void Update()
        {
            if (_cooldownRemaining > 0f)
            {
                _cooldownRemaining -= Time.deltaTime;
            }

            if (_currentTarget == null || weaponData == null)
            {
                return;
            }

            float distance = Vector3.Distance(transform.position, _currentTarget.position);
            if (distance > weaponData.range)
            {
                return;
            }

            var damageable = _currentTarget.GetComponentInParent<IDamageable>();
            if (damageable == null || damageable.IsDead)
            {
                _currentTarget = null;
                return;
            }

            if (_cooldownRemaining <= 0f)
            {
                Fire(damageable);
                float effectiveRate = weaponData.rateOfFire * Mathf.Max(0.01f, _rateOfFireMultiplier);
                _cooldownRemaining = 1f / Mathf.Max(0.01f, effectiveRate);
            }
        }

        private void Fire(IDamageable target)
        {
            if (weaponData.isHitscan || weaponData.projectilePrefab == null)
            {
                target.ApplyDamage(weaponData.damage, weaponData.damageType, this);
                return;
            }

            Vector3 spawnPos = muzzle != null ? muzzle.position : transform.position;
            GameObject instance = ServiceLocator.TryResolve<ObjectPoolManager>(out var pool)
                ? pool.Spawn(weaponData.projectilePrefab, spawnPos, transform.rotation)
                : Instantiate(weaponData.projectilePrefab, spawnPos, transform.rotation);

            if (instance.TryGetComponent<Projectile>(out var projectile))
            {
                projectile.Launch(_currentTarget, _currentTarget.position, weaponData.projectileSpeed,
                    weaponData.damage, weaponData.damageType, weaponData.splashRadius, this);
            }
        }

        /// <summary>Finds the closest valid enemy target in range - call periodically (not every frame) from the owning unit's AI/stance tick.</summary>
        public Transform AcquireClosestTarget(int ownerPlayerId)
        {
            if (weaponData == null)
            {
                return null;
            }

            var colliders = Physics.OverlapSphere(transform.position, weaponData.range, targetableLayers);
            Transform best = null;
            float bestDistSqr = float.MaxValue;

            foreach (var col in colliders)
            {
                var selectable = col.GetComponentInParent<Selection.Selectable>();
                if (selectable == null || selectable.OwnerPlayerId == ownerPlayerId)
                {
                    continue;
                }

                var damageable = col.GetComponentInParent<IDamageable>();
                if (damageable == null || damageable.IsDead)
                {
                    continue;
                }

                float distSqr = (col.transform.position - transform.position).sqrMagnitude;
                if (distSqr < bestDistSqr)
                {
                    bestDistSqr = distSqr;
                    best = col.transform;
                }
            }

            return best;
        }
    }
}
