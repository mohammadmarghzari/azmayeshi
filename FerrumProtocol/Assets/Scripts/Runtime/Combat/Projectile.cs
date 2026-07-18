using FerrumProtocol.Core;
using UnityEngine;

namespace FerrumProtocol.Combat
{
    /// <summary>
    /// Simple homing/ballistic projectile. Spawned and returned via <see cref="ObjectPoolManager"/>
    /// so bursts of fire don't allocate. Splash damage (if any) applies to every IDamageable in
    /// range on impact via an OverlapSphere.
    /// </summary>
    public class Projectile : MonoBehaviour, IPoolable
    {
        [SerializeField] private LayerMask splashLayers = ~0;

        private Transform _target;
        private Vector3 _fallbackTargetPoint;
        private float _speed;
        private float _damage;
        private DamageType _damageType;
        private float _splashRadius;
        private object _source;

        public void Launch(Transform target, Vector3 targetPointFallback, float speed, float damage, DamageType damageType, float splashRadius, object source)
        {
            _target = target;
            _fallbackTargetPoint = targetPointFallback;
            _speed = speed;
            _damage = damage;
            _damageType = damageType;
            _splashRadius = splashRadius;
            _source = source;
        }

        private void Update()
        {
            Vector3 destination = _target != null ? _target.position : _fallbackTargetPoint;
            Vector3 toTarget = destination - transform.position;
            float step = _speed * Time.deltaTime;

            if (toTarget.magnitude <= step)
            {
                Impact(destination);
                return;
            }

            transform.position += toTarget.normalized * step;
            if (toTarget.sqrMagnitude > 0.0001f)
            {
                transform.rotation = Quaternion.LookRotation(toTarget);
            }
        }

        private void Impact(Vector3 point)
        {
            if (_splashRadius > 0f)
            {
                var hits = Physics.OverlapSphere(point, _splashRadius, splashLayers);
                foreach (var hit in hits)
                {
                    var damageable = hit.GetComponentInParent<IDamageable>();
                    damageable?.ApplyDamage(_damage, _damageType, _source);
                }
            }
            else if (_target != null)
            {
                var damageable = _target.GetComponentInParent<IDamageable>();
                damageable?.ApplyDamage(_damage, _damageType, _source);
            }

            if (ServiceLocator.TryResolve<ObjectPoolManager>(out var pool))
            {
                pool.Despawn(gameObject);
            }
            else
            {
                Destroy(gameObject);
            }
        }

        public void OnSpawnFromPool()
        {
        }

        public void OnReturnToPool()
        {
            _target = null;
            _source = null;
        }
    }
}
