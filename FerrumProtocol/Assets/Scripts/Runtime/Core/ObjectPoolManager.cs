using System.Collections.Generic;
using UnityEngine;

namespace FerrumProtocol.Core
{
    /// <summary>
    /// Central place to spawn/despawn pooled objects (projectiles, VFX, short-lived UI).
    /// Never call Instantiate/Destroy directly for anything that spawns frequently -
    /// route it through here so the pool amortizes allocation cost.
    /// </summary>
    public class ObjectPoolManager : MonoBehaviour
    {
        [SerializeField] private int defaultPrewarmCount = 16;

        private readonly Dictionary<GameObject, ObjectPool> _pools = new Dictionary<GameObject, ObjectPool>();
        private readonly Dictionary<GameObject, GameObject> _instanceToPrefab = new Dictionary<GameObject, GameObject>();

        private void Awake()
        {
            ServiceLocator.Register(this);
        }

        private void OnDestroy()
        {
            ServiceLocator.Unregister<ObjectPoolManager>();
        }

        public GameObject Spawn(GameObject prefab, Vector3 position, Quaternion rotation)
        {
            if (!_pools.TryGetValue(prefab, out var pool))
            {
                var poolParent = new GameObject($"Pool_{prefab.name}").transform;
                poolParent.SetParent(transform, worldPositionStays: false);
                pool = new ObjectPool(prefab, poolParent, defaultPrewarmCount);
                _pools[prefab] = pool;
            }

            var instance = pool.Get(position, rotation);
            _instanceToPrefab[instance] = prefab;
            return instance;
        }

        public void Despawn(GameObject instance)
        {
            if (_instanceToPrefab.TryGetValue(instance, out var prefab) && _pools.TryGetValue(prefab, out var pool))
            {
                pool.Return(instance);
            }
            else
            {
                // Not a pooled instance - fall back to destroying it so callers don't leak.
                Destroy(instance);
            }
        }
    }
}
