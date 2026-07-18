using System.Collections.Generic;
using UnityEngine;

namespace FerrumProtocol.Core
{
    /// <summary>A single prefab's pool of instances. Use via <see cref="ObjectPoolManager"/>, not directly.</summary>
    public class ObjectPool
    {
        private readonly GameObject _prefab;
        private readonly Transform _parent;
        private readonly Stack<GameObject> _inactive = new Stack<GameObject>();

        public ObjectPool(GameObject prefab, Transform parent, int prewarmCount)
        {
            _prefab = prefab;
            _parent = parent;

            for (int i = 0; i < prewarmCount; i++)
            {
                var instance = CreateNew();
                Return(instance);
            }
        }

        private GameObject CreateNew()
        {
            var instance = Object.Instantiate(_prefab, _parent);
            instance.SetActive(false);
            return instance;
        }

        public GameObject Get(Vector3 position, Quaternion rotation)
        {
            GameObject instance = _inactive.Count > 0 ? _inactive.Pop() : CreateNew();
            instance.transform.SetPositionAndRotation(position, rotation);
            instance.SetActive(true);

            foreach (var poolable in instance.GetComponentsInChildren<IPoolable>(true))
            {
                poolable.OnSpawnFromPool();
            }

            return instance;
        }

        public void Return(GameObject instance)
        {
            foreach (var poolable in instance.GetComponentsInChildren<IPoolable>(true))
            {
                poolable.OnReturnToPool();
            }

            instance.SetActive(false);
            instance.transform.SetParent(_parent, worldPositionStays: false);
            _inactive.Push(instance);
        }
    }
}
