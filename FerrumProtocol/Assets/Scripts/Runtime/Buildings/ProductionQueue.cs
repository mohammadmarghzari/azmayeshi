using System.Collections.Generic;
using FerrumProtocol.Core;
using FerrumProtocol.Data;
using UnityEngine;

namespace FerrumProtocol.Buildings
{
    /// <summary>
    /// FIFO unit production queue for a single building (Barracks, Vehicle Factory, Airfield...).
    /// Cost is deducted when an item is enqueued (classic RTS behaviour: queuing 3 units commits
    /// the resources immediately, refunded if cancelled) so the player can't oversell their economy.
    /// </summary>
    public class ProductionQueue : MonoBehaviour
    {
        private struct QueueItem
        {
            public UnitDataSO Data;
            public float Elapsed;
        }

        [SerializeField] private BuildingController building;
        [SerializeField] private Transform rallyPoint;
        [SerializeField] private Transform spawnPoint;

        private readonly Queue<QueueItem> _queue = new Queue<QueueItem>();
        public int Count => _queue.Count;

        public event System.Action<UnitDataSO> OnUnitCompleted;

        public bool TryEnqueue(UnitDataSO unitData, Resources.PlayerEconomy economy)
        {
            if (unitData == null || economy == null || building == null || !building.IsConstructed)
            {
                return false;
            }

            if (!economy.TrySpend(unitData.ferriteCost, unitData.voltiumCost))
            {
                return false;
            }

            _queue.Enqueue(new QueueItem { Data = unitData, Elapsed = 0f });
            return true;
        }

        public void CancelLast(Resources.PlayerEconomy economy)
        {
            if (_queue.Count == 0)
            {
                return;
            }

            // Queue doesn't support removing the last element cheaply; rebuild without it.
            var items = new List<QueueItem>(_queue);
            var removed = items[items.Count - 1];
            items.RemoveAt(items.Count - 1);

            _queue.Clear();
            foreach (var item in items)
            {
                _queue.Enqueue(item);
            }

            economy?.Refund(removed.Data.ferriteCost, removed.Data.voltiumCost);
        }

        private void Update()
        {
            if (_queue.Count == 0 || building == null || !building.IsConstructed)
            {
                return;
            }

            var item = _queue.Peek();
            float speedMultiplier = PowerSystem.GetSpeedMultiplier(building.OwnerPlayerId);
            item.Elapsed += Time.deltaTime * speedMultiplier;

            if (item.Elapsed >= item.Data.buildTimeSeconds)
            {
                _queue.Dequeue();
                SpawnUnit(item.Data);
            }
            else
            {
                // structs are value types - write the updated elapsed back via dequeue/requeue.
                _queue.Dequeue();
                _queue.Enqueue(new QueueItem { Data = item.Data, Elapsed = item.Elapsed });
            }
        }

        private void SpawnUnit(UnitDataSO unitData)
        {
            if (unitData.prefab == null)
            {
                Debug.LogWarning($"UnitDataSO '{unitData.unitName}' has no prefab assigned - cannot spawn.");
                return;
            }

            Vector3 spawnPos = spawnPoint != null ? spawnPoint.position : transform.position;
            Quaternion spawnRot = spawnPoint != null ? spawnPoint.rotation : transform.rotation;
            var instance = Instantiate(unitData.prefab, spawnPos, spawnRot);

            var newSelectable = instance.GetComponent<Selection.Selectable>();
            if (newSelectable != null && building != null)
            {
                newSelectable.SetOwner(building.OwnerPlayerId);
            }

            if (rallyPoint != null && instance.TryGetComponent<Units.UnitMotor>(out var motor))
            {
                motor.MoveTo(rallyPoint.position);
            }

            OnUnitCompleted?.Invoke(unitData);
        }
    }
}
