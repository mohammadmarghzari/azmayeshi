using FerrumProtocol.Combat;
using FerrumProtocol.Data;
using FerrumProtocol.Resources;
using FerrumProtocol.Selection;
using UnityEngine;

namespace FerrumProtocol.Buildings
{
    /// <summary>
    /// A placed building instance: handles the construction ramp-up (health/scale grow over
    /// buildTimeSeconds), registers its power draw/output, and optionally acts as a resource
    /// dropoff. Production/upgrades live in separate sibling components (<see cref="ProductionQueue"/>,
    /// <see cref="RepairSystem"/>) so a turret (no production) doesn't carry queue logic it never uses.
    /// </summary>
    public class BuildingController : MonoBehaviour, IResourceDropoff
    {
        [SerializeField] private BuildingDataSO buildingData;
        [SerializeField] private Health health;
        [SerializeField] private Selectable selectable;
        [SerializeField] private Transform visualRoot;

        public BuildingDataSO Data => buildingData;
        public bool IsConstructed { get; private set; }
        public float ConstructionProgress01 { get; private set; }

        public int OwnerPlayerId => selectable != null ? selectable.OwnerPlayerId : -1;
        public Transform Transform => transform;

        private float _elapsed;
        private bool _powerRegistered;

        private void Awake()
        {
            if (health != null && buildingData != null)
            {
                health.SetMaxHealth(buildingData.maxHealth, healToFull: false);
            }

            if (visualRoot != null)
            {
                visualRoot.localScale = Vector3.one * 0.05f;
            }

            if (buildingData != null && buildingData.isResourceDropoff)
            {
                ResourceDropoffRegistry.Register(this);
            }
        }

        private void OnDestroy()
        {
            if (buildingData != null && buildingData.isResourceDropoff)
            {
                ResourceDropoffRegistry.Unregister(this);
            }

            if (_powerRegistered)
            {
                PowerSystem.Unregister(OwnerPlayerId, buildingData.powerProduced, buildingData.powerConsumed);
            }
        }

        private void Update()
        {
            if (IsConstructed || buildingData == null)
            {
                return;
            }

            float speedMultiplier = PowerSystem.GetSpeedMultiplier(OwnerPlayerId);
            _elapsed += Time.deltaTime * speedMultiplier;
            ConstructionProgress01 = Mathf.Clamp01(_elapsed / Mathf.Max(0.01f, buildingData.buildTimeSeconds));

            if (health != null)
            {
                health.SetMaxHealth(buildingData.maxHealth, healToFull: false);
                if (health.CurrentHealth < buildingData.maxHealth * ConstructionProgress01)
                {
                    health.Heal(buildingData.maxHealth * ConstructionProgress01 - health.CurrentHealth);
                }
            }

            if (visualRoot != null)
            {
                visualRoot.localScale = Vector3.one * Mathf.Lerp(0.05f, 1f, ConstructionProgress01);
            }

            if (ConstructionProgress01 >= 1f)
            {
                CompleteConstruction();
            }
        }

        private void CompleteConstruction()
        {
            IsConstructed = true;

            if (buildingData.powerProduced != 0 || buildingData.powerConsumed != 0)
            {
                PowerSystem.Register(OwnerPlayerId, buildingData.powerProduced, buildingData.powerConsumed);
                _powerRegistered = true;
            }
        }

        public void Deposit(ResourceType type, int amount)
        {
            var economy = Core.GameManager.Instance != null ? Core.GameManager.Instance.GetPlayerEconomy(OwnerPlayerId) : null;
            economy?.Add(type, amount);
        }
    }
}
