using FerrumProtocol.Combat;
using FerrumProtocol.Resources;
using FerrumProtocol.Selection;
using FerrumProtocol.Units;
using UnityEngine;

namespace FerrumProtocol.Buildings
{
    /// <summary>
    /// Attach to a worker unit (alongside <see cref="HarvesterUnit"/>) to let it repair
    /// damaged friendly buildings: moves to the target, then heals it over time at a small
    /// continuous Ferrite cost until full health, the target is destroyed, or ordered to stop.
    /// </summary>
    [RequireComponent(typeof(UnitMotor))]
    public class RepairSystem : MonoBehaviour
    {
        [SerializeField] private float healPerSecond = 20f;
        [SerializeField] private int ferriteCostPerSecond = 2;
        [SerializeField] private float stoppingDistance = 2.5f;
        [SerializeField] private Selectable selectable;

        private UnitMotor _motor;
        private Health _targetHealth;
        private BuildingController _targetBuilding;
        private bool _repairing;

        private void Awake()
        {
            _motor = GetComponent<UnitMotor>();
        }

        public void CommandRepair(Transform targetTransform)
        {
            if (targetTransform == null)
            {
                return;
            }

            _targetBuilding = targetTransform.GetComponentInParent<BuildingController>();
            _targetHealth = targetTransform.GetComponentInParent<Health>();

            if (_targetHealth == null)
            {
                return;
            }

            _repairing = false;
            _motor.MoveTo(targetTransform.position);
        }

        public void Stop()
        {
            _repairing = false;
            _targetHealth = null;
            _targetBuilding = null;
        }

        private void Update()
        {
            if (_targetHealth == null || _targetHealth.IsDead)
            {
                Stop();
                return;
            }

            float distance = Vector3.Distance(transform.position, _targetHealth.transform.position);
            if (distance > stoppingDistance)
            {
                _repairing = false;
                return;
            }

            _motor.Stop();
            _repairing = true;

            if (_targetHealth.CurrentHealth >= _targetHealth.MaxHealth)
            {
                Stop();
                return;
            }

            int ownerId = selectable != null ? selectable.OwnerPlayerId : 0;
            var economy = Core.GameManager.Instance != null ? Core.GameManager.Instance.GetPlayerEconomy(ownerId) : null;
            int costThisFrame = Mathf.CeilToInt(ferriteCostPerSecond * Time.deltaTime);

            if (economy != null && costThisFrame > 0 && !economy.TrySpend(costThisFrame, 0))
            {
                return; // out of Ferrite - pause repair rather than stop, in case income resumes
            }

            _targetHealth.Heal(healPerSecond * Time.deltaTime);
        }
    }
}
