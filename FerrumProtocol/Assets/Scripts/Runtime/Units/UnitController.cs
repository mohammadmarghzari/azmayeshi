using FerrumProtocol.Combat;
using FerrumProtocol.Core;
using FerrumProtocol.Data;
using FerrumProtocol.Selection;
using UnityEngine;

namespace FerrumProtocol.Units
{
    /// <summary>
    /// Top-level unit component that wires a <see cref="UnitDataSO"/>'s stats into this
    /// instance's Health/WeaponSystem/UnitMotor, and turns incoming <see cref="UnitCommand"/>s
    /// into motor/stance/weapon calls. Attach alongside Health, WeaponSystem, UnitMotor,
    /// StanceController, Selectable, NavMeshAgent.
    /// </summary>
    [RequireComponent(typeof(UnitMotor))]
    [RequireComponent(typeof(StanceController))]
    public class UnitController : MonoBehaviour, ICommandable
    {
        [SerializeField] private UnitDataSO unitData;
        [SerializeField] private Health health;
        [SerializeField] private WeaponSystem weapon;
        [SerializeField] private Selectable selectable;
        [SerializeField] private Veterancy veterancy;

        private UnitMotor _motor;
        private StanceController _stance;
        private readonly WaypointQueue _waypoints = new WaypointQueue();

        public UnitDataSO Data => unitData;

        private void Awake()
        {
            _motor = GetComponent<UnitMotor>();
            _stance = GetComponent<StanceController>();

            if (unitData != null)
            {
                _motor.Configure(unitData.moveSpeed, unitData.rotationSpeed);
                if (health != null)
                {
                    health.SetMaxHealth(unitData.maxHealth, healToFull: true);
                }
            }

            if (health != null)
            {
                health.OnDeath += HandleDeath;
            }
        }

        private void OnDestroy()
        {
            if (health != null)
            {
                health.OnDeath -= HandleDeath;
            }
        }

        public void IssueCommand(UnitCommand command)
        {
            if (!command.Queued)
            {
                // A fresh (non-shift) order replaces whatever this unit was doing, including
                // any orders still waiting in the queue.
                _waypoints.Clear();
                Execute(command);
                return;
            }

            bool isIdle = !_motor.IsMoving && _waypoints.Count == 0;
            if (isIdle)
            {
                // Nothing currently running and nothing queued - run this one immediately
                // instead of parking it in the queue where nothing would ever dequeue it.
                Execute(command);
            }
            else
            {
                _waypoints.SetOrder(command);
            }
        }

        private void Execute(UnitCommand command)
        {
            switch (command.Type)
            {
                case CommandType.Move:
                    _stance.SetAggressive();
                    _motor.MoveTo(command.TargetPoint);
                    break;
                case CommandType.AttackMove:
                    _stance.SetAttackMove(command.TargetPoint);
                    break;
                case CommandType.Attack:
                    _stance.SetAggressive();
                    if (weapon != null && command.TargetTransform != null)
                    {
                        weapon.SetTarget(command.TargetTransform);
                    }
                    _motor.MoveTo(command.TargetPoint);
                    break;
                case CommandType.Guard:
                    _stance.SetGuard(command.TargetPoint);
                    break;
                case CommandType.HoldPosition:
                    _stance.SetHoldPosition();
                    break;
                case CommandType.Patrol:
                    _stance.SetPatrol(new[] { transform.position, command.TargetPoint });
                    break;
                case CommandType.Stop:
                    _waypoints.Clear();
                    _stance.Stop();
                    break;
                case CommandType.Harvest:
                    GetComponent<Resources.HarvesterUnit>()?.CommandHarvest(command.TargetTransform);
                    break;
                case CommandType.Repair:
                    GetComponent<Buildings.RepairSystem>()?.CommandRepair(command.TargetTransform);
                    break;
            }
        }

        private void Update()
        {
            if (!_motor.IsMoving && _waypoints.Count > 0 && _waypoints.TryDequeueNext(out var next))
            {
                Execute(next);
            }
        }

        private void HandleDeath(object source)
        {
            if (source is WeaponSystem killerWeapon)
            {
                var killerVeterancy = killerWeapon.GetComponentInParent<Veterancy>();
                killerVeterancy?.OnKillCredit(10);
            }

            _motor.Stop();
            enabled = false;
            // Phase 2: play death VFX/ragdoll, then return to pool or destroy after a delay.
        }
    }
}
