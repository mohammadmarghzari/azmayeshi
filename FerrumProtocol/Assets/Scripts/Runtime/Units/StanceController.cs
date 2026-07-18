using System.Collections.Generic;
using FerrumProtocol.Combat;
using FerrumProtocol.Selection;
using UnityEngine;

namespace FerrumProtocol.Units
{
    public enum Stance
    {
        /// <summary>Default: stay put (or keep moving toward the last order) but auto-engage anything that wanders into weapon range.</summary>
        Aggressive,
        AttackMove,
        Patrol,
        Guard,
        HoldPosition
    }

    /// <summary>
    /// Reconciles a unit's current order (stance) with its motor and weapon each tick:
    /// acquires targets, decides whether to chase/hold/return, and drives patrol looping.
    /// Target re-acquisition runs on a timer rather than every frame to keep the
    /// OverlapSphere cost bounded when hundreds of units are on screen.
    /// </summary>
    [RequireComponent(typeof(UnitMotor))]
    public class StanceController : MonoBehaviour
    {
        [SerializeField] private WeaponSystem weapon;
        [SerializeField] private Selectable selectable;
        [SerializeField] private float targetScanInterval = 0.5f;
        [SerializeField] private float leashDistance = 20f; // how far Guard/HoldPosition units chase before returning

        private UnitMotor _motor;
        private Stance _stance = Stance.Aggressive;
        private Vector3 _anchorPoint;
        private Vector3 _orderDestination;
        private readonly List<Vector3> _patrolPoints = new List<Vector3>();
        private int _patrolIndex;
        private float _scanTimer;

        public Stance CurrentStance => _stance;

        private void Awake()
        {
            _motor = GetComponent<UnitMotor>();
            _anchorPoint = transform.position;
        }

        public void SetAggressive()
        {
            _stance = Stance.Aggressive;
        }

        public void SetAttackMove(Vector3 destination)
        {
            _stance = Stance.AttackMove;
            _orderDestination = destination;
            _motor.MoveTo(destination);
        }

        public void SetGuard(Vector3 point)
        {
            _stance = Stance.Guard;
            _anchorPoint = point;
        }

        public void SetHoldPosition()
        {
            _stance = Stance.HoldPosition;
            _anchorPoint = transform.position;
            _motor.Stop();
        }

        public void SetPatrol(IReadOnlyList<Vector3> points)
        {
            if (points == null || points.Count == 0)
            {
                return;
            }

            _stance = Stance.Patrol;
            _patrolPoints.Clear();
            _patrolPoints.AddRange(points);
            _patrolIndex = 0;
            _motor.MoveTo(_patrolPoints[0]);
        }

        public void Stop()
        {
            _stance = Stance.Aggressive;
            _anchorPoint = transform.position;
            _motor.Stop();
        }

        private void Update()
        {
            _scanTimer -= Time.deltaTime;
            if (_scanTimer <= 0f && weapon != null && !weapon.HasTarget)
            {
                _scanTimer = targetScanInterval;
                int ownerId = selectable != null ? selectable.OwnerPlayerId : -1;
                var target = weapon.AcquireClosestTarget(ownerId);
                if (target != null)
                {
                    weapon.SetTarget(target);
                }
            }

            switch (_stance)
            {
                case Stance.Patrol:
                    TickPatrol();
                    break;
                case Stance.Guard:
                case Stance.HoldPosition:
                    TickLeashed();
                    break;
                case Stance.AttackMove:
                case Stance.Aggressive:
                default:
                    break; // motor already has its destination; weapon engages opportunistically
            }
        }

        private void TickPatrol()
        {
            if (_patrolPoints.Count == 0 || _motor.IsMoving)
            {
                return;
            }

            _patrolIndex = (_patrolIndex + 1) % _patrolPoints.Count;
            _motor.MoveTo(_patrolPoints[_patrolIndex]);
        }

        private void TickLeashed()
        {
            if (weapon == null || !weapon.HasTarget)
            {
                return;
            }

            float distanceFromAnchor = Vector3.Distance(transform.position, _anchorPoint);
            if (distanceFromAnchor > leashDistance && _stance == Stance.Guard)
            {
                weapon.SetTarget(null);
                _motor.MoveTo(_anchorPoint);
            }
        }
    }
}
