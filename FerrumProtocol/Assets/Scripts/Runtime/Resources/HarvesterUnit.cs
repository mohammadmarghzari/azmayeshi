using FerrumProtocol.Selection;
using FerrumProtocol.Units;
using UnityEngine;

namespace FerrumProtocol.Resources
{
    public enum HarvesterState
    {
        Idle,
        MovingToNode,
        Harvesting,
        ReturningToDropoff,
        Depositing
    }

    /// <summary>
    /// Worker-unit economy loop: move to node -> extract over time -> return to nearest
    /// friendly dropoff building -> deposit into that player's <see cref="PlayerEconomy"/> -> repeat.
    /// Attach alongside UnitController/UnitMotor/Selectable on a worker unit prefab.
    /// </summary>
    [RequireComponent(typeof(UnitMotor))]
    public class HarvesterUnit : MonoBehaviour
    {
        [SerializeField] private int capacity = 100;
        [SerializeField] private int extractPerTick = 10;
        [SerializeField] private float tickIntervalSeconds = 1f;
        [SerializeField] private Selectable selectable;

        private UnitMotor _motor;
        private ResourceNode _targetNode;
        private IResourceDropoff _targetDropoff;
        private HarvesterState _state = HarvesterState.Idle;
        private int _carriedAmount;
        private ResourceType _carriedType;
        private float _tickTimer;

        public HarvesterState State => _state;

        private void Awake()
        {
            _motor = GetComponent<UnitMotor>();
        }

        public void CommandHarvest(Transform nodeTransform)
        {
            if (nodeTransform == null || !nodeTransform.TryGetComponent<ResourceNode>(out var node))
            {
                return;
            }

            _targetNode = node;
            _state = HarvesterState.MovingToNode;
            _motor.MoveTo(node.transform.position);
        }

        private void Update()
        {
            switch (_state)
            {
                case HarvesterState.MovingToNode:
                    TickMovingToNode();
                    break;
                case HarvesterState.Harvesting:
                    TickHarvesting();
                    break;
                case HarvesterState.ReturningToDropoff:
                    TickReturning();
                    break;
            }
        }

        private void TickMovingToNode()
        {
            if (_targetNode == null || _targetNode.IsDepleted)
            {
                _state = HarvesterState.Idle;
                return;
            }

            if (!_motor.IsMoving)
            {
                _state = HarvesterState.Harvesting;
                _tickTimer = 0f;
            }
        }

        private void TickHarvesting()
        {
            if (_targetNode == null || _targetNode.IsDepleted)
            {
                BeginReturnOrIdle();
                return;
            }

            _tickTimer -= Time.deltaTime;
            if (_tickTimer > 0f)
            {
                return;
            }
            _tickTimer = tickIntervalSeconds;

            int spaceLeft = capacity - _carriedAmount;
            if (spaceLeft <= 0)
            {
                BeginReturnOrIdle();
                return;
            }

            int extracted = _targetNode.Extract(Mathf.Min(extractPerTick, spaceLeft));
            if (extracted > 0)
            {
                _carriedType = _targetNode.Type;
                _carriedAmount += extracted;
            }

            if (_carriedAmount >= capacity || _targetNode.IsDepleted)
            {
                BeginReturnOrIdle();
            }
        }

        private void BeginReturnOrIdle()
        {
            if (_carriedAmount <= 0)
            {
                _state = HarvesterState.Idle;
                return;
            }

            int ownerId = selectable != null ? selectable.OwnerPlayerId : 0;
            _targetDropoff = ResourceDropoffRegistry.FindNearest(transform.position, ownerId);

            if (_targetDropoff == null)
            {
                _state = HarvesterState.Idle;
                return;
            }

            _state = HarvesterState.ReturningToDropoff;
            _motor.MoveTo(_targetDropoff.Transform.position);
        }

        private void TickReturning()
        {
            if (_targetDropoff == null)
            {
                _state = HarvesterState.Idle;
                return;
            }

            if (!_motor.IsMoving)
            {
                _targetDropoff.Deposit(_carriedType, _carriedAmount);
                _carriedAmount = 0;

                if (_targetNode != null && !_targetNode.IsDepleted)
                {
                    _state = HarvesterState.MovingToNode;
                    _motor.MoveTo(_targetNode.transform.position);
                }
                else
                {
                    _state = HarvesterState.Idle;
                }
            }
        }
    }
}
