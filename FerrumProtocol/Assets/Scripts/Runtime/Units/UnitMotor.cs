using UnityEngine;
using UnityEngine.AI;

namespace FerrumProtocol.Units
{
    /// <summary>Thin wrapper around NavMeshAgent: smart pathfinding + obstacle avoidance come from
    /// the agent itself, this class just exposes the subset of control the rest of the game needs
    /// and raises an arrival event the stance/state machine can react to.</summary>
    [RequireComponent(typeof(NavMeshAgent))]
    public class UnitMotor : MonoBehaviour
    {
        [SerializeField] private float arrivalThreshold = 0.3f;

        private NavMeshAgent _agent;
        private bool _wasMoving;

        public event System.Action OnArrived;

        public bool IsMoving => _agent.hasPath && _agent.remainingDistance > arrivalThreshold;

        private void Awake()
        {
            _agent = GetComponent<NavMeshAgent>();
        }

        public void Configure(float speed, float rotationSpeed)
        {
            _agent.speed = speed;
            _agent.angularSpeed = rotationSpeed;
        }

        public void MoveTo(Vector3 destination)
        {
            if (_agent.isOnNavMesh)
            {
                _agent.SetDestination(destination);
            }
        }

        public void Stop()
        {
            if (_agent.isOnNavMesh)
            {
                _agent.ResetPath();
            }
        }

        private void Update()
        {
            bool moving = IsMoving;
            if (_wasMoving && !moving)
            {
                OnArrived?.Invoke();
            }
            _wasMoving = moving;
        }
    }
}
