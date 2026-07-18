using UnityEngine;

namespace FerrumProtocol.Resources
{
    /// <summary>A depletable resource deposit that harvester units extract from.</summary>
    public class ResourceNode : MonoBehaviour
    {
        [SerializeField] private ResourceType resourceType = ResourceType.Ferrite;
        [SerializeField] private int totalAmount = 5000;
        [Tooltip("Strategic (CommandCells) nodes typically never respawn once depleted.")]
        [SerializeField] private bool respawns = false;
        [SerializeField] private float respawnDelaySeconds = 120f;

        public ResourceType Type => resourceType;
        public int RemainingAmount { get; private set; }
        public bool IsDepleted => RemainingAmount <= 0;

        public event System.Action OnDepleted;

        private void Awake()
        {
            RemainingAmount = totalAmount;
        }

        /// <summary>Extracts up to <paramref name="requestedAmount"/>, returns the amount actually granted.</summary>
        public int Extract(int requestedAmount)
        {
            if (IsDepleted || requestedAmount <= 0)
            {
                return 0;
            }

            int granted = Mathf.Min(requestedAmount, RemainingAmount);
            RemainingAmount -= granted;

            if (IsDepleted)
            {
                OnDepleted?.Invoke();
                if (respawns)
                {
                    Invoke(nameof(Respawn), respawnDelaySeconds);
                }
                else
                {
                    gameObject.SetActive(false);
                }
            }

            return granted;
        }

        private void Respawn()
        {
            RemainingAmount = totalAmount;
            gameObject.SetActive(true);
        }
    }
}
