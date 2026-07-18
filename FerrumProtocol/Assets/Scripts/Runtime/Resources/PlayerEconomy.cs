using System.Collections.Generic;
using UnityEngine;

namespace FerrumProtocol.Resources
{
    /// <summary>
    /// Per-player resource stockpile. Pure C# state + a thin MonoBehaviour wrapper so the
    /// spend/refund/afford logic is unit-testable without a scene (see FormationSystemTests-style
    /// tests in Tests/EditMode). One instance per player, registered with GameManager.
    /// </summary>
    public class PlayerEconomy : MonoBehaviour
    {
        [SerializeField] private int playerId;
        [SerializeField] private int startingFerrite = 2000;
        [SerializeField] private int startingVoltium = 500;
        [SerializeField] private int startingCommandCells = 0;

        private readonly Dictionary<ResourceType, int> _stockpile = new Dictionary<ResourceType, int>();
        private readonly Dictionary<ResourceType, float> _incomePerSecond = new Dictionary<ResourceType, float>();

        public int PlayerId => playerId;
        public event System.Action<ResourceType, int> OnResourceChanged;

        private void Awake()
        {
            _stockpile[ResourceType.Ferrite] = startingFerrite;
            _stockpile[ResourceType.Voltium] = startingVoltium;
            _stockpile[ResourceType.CommandCells] = startingCommandCells;

            if (Core.GameManager.Instance != null)
            {
                Core.GameManager.Instance.RegisterPlayerEconomy(playerId, this);
            }
        }

        public int GetAmount(ResourceType type) => _stockpile.TryGetValue(type, out int amount) ? amount : 0;

        public void Add(ResourceType type, int amount)
        {
            if (amount <= 0)
            {
                return;
            }

            _stockpile[type] = GetAmount(type) + amount;
            OnResourceChanged?.Invoke(type, _stockpile[type]);
        }

        public bool CanAfford(int ferriteCost, int voltiumCost)
        {
            return GetAmount(ResourceType.Ferrite) >= ferriteCost && GetAmount(ResourceType.Voltium) >= voltiumCost;
        }

        /// <summary>Attempts to spend Ferrite+Voltium atomically - either both succeed or neither is deducted.</summary>
        public bool TrySpend(int ferriteCost, int voltiumCost)
        {
            if (!CanAfford(ferriteCost, voltiumCost))
            {
                return false;
            }

            if (ferriteCost > 0)
            {
                _stockpile[ResourceType.Ferrite] = GetAmount(ResourceType.Ferrite) - ferriteCost;
                OnResourceChanged?.Invoke(ResourceType.Ferrite, _stockpile[ResourceType.Ferrite]);
            }

            if (voltiumCost > 0)
            {
                _stockpile[ResourceType.Voltium] = GetAmount(ResourceType.Voltium) - voltiumCost;
                OnResourceChanged?.Invoke(ResourceType.Voltium, _stockpile[ResourceType.Voltium]);
            }

            return true;
        }

        public void Refund(int ferriteAmount, int voltiumAmount)
        {
            Add(ResourceType.Ferrite, ferriteAmount);
            Add(ResourceType.Voltium, voltiumAmount);
        }
    }
}
