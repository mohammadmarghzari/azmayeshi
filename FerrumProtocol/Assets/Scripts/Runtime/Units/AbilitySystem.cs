using System.Collections.Generic;
using UnityEngine;

namespace FerrumProtocol.Units
{
    /// <summary>Holds a unit/building's equipped abilities and their independent cooldown timers.</summary>
    public class AbilitySystem : MonoBehaviour
    {
        [SerializeField] private AbilityDataSO[] abilities;

        private readonly Dictionary<AbilityDataSO, float> _cooldownRemaining = new Dictionary<AbilityDataSO, float>();

        public IReadOnlyList<AbilityDataSO> Abilities => abilities;

        private void Update()
        {
            if (abilities == null)
            {
                return;
            }

            foreach (var ability in abilities)
            {
                if (ability == null)
                {
                    continue;
                }

                if (_cooldownRemaining.TryGetValue(ability, out float remaining) && remaining > 0f)
                {
                    _cooldownRemaining[ability] = remaining - Time.deltaTime;
                }
            }
        }

        public float GetCooldownRemaining(AbilityDataSO ability)
        {
            return _cooldownRemaining.TryGetValue(ability, out float remaining) ? Mathf.Max(0f, remaining) : 0f;
        }

        public bool TryActivate(int abilityIndex)
        {
            if (abilities == null || abilityIndex < 0 || abilityIndex >= abilities.Length)
            {
                return false;
            }

            var ability = abilities[abilityIndex];
            if (ability == null || GetCooldownRemaining(ability) > 0f)
            {
                return false;
            }

            bool activated = ability.Activate(gameObject);
            if (activated)
            {
                _cooldownRemaining[ability] = ability.cooldownSeconds;
            }

            return activated;
        }
    }
}
