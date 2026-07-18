using FerrumProtocol.Combat;
using UnityEngine;

namespace FerrumProtocol.Units
{
    [CreateAssetMenu(menuName = "Ferrum Protocol/Unit/Veterancy Rank", fileName = "NewVeterancyRank")]
    public class VeterancyRankSO : ScriptableObject
    {
        public string rankName = "Veteran";
        public int xpRequired = 100;
        [Tooltip("Multiplies max health when this rank is reached.")]
        public float healthMultiplier = 1.1f;
        [Tooltip("Multiplies outgoing weapon damage when this rank is reached.")]
        public float damageMultiplier = 1.1f;
        [Tooltip("Multiplies movement speed when this rank is reached.")]
        public float speedMultiplier = 1.0f;
    }

    /// <summary>
    /// Tracks experience earned from kills and applies the highest unlocked
    /// <see cref="VeterancyRankSO"/>'s stat multipliers to this unit's Health/Weapon.
    /// </summary>
    public class Veterancy : MonoBehaviour
    {
        [SerializeField] private VeterancyRankSO[] ranks;
        [SerializeField] private Health health;
        [SerializeField] private WeaponSystem weapon;

        public int CurrentXp { get; private set; }
        public int CurrentRankIndex { get; private set; } = -1;
        public event System.Action<VeterancyRankSO> OnRankUp;

        private float _baseMaxHealth;

        private void Awake()
        {
            if (health != null)
            {
                _baseMaxHealth = health.MaxHealth;
            }
        }

        public void AddXp(int amount)
        {
            if (amount <= 0 || ranks == null || ranks.Length == 0)
            {
                return;
            }

            CurrentXp += amount;

            int newRankIndex = CurrentRankIndex;
            for (int i = 0; i < ranks.Length; i++)
            {
                if (CurrentXp >= ranks[i].xpRequired)
                {
                    newRankIndex = i;
                }
            }

            if (newRankIndex != CurrentRankIndex)
            {
                CurrentRankIndex = newRankIndex;
                ApplyRank(ranks[newRankIndex]);
                OnRankUp?.Invoke(ranks[newRankIndex]);
            }
        }

        /// <summary>Convenience hook: wire this to a Health.OnDeath of a unit this weapon killed, e.g. from WeaponSystem/DamageCalculator call sites.</summary>
        public void OnKillCredit(int xpForKill) => AddXp(xpForKill);

        private void ApplyRank(VeterancyRankSO rank)
        {
            if (health != null)
            {
                health.SetMaxHealth(_baseMaxHealth * rank.healthMultiplier, healToFull: false);
            }
            // Weapon damage multiplier is applied at the moment of firing in a full implementation
            // (Phase 2) - kept as a data hook here so the balance model is already wired end to end.
        }
    }
}
