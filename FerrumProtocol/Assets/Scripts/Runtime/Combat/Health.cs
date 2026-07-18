using FerrumProtocol.Core;
using UnityEngine;

namespace FerrumProtocol.Combat
{
    /// <summary>Hit-point pool shared by units and buildings. Implements <see cref="IDamageable"/>.</summary>
    public class Health : MonoBehaviour, IDamageable
    {
        [SerializeField] private float maxHealth = 100f;
        [SerializeField] private ArmorType armorType = ArmorType.Unarmored;
        [SerializeField] private DamageMatrixSO damageMatrixOverride;

        public float MaxHealth => maxHealth;
        public float CurrentHealth { get; private set; }
        public bool IsDead { get; private set; }
        public ArmorType ArmorType => armorType;

        public event System.Action<float, float> OnHealthChanged; // (current, max)
        public event System.Action<object> OnDeath; // (killer/source)
        public event System.Action<float, DamageType, object> OnDamaged; // (amount applied, type, source)

        private void Awake()
        {
            CurrentHealth = maxHealth;
        }

        public void SetMaxHealth(float newMax, bool healToFull = false)
        {
            maxHealth = newMax;
            if (healToFull)
            {
                CurrentHealth = maxHealth;
            }
            else
            {
                CurrentHealth = Mathf.Min(CurrentHealth, maxHealth);
            }
            OnHealthChanged?.Invoke(CurrentHealth, maxHealth);
        }

        public void ApplyDamage(float rawAmount, DamageType damageType, object source)
        {
            if (IsDead || rawAmount <= 0f)
            {
                return;
            }

            float applied = DamageCalculator.ComputeDamage(rawAmount, armorType, damageType, damageMatrixOverride);
            CurrentHealth = Mathf.Max(0f, CurrentHealth - applied);
            OnDamaged?.Invoke(applied, damageType, source);
            OnHealthChanged?.Invoke(CurrentHealth, maxHealth);

            if (CurrentHealth <= 0f)
            {
                IsDead = true;
                OnDeath?.Invoke(source);
            }
        }

        public void Heal(float amount)
        {
            if (IsDead || amount <= 0f)
            {
                return;
            }

            CurrentHealth = Mathf.Min(maxHealth, CurrentHealth + amount);
            OnHealthChanged?.Invoke(CurrentHealth, maxHealth);
        }

        public void Revive(float healthFraction = 1f)
        {
            IsDead = false;
            CurrentHealth = maxHealth * Mathf.Clamp01(healthFraction);
            OnHealthChanged?.Invoke(CurrentHealth, maxHealth);
        }
    }
}
