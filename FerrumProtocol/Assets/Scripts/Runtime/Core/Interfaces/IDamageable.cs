using FerrumProtocol.Combat;

namespace FerrumProtocol.Core
{
    /// <summary>Anything with hit points that can take damage and die.</summary>
    public interface IDamageable
    {
        bool IsDead { get; }
        ArmorType ArmorType { get; }
        void ApplyDamage(float rawAmount, DamageType damageType, object source);
    }
}
