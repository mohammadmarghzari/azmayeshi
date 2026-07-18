namespace FerrumProtocol.Combat
{
    /// <summary>
    /// Pure, static, unit-testable damage math. A live match may override the multiplier
    /// table via a <see cref="DamageMatrixSO"/>; if none is supplied this uses a built-in
    /// balance-placeholder table so the formula itself can be tested in isolation.
    /// </summary>
    public static class DamageCalculator
    {
        public static float ComputeDamage(float rawAmount, ArmorType armor, DamageType damageType, DamageMatrixSO overrideMatrix = null)
        {
            float multiplier = overrideMatrix != null
                ? overrideMatrix.GetMultiplier(armor, damageType)
                : DefaultMultiplier(armor, damageType);

            return UnityEngine.Mathf.Max(0f, rawAmount * multiplier);
        }

        /// <summary>
        /// Built-in placeholder rock-paper-scissors table:
        /// Kinetic favors Unarmored, Explosive favors Structure/HeavyArmor, Energy favors Aircraft,
        /// Piercing favors HeavyArmor, Incendiary favors LightArmor/Unarmored and is weak vs Structure.
        /// Purely a Phase 1 baseline - tune freely once real balance testing starts.
        /// </summary>
        public static float DefaultMultiplier(ArmorType armor, DamageType damageType)
        {
            switch (damageType)
            {
                case DamageType.Kinetic:
                    return armor switch
                    {
                        ArmorType.Unarmored => 1.25f,
                        ArmorType.LightArmor => 1.0f,
                        ArmorType.HeavyArmor => 0.6f,
                        ArmorType.Structure => 0.5f,
                        ArmorType.Aircraft => 0.75f,
                        _ => 1.0f
                    };
                case DamageType.Explosive:
                    return armor switch
                    {
                        ArmorType.Unarmored => 1.0f,
                        ArmorType.LightArmor => 1.1f,
                        ArmorType.HeavyArmor => 1.25f,
                        ArmorType.Structure => 1.5f,
                        ArmorType.Aircraft => 0.6f,
                        _ => 1.0f
                    };
                case DamageType.Energy:
                    return armor switch
                    {
                        ArmorType.Unarmored => 1.0f,
                        ArmorType.LightArmor => 1.0f,
                        ArmorType.HeavyArmor => 0.85f,
                        ArmorType.Structure => 0.85f,
                        ArmorType.Aircraft => 1.35f,
                        _ => 1.0f
                    };
                case DamageType.Piercing:
                    return armor switch
                    {
                        ArmorType.Unarmored => 0.9f,
                        ArmorType.LightArmor => 1.1f,
                        ArmorType.HeavyArmor => 1.4f,
                        ArmorType.Structure => 1.0f,
                        ArmorType.Aircraft => 0.8f,
                        _ => 1.0f
                    };
                case DamageType.Incendiary:
                    return armor switch
                    {
                        ArmorType.Unarmored => 1.3f,
                        ArmorType.LightArmor => 1.2f,
                        ArmorType.HeavyArmor => 0.7f,
                        ArmorType.Structure => 0.4f,
                        ArmorType.Aircraft => 0.5f,
                        _ => 1.0f
                    };
                default:
                    return 1.0f;
            }
        }
    }
}
