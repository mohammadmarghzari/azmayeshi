using FerrumProtocol.Combat;
using NUnit.Framework;
using UnityEngine;

namespace FerrumProtocol.Tests
{
    public class DamageCalculatorTests
    {
        [Test]
        public void KineticDamage_IsStrongAgainstUnarmored()
        {
            float damage = DamageCalculator.ComputeDamage(100f, ArmorType.Unarmored, DamageType.Kinetic);
            Assert.Greater(damage, 100f);
        }

        [Test]
        public void ExplosiveDamage_IsStrongAgainstStructure()
        {
            float damage = DamageCalculator.ComputeDamage(100f, ArmorType.Structure, DamageType.Explosive);
            Assert.Greater(damage, 100f);
        }

        [Test]
        public void EnergyDamage_IsStrongAgainstAircraft()
        {
            float damage = DamageCalculator.ComputeDamage(100f, ArmorType.Aircraft, DamageType.Energy);
            Assert.Greater(damage, 100f);
        }

        [Test]
        public void ZeroRawDamage_ProducesZeroResult()
        {
            float damage = DamageCalculator.ComputeDamage(0f, ArmorType.HeavyArmor, DamageType.Piercing);
            Assert.AreEqual(0f, damage);
        }

        [Test]
        public void DamageMatrixSO_OverridesDefaultTable()
        {
            var matrix = ScriptableObject.CreateInstance<DamageMatrixSO>();
            var so = new UnityEditor.SerializedObject(matrix);
            var entries = so.FindProperty("entries");
            entries.InsertArrayElementAtIndex(0);
            var entry = entries.GetArrayElementAtIndex(0);
            entry.FindPropertyRelative("armorType").enumValueIndex = (int)ArmorType.HeavyArmor;
            entry.FindPropertyRelative("damageType").enumValueIndex = (int)DamageType.Kinetic;
            entry.FindPropertyRelative("multiplier").floatValue = 3f;
            so.ApplyModifiedPropertiesWithoutUndo();

            float damage = DamageCalculator.ComputeDamage(10f, ArmorType.HeavyArmor, DamageType.Kinetic, matrix);
            Assert.AreEqual(30f, damage, 0.001f);
        }
    }
}
