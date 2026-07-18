using System.Collections;
using FerrumProtocol.Combat;
using UnityEngine;

namespace FerrumProtocol.Units
{
    /// <summary>Reference/example ability implementation: temporarily boosts the caster's weapon rate of fire.</summary>
    [CreateAssetMenu(menuName = "Ferrum Protocol/Abilities/Overcharge", fileName = "OverchargeAbility")]
    public class OverchargeAbilitySO : AbilityDataSO
    {
        [SerializeField] private float durationSeconds = 6f;
        [SerializeField] private float fireRateMultiplier = 2f;

        public override bool Activate(GameObject caster)
        {
            var weapon = caster.GetComponentInChildren<WeaponSystem>();
            if (weapon == null || weapon.WeaponData == null)
            {
                return false;
            }

            var runner = caster.GetComponent<MonoBehaviour>();
            if (runner != null)
            {
                runner.StartCoroutine(OverchargeRoutine(weapon));
            }

            return true;
        }

        private IEnumerator OverchargeRoutine(WeaponSystem weapon)
        {
            weapon.SetRateOfFireMultiplier(fireRateMultiplier);
            yield return new WaitForSeconds(durationSeconds);
            weapon.SetRateOfFireMultiplier(1f);
        }
    }
}
