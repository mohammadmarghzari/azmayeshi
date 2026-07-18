using UnityEngine;

namespace FerrumProtocol.Units
{
    /// <summary>
    /// Base class for a special ability. Subclass and override <see cref="Activate"/> for each
    /// concrete ability (e.g. an energy overcharge, a cloak field, an artillery barrage) - the
    /// strategy pattern lets designers add new abilities as new asset types without touching
    /// <see cref="AbilitySystem"/>.
    /// </summary>
    public abstract class AbilityDataSO : ScriptableObject
    {
        public string abilityName = "New Ability";
        public Sprite icon;
        public float cooldownSeconds = 20f;
        public int voltiumCost = 0;

        /// <summary>Return true if the ability actually fired (false lets AbilitySystem avoid starting the cooldown, e.g. no valid target).</summary>
        public abstract bool Activate(GameObject caster);
    }
}
