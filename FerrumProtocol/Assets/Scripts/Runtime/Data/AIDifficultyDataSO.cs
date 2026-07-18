using UnityEngine;

namespace FerrumProtocol.Data
{
    /// <summary>One asset per difficulty tier - scales the AI's timing/economy/aggression without branching logic in code.</summary>
    [CreateAssetMenu(menuName = "Ferrum Protocol/AI/Difficulty", fileName = "NewDifficulty")]
    public class AIDifficultyDataSO : ScriptableObject
    {
        public string difficultyName = "Normal";
        [Tooltip("Seconds between AI economy/build decisions - lower is faster/more reactive.")]
        public float decisionIntervalSeconds = 5f;
        [Tooltip("Multiplies resource gathering / build speed effectiveness the AI plans around.")]
        public float economyEfficiencyMultiplier = 1f;
        [Tooltip("Higher values make the AI commit to attacks sooner / with smaller armies.")]
        public float aggressionMultiplier = 1f;
        public int targetHarvesterCount = 6;
        public int minAttackForceSize = 6;
    }
}
