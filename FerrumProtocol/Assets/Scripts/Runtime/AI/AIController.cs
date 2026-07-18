using FerrumProtocol.Core;
using FerrumProtocol.Data;
using UnityEngine;

namespace FerrumProtocol.AI
{
    /// <summary>
    /// Top-level per-AI-player brain: composes <see cref="EconomyAI"/>, <see cref="CombatAI"/>,
    /// and <see cref="BasePlanner"/> behind a swappable <see cref="AIDifficultyDataSO"/>. One of
    /// these runs per AI-controlled slot in a Skirmish/Campaign match.
    /// </summary>
    public class AIController : MonoBehaviour
    {
        [SerializeField] private int playerId;
        [SerializeField] private AIDifficultyDataSO difficulty;

        private EconomyAI _economyAi;
        private CombatAI _combatAi;
        private BasePlanner _basePlanner;
        private float _decisionTimer;

        private void Awake()
        {
            if (difficulty == null)
            {
                Debug.LogWarning($"{nameof(AIController)} on {name} has no difficulty asset assigned - disabling.");
                enabled = false;
                return;
            }

            _basePlanner = new BasePlanner(BasePlanner.DefaultConfig());
            _economyAi = new EconomyAI(playerId, difficulty);
            _combatAi = new CombatAI(playerId, difficulty, _basePlanner);
        }

        private void Update()
        {
            _decisionTimer -= Time.deltaTime;
            if (_decisionTimer > 0f)
            {
                return;
            }
            _decisionTimer = difficulty.decisionIntervalSeconds;

            var economy = GameManager.Instance != null ? GameManager.Instance.GetPlayerEconomy(playerId) : null;
            _economyAi.Tick(economy);
            _combatAi.Tick();
        }
    }
}
