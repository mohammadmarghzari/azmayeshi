using System.Collections.Generic;
using FerrumProtocol.Resources;
using UnityEngine;

namespace FerrumProtocol.Core
{
    public enum MatchState
    {
        PreGame,
        Playing,
        Paused,
        Ended
    }

    /// <summary>
    /// Top-level orchestrator for a single match. Owns match state and per-player economy,
    /// and registers shared services into the ServiceLocator on Awake so other systems can
    /// resolve them without a chain of manual Inspector references.
    /// </summary>
    public class GameManager : MonoBehaviour
    {
        public static GameManager Instance { get; private set; }

        [SerializeField] private int localPlayerId = 0;

        public MatchState State { get; private set; } = MatchState.PreGame;
        public int LocalPlayerId => localPlayerId;

        private readonly Dictionary<int, PlayerEconomy> _playerEconomies = new Dictionary<int, PlayerEconomy>();

        public event System.Action<MatchState> OnMatchStateChanged;

        private void Awake()
        {
            if (Instance != null && Instance != this)
            {
                Destroy(gameObject);
                return;
            }

            Instance = this;
            ServiceLocator.Register(this);
        }

        private void OnDestroy()
        {
            if (Instance == this)
            {
                ServiceLocator.Unregister<GameManager>();
                Instance = null;
            }
        }

        public void RegisterPlayerEconomy(int playerId, PlayerEconomy economy)
        {
            _playerEconomies[playerId] = economy;
        }

        public PlayerEconomy GetPlayerEconomy(int playerId)
        {
            return _playerEconomies.TryGetValue(playerId, out var economy) ? economy : null;
        }

        public void SetMatchState(MatchState newState)
        {
            if (State == newState)
            {
                return;
            }

            State = newState;
            OnMatchStateChanged?.Invoke(newState);
        }
    }
}
