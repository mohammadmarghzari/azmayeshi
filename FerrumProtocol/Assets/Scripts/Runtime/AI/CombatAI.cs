using System.Collections.Generic;
using System.Linq;
using FerrumProtocol.Core;
using FerrumProtocol.Data;
using FerrumProtocol.Resources;
using FerrumProtocol.Selection;
using FerrumProtocol.Units;
using UnityEngine;

namespace FerrumProtocol.AI
{
    /// <summary>
    /// Heuristic combat management: waits for a standing army at least as large as the
    /// difficulty's minimum attack force, then attack-moves every idle combat unit at an
    /// enemy target chosen by <see cref="BasePlanner"/>. Retreat/army-composition scoring
    /// are Phase 4 targets - this is the smallest loop that produces a real, beatable AI attack.
    /// </summary>
    public class CombatAI
    {
        private readonly int _playerId;
        private readonly AIDifficultyDataSO _difficulty;
        private readonly BasePlanner _basePlanner;

        public CombatAI(int playerId, AIDifficultyDataSO difficulty, BasePlanner basePlanner)
        {
            _playerId = playerId;
            _difficulty = difficulty;
            _basePlanner = basePlanner;
        }

        public void Tick()
        {
            if (SelectionManager.Instance == null)
            {
                return;
            }

            var combatUnits = SelectionManager.Instance.GetAllOwnedBy(_playerId)
                .Where(s => s.GetComponent<HarvesterUnit>() == null && s.GetComponent<UnitController>() != null)
                .ToList();

            if (combatUnits.Count < _difficulty.minAttackForceSize)
            {
                return;
            }

            Vector3? target = _basePlanner.ChooseAttackTarget(_playerId);
            if (target == null)
            {
                return;
            }

            foreach (var unit in combatUnits)
            {
                var commandable = unit.GetComponent<ICommandable>();
                commandable?.IssueCommand(new UnitCommand
                {
                    Type = CommandType.AttackMove,
                    TargetPoint = target.Value,
                    Queued = false,
                    TargetNetId = -1
                });
            }
        }
    }
}
