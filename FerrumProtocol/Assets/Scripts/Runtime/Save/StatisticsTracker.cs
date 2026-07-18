using UnityEngine;

namespace FerrumProtocol.Save
{
    /// <summary>Accumulates per-match statistics for the post-game summary screen and profile stat history (Phase 5).</summary>
    public class StatisticsTracker : MonoBehaviour
    {
        public MatchStatistics Current { get; private set; } = new MatchStatistics();

        private void Update()
        {
            Current.matchDurationSeconds += Time.deltaTime;
        }

        public void RecordKill() => Current.unitsKilled++;
        public void RecordUnitLost() => Current.unitsLost++;
        public void RecordBuildingLost() => Current.buildingsLost++;
        public void RecordFerriteGathered(int amount) => Current.ferriteGathered += amount;
        public void RecordVoltiumGathered(int amount) => Current.voltiumGathered += amount;

        public void ResetForNewMatch() => Current = new MatchStatistics();
    }
}
