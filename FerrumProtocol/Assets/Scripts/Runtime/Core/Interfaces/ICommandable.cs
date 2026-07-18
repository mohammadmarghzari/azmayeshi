using UnityEngine;

namespace FerrumProtocol.Core
{
    /// <summary>Order types a player (or the AI) can issue to a selection of units.</summary>
    public enum CommandType
    {
        Move,
        AttackMove,
        Attack,
        Patrol,
        Guard,
        HoldPosition,
        Stop,
        Repair,
        Build,
        Harvest,
        UseAbility
    }

    public struct UnitCommand
    {
        public CommandType Type;
        public Vector3 TargetPoint;
        public int TargetNetId; // networked id of a target entity, -1 if none (used once Mirror commands are wired in Phase 3)
        public Transform TargetTransform; // local-only convenience reference for pre-networking single-machine play
        public bool Queued;     // shift-queue this command after current orders
    }

    /// <summary>Anything that can receive player/AI orders (units, and some buildings e.g. rally points).</summary>
    public interface ICommandable
    {
        void IssueCommand(UnitCommand command);
    }
}
