using Mirror;
using UnityEngine;

namespace FerrumProtocol.Networking
{
    /// <summary>
    /// Wire messages for client-intent -> server-validation flow. Clients never assert
    /// "I moved to X"; they ask "please move unit N to X" and the server decides what
    /// actually happens (see Documentation/NETWORKING.md).
    /// </summary>
    public struct MoveCommandMessage : NetworkMessage
    {
        public uint[] UnitNetIds;
        public Vector3 Destination;
        public bool Queued;
    }

    public struct AttackCommandMessage : NetworkMessage
    {
        public uint[] UnitNetIds;
        public uint TargetNetId;
        public bool Queued;
    }

    public struct BuildCommandMessage : NetworkMessage
    {
        public int BuildingDataIndex; // index into the match's synced building catalog
        public Vector3 Position;
    }

    public struct ProduceUnitCommandMessage : NetworkMessage
    {
        public uint ProducerBuildingNetId;
        public int UnitDataIndex;
    }
}
