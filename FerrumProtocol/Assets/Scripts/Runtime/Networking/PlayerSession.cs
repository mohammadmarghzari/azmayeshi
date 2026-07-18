using Mirror;
using UnityEngine;

namespace FerrumProtocol.Networking
{
    /// <summary>Server-authoritative per-connection state: chosen faction, color, ready status.</summary>
    public class PlayerSession : NetworkBehaviour
    {
        [SyncVar] public int PlayerId;
        [SyncVar] public int FactionIndex = -1;
        [SyncVar] public Color32 TeamColor = new Color32(255, 255, 255, 255);
        [SyncVar] public bool IsReady;

        [Command]
        public void CmdSetFaction(int factionIndex)
        {
            FactionIndex = factionIndex;
        }

        [Command]
        public void CmdSetReady(bool ready)
        {
            IsReady = ready;
        }
    }
}
