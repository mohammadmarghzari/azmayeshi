using Mirror;
using UnityEngine;

namespace FerrumProtocol.Networking
{
    /// <summary>
    /// Match entry point: extends Mirror's NetworkManager to spawn a <see cref="PlayerSession"/>
    /// per connection and keep a server-side <see cref="LobbyManager"/> in sync. Match-start
    /// (loading the actual gameplay scene with bases/economy) is triggered once the lobby
    /// reaches <see cref="LobbyPhase.AllReady"/> and the host confirms.
    /// </summary>
    public class RTSNetworkManager : NetworkManager
    {
        [SerializeField] private GameObject playerSessionPrefab;

        public LobbyManager Lobby { get; } = new LobbyManager();

        private int _nextPlayerId;

        public override void OnServerAddPlayer(NetworkConnectionToClient conn)
        {
            int playerId = _nextPlayerId++;
            GameObject sessionObject = Instantiate(playerSessionPrefab != null ? playerSessionPrefab : playerPrefab);

            if (sessionObject.TryGetComponent<PlayerSession>(out var session))
            {
                session.PlayerId = playerId;
            }

            NetworkServer.AddPlayerForConnection(conn, sessionObject);
            Lobby.AddPlayer(playerId, $"Player{playerId}");
        }

        public override void OnServerDisconnect(NetworkConnectionToClient conn)
        {
            if (conn.identity != null && conn.identity.TryGetComponent<PlayerSession>(out var session))
            {
                Lobby.RemovePlayer(session.PlayerId);
            }

            base.OnServerDisconnect(conn);
        }

        public void ServerRequestStartMatch()
        {
            if (!Lobby.TryBeginStarting())
            {
                Debug.LogWarning("Cannot start match: not all players are ready.");
                return;
            }

            ServerChangeScene(onlineScene);
        }
    }
}
