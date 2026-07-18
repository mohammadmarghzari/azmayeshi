using FerrumProtocol.Core;
using FerrumProtocol.Units;
using Mirror;
using UnityEngine;

namespace FerrumProtocol.Networking
{
    /// <summary>
    /// Server-authoritative position/command sync for a unit. The owning client sends intent
    /// via <see cref="CmdIssueCommand"/>; the server is the only place that actually calls
    /// <see cref="UnitController.IssueCommand"/>. Position is replicated through a SyncVar with
    /// a client-side hook that smooth-corrects rather than snapping - the actual predict-locally
    /// -then-reconcile logic is a Phase 3 addition, but the hook point exists now so retrofitting
    /// it later doesn't require touching this class's public surface.
    /// </summary>
    public class NetworkedUnit : NetworkBehaviour
    {
        [SerializeField] private UnitController unitController;

        [SyncVar(hook = nameof(OnServerPositionChanged))]
        private Vector3 _syncedPosition;

        [SerializeField] private float clientCorrectionSpeed = 10f;

        public override void OnStartServer()
        {
            _syncedPosition = transform.position;
        }

        private void Update()
        {
            if (isServer)
            {
                _syncedPosition = transform.position;
            }
            else if (!isOwned)
            {
                // Remote clients smooth-correct toward the authoritative position rather than
                // teleporting on every SyncVar tick - the actual prediction/reconciliation
                // layer (extrapolate locally, reconcile on divergence) lands in Phase 3.
                transform.position = Vector3.Lerp(transform.position, _syncedPosition, Time.deltaTime * clientCorrectionSpeed);
            }
        }

        private void OnServerPositionChanged(Vector3 oldPos, Vector3 newPos)
        {
            // Hook reserved for Phase 3 reconciliation (compare against locally-predicted position).
        }

        /// <summary>
        /// Mirror's weaver can only auto-serialize primitive/networked-reference types, so this
        /// takes plain fields rather than the local-only <see cref="UnitCommand"/> struct (which
        /// carries a raw Transform reference that has no meaning across the network). The server
        /// reconstructs a full UnitCommand here, resolving <paramref name="targetNetId"/> to a
        /// Transform via the spawned-object table.
        /// </summary>
        [Command]
        public void CmdIssueCommand(CommandType type, Vector3 targetPoint, uint targetNetId, bool queued)
        {
            // Server validates ownership implicitly: this Command RPC can only run on the
            // object the calling client has authority over (Mirror enforces this).
            Transform targetTransform = null;
            if (targetNetId != 0 && NetworkServer.spawned.TryGetValue(targetNetId, out var identity))
            {
                targetTransform = identity.transform;
            }

            unitController?.IssueCommand(new UnitCommand
            {
                Type = type,
                TargetPoint = targetPoint,
                TargetNetId = (int)targetNetId,
                TargetTransform = targetTransform,
                Queued = queued
            });
        }
    }
}
