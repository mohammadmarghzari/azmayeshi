using UnityEngine;

namespace FerrumProtocol.CameraSystem
{
    /// <summary>World-space rectangle (XZ) the RTS camera's focus point is clamped to, so panning can't leave the playable map.</summary>
    [System.Serializable]
    public struct CameraBounds
    {
        public float minX;
        public float maxX;
        public float minZ;
        public float maxZ;

        public static CameraBounds FromCenterExtents(Vector3 center, Vector2 halfExtents)
        {
            return new CameraBounds
            {
                minX = center.x - halfExtents.x,
                maxX = center.x + halfExtents.x,
                minZ = center.z - halfExtents.y,
                maxZ = center.z + halfExtents.y
            };
        }

        public Vector3 Clamp(Vector3 point)
        {
            point.x = Mathf.Clamp(point.x, minX, maxX);
            point.z = Mathf.Clamp(point.z, minZ, maxZ);
            return point;
        }
    }
}
