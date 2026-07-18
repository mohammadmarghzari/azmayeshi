using UnityEngine;

namespace FerrumProtocol.Utils
{
    public static class MathUtils
    {
        /// <summary>Flattens a position onto the XZ plane at y=0 - convenient for RTS ground-plane math (selection, formations).</summary>
        public static Vector3 FlattenY(Vector3 v) => new Vector3(v.x, 0f, v.z);

        public static bool IsWithinRange(Vector3 a, Vector3 b, float range) => (a - b).sqrMagnitude <= range * range;
    }
}
