using UnityEngine;

namespace FerrumProtocol.Utils
{
    /// <summary>Small helpers for OnDrawGizmos/OnDrawGizmosSelected across gameplay components (sight/weapon ranges, formation previews).</summary>
    public static class GizmoUtils
    {
        public static void DrawWireCircleXZ(Vector3 center, float radius, Color color, int segments = 32)
        {
            Gizmos.color = color;
            Vector3 prev = center + new Vector3(radius, 0f, 0f);

            for (int i = 1; i <= segments; i++)
            {
                float angle = (Mathf.PI * 2f / segments) * i;
                Vector3 next = center + new Vector3(Mathf.Cos(angle) * radius, 0f, Mathf.Sin(angle) * radius);
                Gizmos.DrawLine(prev, next);
                prev = next;
            }
        }
    }
}
