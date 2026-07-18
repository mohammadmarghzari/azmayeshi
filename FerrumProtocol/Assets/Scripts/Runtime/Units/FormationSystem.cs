using System.Collections.Generic;
using UnityEngine;

namespace FerrumProtocol.Units
{
    public enum FormationType
    {
        Line,
        Box,
        Wedge
    }

    /// <summary>
    /// Pure math: given a move order for N units to a destination, returns per-unit target
    /// offsets so they arrive spread out instead of stacking on one point. No MonoBehaviour
    /// dependency, so this is directly unit-testable.
    /// </summary>
    public static class FormationSystem
    {
        public static List<Vector3> ComputeDestinations(Vector3 center, Vector3 facingDirection, int unitCount, FormationType formation, float spacing = 2.5f)
        {
            var results = new List<Vector3>(unitCount);
            if (unitCount <= 0)
            {
                return results;
            }

            if (facingDirection.sqrMagnitude < 0.0001f)
            {
                facingDirection = Vector3.forward;
            }
            facingDirection.Normalize();
            Vector3 right = Vector3.Cross(Vector3.up, facingDirection).normalized;

            switch (formation)
            {
                case FormationType.Line:
                    ComputeLine(center, right, unitCount, spacing, results);
                    break;
                case FormationType.Wedge:
                    ComputeWedge(center, facingDirection, right, unitCount, spacing, results);
                    break;
                case FormationType.Box:
                default:
                    ComputeBox(center, facingDirection, right, unitCount, spacing, results);
                    break;
            }

            return results;
        }

        private static void ComputeLine(Vector3 center, Vector3 right, int count, float spacing, List<Vector3> results)
        {
            float totalWidth = (count - 1) * spacing;
            float startOffset = -totalWidth * 0.5f;

            for (int i = 0; i < count; i++)
            {
                results.Add(center + right * (startOffset + i * spacing));
            }
        }

        private static void ComputeBox(Vector3 center, Vector3 forward, Vector3 right, int count, float spacing, List<Vector3> results)
        {
            int columns = Mathf.CeilToInt(Mathf.Sqrt(count));
            int row = 0, col = 0;

            for (int i = 0; i < count; i++)
            {
                float x = (col - (columns - 1) * 0.5f) * spacing;
                float z = row * spacing;
                results.Add(center + right * x + forward * z);

                col++;
                if (col >= columns)
                {
                    col = 0;
                    row++;
                }
            }
        }

        private static void ComputeWedge(Vector3 center, Vector3 forward, Vector3 right, int count, float spacing, List<Vector3> results)
        {
            int placed = 0;
            int row = 0;

            while (placed < count)
            {
                int unitsInRow = row + 1;
                float rowWidth = (unitsInRow - 1) * spacing;
                float startOffset = -rowWidth * 0.5f;

                for (int i = 0; i < unitsInRow && placed < count; i++, placed++)
                {
                    float x = startOffset + i * spacing;
                    float z = -row * spacing; // rows fan out behind the tip
                    results.Add(center + right * x + forward * z);
                }

                row++;
            }
        }
    }
}
