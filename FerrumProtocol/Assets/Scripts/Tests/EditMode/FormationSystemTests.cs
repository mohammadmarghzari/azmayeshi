using FerrumProtocol.Units;
using NUnit.Framework;
using UnityEngine;

namespace FerrumProtocol.Tests
{
    public class FormationSystemTests
    {
        [Test]
        public void ComputeDestinations_ReturnsOnePointPerUnit()
        {
            var points = FormationSystem.ComputeDestinations(Vector3.zero, Vector3.forward, 9, FormationType.Box);
            Assert.AreEqual(9, points.Count);
        }

        [Test]
        public void ComputeDestinations_ZeroUnits_ReturnsEmpty()
        {
            var points = FormationSystem.ComputeDestinations(Vector3.zero, Vector3.forward, 0, FormationType.Line);
            Assert.AreEqual(0, points.Count);
        }

        [Test]
        public void LineFormation_IsCenteredOnTarget()
        {
            var points = FormationSystem.ComputeDestinations(Vector3.zero, Vector3.forward, 3, FormationType.Line, spacing: 2f);
            Vector3 average = Vector3.zero;
            foreach (var p in points) average += p;
            average /= points.Count;

            Assert.AreEqual(0f, average.x, 0.001f);
            Assert.AreEqual(0f, average.z, 0.001f);
        }

        [Test]
        public void WedgeFormation_PlacesTipClosestToTarget()
        {
            var points = FormationSystem.ComputeDestinations(Vector3.zero, Vector3.forward, 6, FormationType.Wedge, spacing: 2f);
            Assert.AreEqual(6, points.Count);
            // The first placed point (row 0) should have the smallest offset opposite the facing direction.
            Assert.AreEqual(0f, points[0].z, 0.001f);
        }
    }
}
