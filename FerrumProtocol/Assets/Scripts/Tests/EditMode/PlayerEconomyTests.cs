using FerrumProtocol.Resources;
using NUnit.Framework;
using UnityEngine;

namespace FerrumProtocol.Tests
{
    public class PlayerEconomyTests
    {
        private GameObject _go;
        private PlayerEconomy _economy;

        [SetUp]
        public void SetUp()
        {
            _go = new GameObject("TestEconomy");
            _economy = _go.AddComponent<PlayerEconomy>();
        }

        [TearDown]
        public void TearDown()
        {
            Object.DestroyImmediate(_go);
        }

        [Test]
        public void StartingFerrite_MatchesDefault()
        {
            Assert.AreEqual(2000, _economy.GetAmount(ResourceType.Ferrite));
        }

        [Test]
        public void TrySpend_DeductsBothResourcesWhenAffordable()
        {
            bool result = _economy.TrySpend(500, 100);
            Assert.IsTrue(result);
            Assert.AreEqual(1500, _economy.GetAmount(ResourceType.Ferrite));
            Assert.AreEqual(400, _economy.GetAmount(ResourceType.Voltium));
        }

        [Test]
        public void TrySpend_FailsAtomically_WhenOnlyOneResourceInsufficient()
        {
            // Starting: 2000 Ferrite, 500 Voltium. Ask for affordable Ferrite but too much Voltium.
            bool result = _economy.TrySpend(100, 999999);
            Assert.IsFalse(result);
            Assert.AreEqual(2000, _economy.GetAmount(ResourceType.Ferrite), "Ferrite must not be deducted when the overall spend fails.");
        }

        [Test]
        public void Refund_AddsBackResources()
        {
            _economy.TrySpend(500, 100);
            _economy.Refund(500, 100);
            Assert.AreEqual(2000, _economy.GetAmount(ResourceType.Ferrite));
            Assert.AreEqual(500, _economy.GetAmount(ResourceType.Voltium));
        }

        [Test]
        public void CanAfford_ReturnsFalse_WhenInsufficientFunds()
        {
            Assert.IsFalse(_economy.CanAfford(999999, 0));
        }
    }
}
