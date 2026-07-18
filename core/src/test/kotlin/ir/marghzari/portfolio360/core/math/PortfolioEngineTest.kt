package ir.marghzari.portfolio360.core.math

import ir.marghzari.portfolio360.core.model.PortfolioStyle
import ir.marghzari.portfolio360.core.model.RiskInputs
import kotlin.random.Random
import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class PortfolioEngineTest {

    private fun syntheticReturns(nDays: Int, nAssets: Int, vols: DoubleArray, seed: Int): Array<DoubleArray> {
        val random = Random(seed)
        return Array(nDays) { DoubleArray(nAssets) { j -> (random.nextDouble() - 0.5) * 2.0 * vols[j] } }
    }

    @Test
    fun `equal weight sums to one and is uniform`() {
        val tickers = listOf("A", "B", "C", "D")
        val returns = syntheticReturns(300, 4, doubleArrayOf(0.01, 0.01, 0.01, 0.01), seed = 1)
        val result = PortfolioEngine.calcWeights(PortfolioStyle.EQUAL_WEIGHT, returns, 0.02, tickers)
        assertEquals(1.0, result.weights.sum(), 1e-9)
        result.weights.forEach { assertEquals(0.25, it, 1e-9) }
    }

    @Test
    fun `min variance favors the lower-volatility asset and stays feasible`() {
        val tickers = listOf("LOW_VOL", "HIGH_VOL", "MID_VOL")
        // Three independent assets with very different volatility -> min-variance should overweight LOW_VOL.
        val returns = syntheticReturns(500, 3, doubleArrayOf(0.003, 0.05, 0.015), seed = 2)
        val result = PortfolioEngine.calcWeights(PortfolioStyle.MIN_VARIANCE, returns, 0.02, tickers)
        assertEquals(1.0, result.weights.sum(), 0.01)
        result.weights.forEach { assertTrue(it in 0.0..0.61) }
        assertTrue(result.weights[0] > result.weights[1], "low-vol weight ${result.weights[0]} should exceed high-vol weight ${result.weights[1]}")
    }

    @Test
    fun `risk parity produces roughly equal risk contributions for symmetric assets`() {
        val tickers = listOf("A", "B")
        val returns = syntheticReturns(400, 2, doubleArrayOf(0.02, 0.02), seed = 3)
        val result = PortfolioEngine.calcWeights(PortfolioStyle.RISK_PARITY, returns, 0.02, tickers)
        // Symmetric vol/corr assets -> risk parity should land close to equal weight.
        assertEquals(0.5, result.weights[0], 0.05)
    }

    @Test
    fun `portfolio metrics produce finite sensible values`() {
        val tickers = listOf("A", "B")
        val returns = syntheticReturns(300, 2, doubleArrayOf(0.01, 0.015), seed = 4)
        val weights = doubleArrayOf(0.5, 0.5)
        val metrics = PortfolioEngine.portfolioMetrics(weights, returns, 0.02, RiskInputs())
        assertTrue(metrics.annualVolatility > 0)
        assertTrue(metrics.maxDrawdown <= 0.0)
        assertTrue(metrics.cvar95.isFinite())
    }

    @Test
    fun `risk penalty is capped at 50 percent and matches the weighted formula`() {
        val full = PortfolioEngine.riskPenalty(RiskInputs(riskGeoPct = 100.0, riskMonPct = 100.0, riskSysPct = 100.0))
        assertEquals(0.50, full, 1e-9)
        val partial = PortfolioEngine.riskPenalty(RiskInputs(riskGeoPct = 40.0, riskMonPct = 0.0, riskSysPct = 0.0))
        assertEquals((40.0 * 0.40 / 100.0) * 0.5, partial, 1e-9)
    }
}
