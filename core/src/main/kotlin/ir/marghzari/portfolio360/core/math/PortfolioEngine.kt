package ir.marghzari.portfolio360.core.math

import ir.marghzari.portfolio360.core.model.PortfolioMetrics
import ir.marghzari.portfolio360.core.model.PortfolioStyle
import ir.marghzari.portfolio360.core.model.RiskInputs
import ir.marghzari.portfolio360.core.model.TalebSets
import ir.marghzari.portfolio360.core.model.WeightsResult
import kotlin.math.min
import kotlin.random.Random

/** Direct port of `calc_risk_penalty`, `calc_weights`, and `portfolio_metrics` from app.py. */
object PortfolioEngine {

    fun riskPenalty(risk: RiskInputs): Double {
        val weighted = (risk.riskGeoPct * 0.40 + risk.riskMonPct * 0.35 + risk.riskSysPct * 0.25) / 100.0
        return min(weighted * 0.5, 0.50)
    }

    /**
     * @param returns daily returns matrix [day][assetIndex], in the same order as [tickers].
     */
    fun calcWeights(
        style: PortfolioStyle,
        returns: Array<DoubleArray>,
        rf: Double,
        tickers: List<String>,
        risk: RiskInputs = RiskInputs(),
        random: Random = Random(42),
    ): WeightsResult {
        val n = tickers.size
        val meanAnnual = Stats.annualizedMean(returns, n)
        val cov = Stats.annualizedCovariance(returns, n)
        val lo = 0.01
        val hi = 0.6
        val effectiveRf = rf + riskPenalty(risk)
        val targetAnnual = risk.expectedReturnPct / 100.0

        val weights: DoubleArray = when (style) {
            PortfolioStyle.EQUAL_WEIGHT -> DoubleArray(n) { 1.0 / n }

            PortfolioStyle.MAX_SHARPE -> GenericOptimizer.projectedGradientDescent(n, lo, hi) { w ->
                val ret = LinAlg.dot(w, meanAnnual)
                val vol = kotlin.math.sqrt(LinAlg.quadForm(w, cov)) + 1e-9
                -(ret - effectiveRf) / vol + GenericOptimizer.returnFloorPenalty(w, meanAnnual, targetAnnual)
            }.let { w -> val s = w.sum(); if (s > 1e-9) DoubleArray(n) { w[it] / s } else w }

            PortfolioStyle.MIN_VARIANCE -> GenericOptimizer.projectedGradientDescent(n, lo, hi) { w ->
                LinAlg.quadForm(w, cov) + GenericOptimizer.returnFloorPenalty(w, meanAnnual, targetAnnual)
            }.let { w -> val s = w.sum(); if (s > 1e-9) DoubleArray(n) { w[it] / s } else w }

            PortfolioStyle.RISK_PARITY -> GenericOptimizer.projectedGradientDescent(n, 0.01, 1.0) { w ->
                val sigma = kotlin.math.sqrt(LinAlg.quadForm(w, cov)) + 1e-9
                val mrc = LinAlg.matVec(cov, w).let { LinAlg.scale(it, 1.0 / sigma) }
                val rc = DoubleArray(n) { w[it] * mrc[it] }
                val target = sigma / n
                rc.sumOf { (it - target) * (it - target) }
            }

            PortfolioStyle.MONTE_CARLO_CVAR -> {
                val nDays = returns.size
                var bestW = DoubleArray(n) { 1.0 / n }
                var bestVal = Double.POSITIVE_INFINITY
                repeat(200) {
                    val sampleIdx = IntArray(nDays) { random.nextInt(nDays) }
                    val sample = Array(nDays) { i -> returns[sampleIdx[i]] }
                    val w = GenericOptimizer.projectedGradientDescent(n, lo, hi, iterations = 120) { w ->
                        val portRet = DoubleArray(nDays) { d -> LinAlg.dot(sample[d], w) }
                        -Stats.percentile(portRet, 5.0)
                    }
                    val portRet = DoubleArray(nDays) { d -> LinAlg.dot(sample[d], w) }
                    val v = -Stats.percentile(portRet, 5.0)
                    if (v < bestVal) { bestVal = v; bestW = w }
                }
                bestW
            }

            PortfolioStyle.TALEB_BARBELL -> {
                val safeIdx = tickers.indices.filter { tickers[it] in TalebSets.SAFE }
                val riskyIdx = tickers.indices.filter { tickers[it] in TalebSets.RISKY }
                if (safeIdx.isEmpty() && riskyIdx.isEmpty()) {
                    DoubleArray(n) { 1.0 / n }
                } else {
                    val w = DoubleArray(n) { 0.10 / n }
                    if (riskyIdx.isNotEmpty()) {
                        safeIdx.forEach { w[it] = 0.90 / safeIdx.size }
                        riskyIdx.forEach { w[it] = 0.10 / riskyIdx.size }
                        tickers.indices.filter { it !in safeIdx && it !in riskyIdx }.forEach { w[it] = 0.0 }
                    } else {
                        val others = tickers.indices.filter { it !in safeIdx }
                        safeIdx.forEach { w[it] = 0.90 / safeIdx.size }
                        if (others.isNotEmpty()) others.forEach { w[it] = 0.10 / others.size }
                    }
                    val clipped = DoubleArray(n) { w[it].coerceIn(0.0, 1.0) }
                    val s = clipped.sum()
                    if (s > 1e-9) DoubleArray(n) { clipped[it] / s } else clipped
                }
            }
        }
        return WeightsResult(weights, cov)
    }

    fun portfolioMetrics(
        weights: DoubleArray,
        returns: Array<DoubleArray>,
        rf: Double,
        risk: RiskInputs = RiskInputs(),
    ): PortfolioMetrics {
        val portRet = Stats.portfolioReturns(returns, weights)
        val annRet = Stats.annualizedReturn(portRet)
        val annVol = Stats.annualizedVol(portRet)
        val penalty = riskPenalty(risk)
        val riskAdjRet = annRet * (1 - penalty)
        val sharpe = (riskAdjRet - rf) / (annVol + 1e-9)
        val cum = Stats.cumulative(portRet)
        val dd = Stats.drawdownSeries(cum)
        val maxDd = dd.minOrNull() ?: 0.0
        val recovery = Stats.longestUnderwaterStreak(dd)
        val cvar = -Stats.percentile(portRet, 5.0)
        val calmar = riskAdjRet / (kotlin.math.abs(maxDd) + 1e-9)
        val gap = if (risk.expectedReturnPct > 0) riskAdjRet - risk.expectedReturnPct / 100.0 else null
        return PortfolioMetrics(
            annualReturn = annRet,
            riskAdjustedReturn = riskAdjRet,
            annualVolatility = annVol,
            sharpeRatio = sharpe,
            maxDrawdown = maxDd,
            cvar95 = cvar,
            calmarRatio = calmar,
            recoveryDays = recovery,
            returnGap = gap,
            riskDiscountPct = penalty * 100.0,
        )
    }
}
