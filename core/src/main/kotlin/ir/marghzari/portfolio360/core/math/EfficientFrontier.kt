package ir.marghzari.portfolio360.core.math

import kotlin.math.ln
import kotlin.random.Random

data class SimulatedPortfolio(val volPct: Double, val retPct: Double, val sharpe: Double, val weights: DoubleArray)
data class FrontierPoint(val volPct: Double, val retPct: Double)

data class EfficientFrontierResult(
    val simulated: List<SimulatedPortfolio>,
    val frontier: List<FrontierPoint>,
    val currentVolPct: Double, val currentRetPct: Double, val currentSharpe: Double,
    val bestSharpePoint: SimulatedPortfolio?,
)

/** Direct port of the Efficient Frontier tab's inline random-portfolio simulation + frontier trace. */
object EfficientFrontier {

    /** Dirichlet(1,...,1) sample via normalized Exponential(1) draws — standard construction. */
    private fun dirichletUniform(n: Int, random: Random): DoubleArray {
        val draws = DoubleArray(n) { -ln(random.nextDouble().coerceAtLeast(1e-12)) }
        val s = draws.sum()
        return DoubleArray(n) { draws[it] / s }
    }

    fun compute(
        weights: DoubleArray,
        returns: Array<DoubleArray>,
        tickers: List<String>,
        rf: Double,
        nSimulations: Int = 1500,
        nFrontierPoints: Int = 60,
        random: Random = Random(7),
    ): EfficientFrontierResult {
        val n = tickers.size
        val mean = Stats.annualizedMean(returns, n)
        val cov = Stats.annualizedCovariance(returns, n)

        val sims = (0 until nSimulations).map {
            val w = dirichletUniform(n, random)
            val r = LinAlg.dot(w, mean)
            val v = kotlin.math.sqrt(LinAlg.quadForm(w, cov))
            val sh = (r - rf) / (v + 1e-9)
            SimulatedPortfolio(v * 100, r * 100, sh, w)
        }

        val minRet = sims.minOf { it.retPct }
        val maxRet = sims.maxOf { it.retPct }
        val frontier = mutableListOf<FrontierPoint>()
        for (i in 0 until nFrontierPoints) {
            val targetPct = minRet + (maxRet - minRet) * i / (nFrontierPoints - 1).coerceAtLeast(1)
            val target = targetPct / 100.0
            val w = GenericOptimizer.projectedGradientDescent(n, 0.01, 0.6, iterations = 200) { w ->
                LinAlg.quadForm(w, cov) + GenericOptimizer.returnFloorPenalty(w, mean, target, weight = 200.0) +
                    GenericOptimizer.returnFloorPenalty(DoubleArray(n) { -w[it] }, DoubleArray(n) { -mean[it] }, -target, weight = 200.0)
            }
            val achievedRet = LinAlg.dot(w, mean) * 100
            val vol = kotlin.math.sqrt(LinAlg.quadForm(w, cov)) * 100
            if (kotlin.math.abs(achievedRet - targetPct) < 2.0) frontier.add(FrontierPoint(vol, achievedRet))
        }

        val curRet = LinAlg.dot(weights, mean) * 100
        val curVol = kotlin.math.sqrt(LinAlg.quadForm(weights, cov)) * 100
        val curSharpe = (curRet / 100 - rf) / (curVol / 100 + 1e-9)
        val bestSharpe = sims.maxByOrNull { it.sharpe }

        return EfficientFrontierResult(sims, frontier.sortedBy { it.volPct }, curVol, curRet, curSharpe, bestSharpe)
    }
}
