package ir.marghzari.portfolio360.core.math

import kotlin.math.sqrt
import kotlin.random.Random

data class MonteCarloResult(
    val pct5: DoubleArray, val pct25: DoubleArray, val pct50: DoubleArray, val pct75: DoubleArray, val pct95: DoubleArray,
    val final: DoubleArray,
    val probProfitPct: Double, val prob2xPct: Double, val median: Double, val worst5: Double, val best5: Double,
    val nDays: Int,
)

/** Direct port of `monte_carlo_future`: iid-Normal daily-return simulation, matching the source app exactly. */
object MonteCarloEngine {

    private fun gaussian(random: Random, mu: Double, sigma: Double): Double {
        // Box-Muller transform.
        val u1 = 1.0 - random.nextDouble()
        val u2 = random.nextDouble()
        val z0 = sqrt(-2.0 * kotlin.math.ln(u1)) * kotlin.math.cos(2.0 * kotlin.math.PI * u2)
        return mu + sigma * z0
    }

    fun simulate(
        weights: DoubleArray,
        returns: Array<DoubleArray>,
        nSims: Int = 400,
        horizonYears: Double = 3.0,
        random: Random = Random.Default,
    ): MonteCarloResult {
        val portRet = Stats.portfolioReturns(returns, weights)
        val mu = Stats.mean(portRet)
        val sigma = Stats.std(portRet)
        val nDays = (horizonYears * 252).toInt()

        val paths = Array(nSims) { DoubleArray(nDays) }
        for (i in 0 until nSims) {
            var acc = 1.0
            for (d in 0 until nDays) {
                acc *= (1.0 + gaussian(random, mu, sigma))
                paths[i][d] = acc
            }
        }
        val final = DoubleArray(nSims) { paths[it][nDays - 1] }

        fun percentileByDay(p: Double): DoubleArray = DoubleArray(nDays) { d ->
            Stats.percentile(DoubleArray(nSims) { s -> paths[s][d] }, p)
        }

        return MonteCarloResult(
            pct5 = percentileByDay(5.0), pct25 = percentileByDay(25.0), pct50 = percentileByDay(50.0),
            pct75 = percentileByDay(75.0), pct95 = percentileByDay(95.0),
            final = final,
            probProfitPct = final.count { it > 1.0 } * 100.0 / nSims,
            prob2xPct = final.count { it > 2.0 } * 100.0 / nSims,
            median = Stats.percentile(final, 50.0),
            worst5 = Stats.percentile(final, 5.0),
            best5 = Stats.percentile(final, 95.0),
            nDays = nDays,
        )
    }
}
