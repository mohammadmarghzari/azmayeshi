package ir.marghzari.portfolio360.core.math

import kotlin.math.min
import kotlin.math.sqrt

data class FactorRow(val ticker: String, val momentum6m: Double, val annualVol: Double, val beta: Double, val sharpe: Double)

/** Direct port of `compute_factor_exposure` from app.py. */
object FactorExposure {
    fun compute(returns: Array<DoubleArray>, tickers: List<String>): List<FactorRow> {
        val nDays = returns.size
        val n = tickers.size
        val market = DoubleArray(nDays) { d -> returns[d].average() }
        val tailStart = maxOf(0, nDays - 126)
        return tickers.indices.map { j ->
            val col = DoubleArray(nDays) { returns[it][j] }
            var momProd = 1.0
            for (d in tailStart until nDays) momProd *= (1 + col[d])
            val momentum = (momProd - 1.0) * 100
            val vol = Stats.std(col) * sqrt(252.0) * 100
            val beta = Stats.covariance(col, market) / (Stats.populationVariance(market) + 1e-9)
            val sharpe = (Stats.mean(col) / (Stats.std(col) + 1e-9)) * sqrt(252.0)
            FactorRow(tickers[j], momentum, vol, beta, sharpe)
        }
    }
}
