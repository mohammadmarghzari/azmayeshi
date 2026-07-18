package ir.marghzari.portfolio360.core.math

import kotlin.math.sqrt

data class RollingMetricsResult(val rollingVolPct: DoubleArray, val rollingReturnPct: DoubleArray, val rollingSharpe: DoubleArray)

/** Rolling-window vol/return/Sharpe over a portfolio's daily-return series, matching the Efficient Frontier tab. */
object RollingMetrics {
    fun compute(dailyReturns: DoubleArray, window: Int, rf: Double): RollingMetricsResult {
        val n = dailyReturns.size
        val vol = DoubleArray(n) { Double.NaN }
        val ret = DoubleArray(n) { Double.NaN }
        val sharpe = DoubleArray(n) { Double.NaN }
        for (i in window until n + 1) {
            val w = dailyReturns.sliceArray(i - window until i)
            val v = Stats.std(w) * sqrt(252.0) * 100.0
            val r = Stats.mean(w) * 252.0 * 100.0
            vol[i - 1] = v
            ret[i - 1] = r
            sharpe[i - 1] = (r / 100.0 - rf) / (v / 100.0 + 1e-9)
        }
        return RollingMetricsResult(vol, ret, sharpe)
    }
}
