package ir.marghzari.portfolio360.core.math

import kotlin.math.pow
import kotlin.math.sqrt

/** Basic statistics over daily-return matrices, mirroring the pandas operations used in the source app. */
object Stats {

    fun mean(x: DoubleArray): Double = if (x.isEmpty()) 0.0 else x.sum() / x.size

    /** Sample standard deviation (ddof=1), matching pandas default `.std()`. */
    fun std(x: DoubleArray): Double {
        if (x.size < 2) return 0.0
        val m = mean(x)
        val ss = x.sumOf { (it - m).pow(2) }
        return sqrt(ss / (x.size - 1))
    }

    /** Sample covariance (ddof=1) between two series, matching pandas `.cov()`. */
    fun covariance(x: DoubleArray, y: DoubleArray): Double {
        val n = x.size
        if (n < 2) return 0.0
        val mx = mean(x); val my = mean(y)
        var s = 0.0
        for (i in 0 until n) s += (x[i] - mx) * (y[i] - my)
        return s / (n - 1)
    }

    /** Population variance (ddof=0), matching numpy's default `np.var`. */
    fun populationVariance(x: DoubleArray): Double {
        if (x.isEmpty()) return 0.0
        val m = mean(x)
        return x.sumOf { (it - m).pow(2) } / x.size
    }

    fun correlation(x: DoubleArray, y: DoubleArray): Double {
        val sx = std(x); val sy = std(y)
        if (sx < 1e-12 || sy < 1e-12) return 0.0
        return covariance(x, y) / (sx * sy)
    }

    /** Linear-interpolated percentile, matching numpy's default `np.percentile`. */
    fun percentile(x: DoubleArray, p: Double): Double {
        if (x.isEmpty()) return 0.0
        val sorted = x.sortedArray()
        if (sorted.size == 1) return sorted[0]
        val rank = (p / 100.0) * (sorted.size - 1)
        val lo = rank.toInt().coerceIn(0, sorted.size - 1)
        val hi = (lo + 1).coerceAtMost(sorted.size - 1)
        val frac = rank - lo
        return sorted[lo] + (sorted[hi] - sorted[lo]) * frac
    }

    /** Returns matrix (rows = time, cols = assets) -> annualized covariance matrix (x252). */
    fun annualizedCovariance(returns: Array<DoubleArray>, nAssets: Int, factor: Double = 252.0): Array<DoubleArray> {
        val cols = Array(nAssets) { j -> DoubleArray(returns.size) { i -> returns[i][j] } }
        return Array(nAssets) { i -> DoubleArray(nAssets) { j -> covariance(cols[i], cols[j]) * factor } }
    }

    fun annualizedMean(returns: Array<DoubleArray>, nAssets: Int, factor: Double = 252.0): DoubleArray {
        val cols = Array(nAssets) { j -> DoubleArray(returns.size) { i -> returns[i][j] } }
        return DoubleArray(nAssets) { mean(cols[it]) * factor }
    }

    /** Portfolio daily-return series = returns @ weights. */
    fun portfolioReturns(returns: Array<DoubleArray>, weights: DoubleArray): DoubleArray =
        DoubleArray(returns.size) { i -> LinAlg.dot(returns[i], weights) }

    /** Annualized compounded return from a daily-return series, matching `(1+r).prod()**(252/n) - 1`. */
    fun annualizedReturn(dailyReturns: DoubleArray, periodsPerYear: Double = 252.0): Double {
        if (dailyReturns.isEmpty()) return 0.0
        var logSum = 0.0
        for (r in dailyReturns) logSum += kotlin.math.ln(1.0 + r)
        val total = kotlin.math.exp(logSum)
        return total.pow(periodsPerYear / dailyReturns.size) - 1.0
    }

    fun annualizedVol(dailyReturns: DoubleArray, periodsPerYear: Double = 252.0): Double =
        std(dailyReturns) * sqrt(periodsPerYear)

    /** Cumulative product curve (1+r).cumprod(). */
    fun cumulative(dailyReturns: DoubleArray): DoubleArray {
        var acc = 1.0
        return DoubleArray(dailyReturns.size) { i -> acc *= (1.0 + dailyReturns[i]); acc }
    }

    /** Max drawdown (negative fraction) and the day-index series of the underwater curve. */
    fun drawdownSeries(cum: DoubleArray): DoubleArray {
        var peak = Double.NEGATIVE_INFINITY
        return DoubleArray(cum.size) { i ->
            peak = maxOf(peak, cum[i])
            if (peak > 0) (cum[i] - peak) / peak else 0.0
        }
    }

    fun maxDrawdown(cum: DoubleArray): Double = drawdownSeries(cum).minOrNull() ?: 0.0

    /** Longest consecutive run of days where drawdown < -1%, matching the Python "recovery time" heuristic. */
    fun longestUnderwaterStreak(dd: DoubleArray, threshold: Double = -0.01): Int {
        var longest = 0
        var current = 0
        for (v in dd) {
            if (v < threshold) { current++; longest = maxOf(longest, current) } else current = 0
        }
        return longest
    }
}
