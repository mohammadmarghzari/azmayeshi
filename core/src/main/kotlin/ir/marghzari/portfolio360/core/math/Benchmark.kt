package ir.marghzari.portfolio360.core.math

import kotlinx.datetime.LocalDate
import kotlin.math.sqrt

data class BenchmarkComparison(
    val portfolioAnnualReturnPct: Double, val benchmarkAnnualReturnPct: Double,
    val portfolioVolPct: Double, val benchmarkVolPct: Double,
    val alphaPct: Double, val beta: Double, val trackingErrorPct: Double, val informationRatio: Double,
    val dates: List<LocalDate>, val portfolioCumulative: DoubleArray, val benchmarkCumulative: DoubleArray,
    val outperform: Boolean,
)

/** Direct port of `compare_to_benchmark` from app.py. Both series must already be aligned on [dates]. */
object Benchmark {
    fun compare(dates: List<LocalDate>, portfolioDailyReturns: DoubleArray, benchmarkDailyReturns: DoubleArray): BenchmarkComparison? {
        if (dates.size < 20) return null
        val p = portfolioDailyReturns
        val b = benchmarkDailyReturns
        val portAnn = Stats.annualizedReturn(p)
        val benchAnn = Stats.annualizedReturn(b)
        val portVol = Stats.annualizedVol(p)
        val benchVol = Stats.annualizedVol(b)
        val covPb = Stats.covariance(p, b)
        val varB = Stats.covariance(b, b)
        val beta = covPb / (varB + 1e-9)
        val alpha = portAnn - beta * benchAnn
        val diff = DoubleArray(p.size) { p[it] - b[it] }
        val te = Stats.std(diff) * sqrt(252.0)
        val ir = alpha / (te + 1e-9)
        return BenchmarkComparison(
            portfolioAnnualReturnPct = portAnn * 100, benchmarkAnnualReturnPct = benchAnn * 100,
            portfolioVolPct = portVol * 100, benchmarkVolPct = benchVol * 100,
            alphaPct = alpha * 100, beta = beta, trackingErrorPct = te * 100, informationRatio = ir,
            dates = dates, portfolioCumulative = Stats.cumulative(p), benchmarkCumulative = Stats.cumulative(b),
            outperform = portAnn > benchAnn,
        )
    }
}
