package ir.marghzari.portfolio360.core.math

import kotlinx.datetime.LocalDate
import kotlin.math.sqrt

data class CrisisPeriod(val nameFa: String, val start: LocalDate, val end: LocalDate)

data class StressTestRow(val crisisName: String, val totalReturnPct: Double, val annualVolPct: Double, val maxDrawdownPct: Double, val days: Int)

/** Direct port of `CRISIS_PERIODS` + `run_stress_tests` from app.py: replays actual historical returns, not hypothetical shocks. */
object StressTest {
    val CRISIS_PERIODS = listOf(
        CrisisPeriod("بحران مالی ۲۰۰۸", LocalDate(2008, 9, 1), LocalDate(2009, 3, 31)),
        CrisisPeriod("Flash Crash 2010", LocalDate(2010, 4, 23), LocalDate(2010, 7, 2)),
        CrisisPeriod("بحران اروپا ۲۰۱۱", LocalDate(2011, 7, 1), LocalDate(2011, 10, 3)),
        CrisisPeriod("افت چین ۲۰۱۵", LocalDate(2015, 6, 12), LocalDate(2015, 9, 29)),
        CrisisPeriod("کرونا ۲۰۲۰", LocalDate(2020, 2, 19), LocalDate(2020, 3, 23)),
        CrisisPeriod("افت تورم ۲۰۲۲", LocalDate(2022, 1, 1), LocalDate(2022, 10, 13)),
    )

    fun run(dates: List<LocalDate>, prices: Array<DoubleArray>, tickers: List<String>, weights: DoubleArray): List<StressTestRow> {
        val rows = mutableListOf<StressTestRow>()
        for (crisis in CRISIS_PERIODS) {
            val idx = dates.indices.filter { dates[it] >= crisis.start && dates[it] <= crisis.end }
            if (idx.size < 5) continue
            val subPrices = idx.map { prices[it] }
            val subReturns = Array(subPrices.size - 1) { i -> DoubleArray(tickers.size) { j -> subPrices[i + 1][j] / subPrices[i][j] - 1.0 } }
            if (subReturns.isEmpty()) continue
            val wSum = weights.sum()
            val wNorm = DoubleArray(weights.size) { weights[it] / wSum }
            val portRet = Stats.portfolioReturns(subReturns, wNorm)
            val cum = ((portRet.fold(1.0) { acc, r -> acc * (1 + r) }) - 1.0)
            val vol = Stats.std(portRet) * sqrt(252.0)
            val cumCurve = Stats.cumulative(portRet)
            val dd = Stats.maxDrawdown(cumCurve)
            rows.add(StressTestRow(crisis.nameFa, cum * 100, vol * 100, dd * 100, portRet.size))
        }
        return rows
    }
}
