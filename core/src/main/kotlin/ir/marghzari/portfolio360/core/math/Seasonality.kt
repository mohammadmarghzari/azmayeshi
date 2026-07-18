package ir.marghzari.portfolio360.core.math

import kotlinx.datetime.LocalDate

data class MonthlyReturn(val year: Int, val month: Int, val returnPct: Double)
data class SeasonalityMonthStat(val month: Int, val avgReturnPct: Double, val stdReturnPct: Double, val positiveRatePct: Double)

/** Direct port of the "Seasonality" section in tab_live: month-end resample -> pct_change -> group by calendar month. */
object Seasonality {

    fun monthlyReturns(dates: List<LocalDate>, prices: DoubleArray): List<MonthlyReturn> {
        if (dates.isEmpty()) return emptyList()
        // Last observation per (year, month), in chronological order.
        val monthEnds = LinkedHashMap<Pair<Int, Int>, Double>()
        for (i in dates.indices) {
            monthEnds[dates[i].year to dates[i].monthNumber] = prices[i]
        }
        val entries = monthEnds.entries.toList()
        val out = mutableListOf<MonthlyReturn>()
        for (i in 1 until entries.size) {
            val prev = entries[i - 1].value
            val cur = entries[i].value
            if (prev == 0.0) continue
            out.add(MonthlyReturn(entries[i].key.first, entries[i].key.second, (cur / prev - 1.0) * 100.0))
        }
        return out
    }

    fun byMonth(monthly: List<MonthlyReturn>): List<SeasonalityMonthStat> {
        return (1..12).map { m ->
            val vals = monthly.filter { it.month == m }.map { it.returnPct }
            if (vals.isEmpty()) {
                SeasonalityMonthStat(m, 0.0, 0.0, 0.0)
            } else {
                val avg = vals.average()
                val std = Stats.std(vals.toDoubleArray())
                val posRate = vals.count { it > 0 } * 100.0 / vals.size
                SeasonalityMonthStat(m, avg, std, posRate)
            }
        }
    }

    val MONTH_ABBREV_EN = listOf("Jan", "Feb", "Mar", "Apr", "May", "Jun", "Jul", "Aug", "Sep", "Oct", "Nov", "Dec")
}
