package ir.marghzari.portfolio360.core.network

import ir.marghzari.portfolio360.core.model.PriceSeries
import kotlinx.datetime.LocalDate

/**
 * Combines several per-ticker histories into one date-aligned [PriceSeries], mirroring
 * `pd.concat(...).ffill().bfill().dropna()` from `fetch_data` in app.py: forward-fill gaps
 * (e.g. crypto trading on days stocks don't), then back-fill any leading gap, then drop any
 * date that still has a hole (only possible if a ticker fetch failed entirely).
 */
object PriceAligner {

    data class AlignResult(val series: PriceSeries?, val failedTickers: List<String>)

    fun align(histories: List<TickerHistory?>, requestedTickers: List<String>): AlignResult {
        val failed = requestedTickers.filterIndexed { i, _ -> histories.getOrNull(i) == null }
        val valid = histories.filterNotNull()
        if (valid.isEmpty()) return AlignResult(null, failed)

        val allDates = sortedSetOf<LocalDate>()
        valid.forEach { allDates.addAll(it.dates) }
        val dateList = allDates.toList()
        val dateIndex = dateList.withIndex().associate { (i, d) -> d to i }

        val nDates = dateList.size
        val tickers = valid.map { it.ticker }
        val grid = Array(nDates) { arrayOfNulls<Double>(tickers.size) }
        for ((col, h) in valid.withIndex()) {
            for (i in h.dates.indices) {
                grid[dateIndex.getValue(h.dates[i])][col] = h.closes[i]
            }
        }

        // Forward-fill then back-fill each column.
        for (col in tickers.indices) {
            var last: Double? = null
            for (row in 0 until nDates) {
                if (grid[row][col] != null) last = grid[row][col] else if (last != null) grid[row][col] = last
            }
            var first: Double? = null
            for (row in nDates - 1 downTo 0) {
                if (grid[row][col] != null) first = grid[row][col] else if (first != null) grid[row][col] = first
            }
        }

        val keepRows = (0 until nDates).filter { row -> grid[row].all { it != null } }
        if (keepRows.isEmpty()) return AlignResult(null, failed)

        val values = Array(keepRows.size) { r -> DoubleArray(tickers.size) { c -> grid[keepRows[r]][c]!! } }
        val dates = keepRows.map { dateList[it] }
        return AlignResult(PriceSeries(dates, tickers, values), failed)
    }
}
