package ir.marghzari.portfolio360.core.math

import kotlinx.datetime.LocalDate

data class CorrelationRegimePoint(val date: LocalDate, val corrShort: Double, val corrLong: Double, val signal: Boolean)
data class CorrelationRegimeResult(val points: List<CorrelationRegimePoint>, val regimeLabelFa: String)

/** Direct port of `detect_correlation_regime` from app.py. */
object CorrelationRegime {

    private fun avgOffDiagCorrelation(window: Array<DoubleArray>, nAssets: Int): Double {
        if (window.size < 2) return Double.NaN
        val cols = Array(nAssets) { j -> DoubleArray(window.size) { i -> window[i][j] } }
        var sum = 0.0
        var count = 0
        for (i in 0 until nAssets) {
            for (j in 0 until nAssets) {
                if (i == j) continue
                val c = Stats.correlation(cols[i], cols[j])
                if (!c.isNaN()) { sum += c; count++ }
            }
        }
        return if (count > 0) sum / count else Double.NaN
    }

    fun detect(
        dates: List<LocalDate>,
        returns: Array<DoubleArray>,
        nAssets: Int,
        windowShort: Int = 30,
        windowLong: Int = 126,
    ): CorrelationRegimeResult {
        if (nAssets < 2) return CorrelationRegimeResult(emptyList(), "⚠ حداقل ۲ دارایی برای تشخیص رژیم لازم است")
        val points = mutableListOf<CorrelationRegimePoint>()
        for (i in windowLong until returns.size) {
            val shortWindow = returns.sliceArray(i - windowShort until i)
            val longWindow = returns.sliceArray(i - windowLong until i)
            val cs = avgOffDiagCorrelation(shortWindow, nAssets)
            val cl = avgOffDiagCorrelation(longWindow, nAssets)
            val signal = cs > cl + 0.10
            points.add(CorrelationRegimePoint(dates[i], cs, cl, signal))
        }
        val regime = if (points.isNotEmpty() && points.last().signal) {
            "🔴 بحران — همبستگی‌ها بالا رفته (هشدار تنوع‌بخشی)"
        } else {
            "🟢 عادی — همبستگی در محدوده نرمال"
        }
        return CorrelationRegimeResult(points, regime)
    }
}
