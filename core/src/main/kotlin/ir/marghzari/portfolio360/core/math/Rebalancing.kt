package ir.marghzari.portfolio360.core.math

data class RebalanceRow(
    val ticker: String, val targetWeightPct: Double, val currentWeightPct: Double,
    val driftPct: Double, val tradeDollar: Double, val needsRebalance: Boolean,
)

/** Direct port of `calc_rebalancing` from app.py. */
object Rebalancing {
    fun compute(
        tickers: List<String>,
        currentPrices: DoubleArray,
        targetWeights: DoubleArray,
        totalCapital: Double,
        threshold: Double = 0.05,
    ): List<RebalanceRow> {
        val total = currentPrices.sum()
        if (total <= 0.0) return emptyList()
        return tickers.indices.map { i ->
            val tw = targetWeights[i]
            val cw = currentPrices[i] / total
            val drift = cw - tw
            val trade = (tw - cw) * totalCapital
            RebalanceRow(tickers[i], tw * 100, cw * 100, drift * 100, trade, kotlin.math.abs(drift) > threshold)
        }
    }
}
