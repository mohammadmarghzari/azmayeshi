package ir.marghzari.portfolio360.core.model

/** Portfolio allocation styles, mirroring the Python `STYLES` dict. */
enum class PortfolioStyle(val faLabel: String) {
    MAX_SHARPE("بیشترین شارپ (Markowitz)"),
    MIN_VARIANCE("کمترین واریانس (Min Variance)"),
    MONTE_CARLO_CVAR("مونت‌کارلو (CVaR)"),
    EQUAL_WEIGHT("وزن برابر (Equal Weight)"),
    RISK_PARITY("ریسک پاریتی (Risk Parity)"),
    TALEB_BARBELL("۹۰/۱۰ طالب (Taleb Barbell)"),
}

enum class HistoryPeriod(val faLabel: String, val apiCode: String) {
    M6("۶ ماه", "6mo"),
    Y1("۱ سال", "1y"),
    Y2("۲ سال", "2y"),
    Y5("۵ سال", "5y"),
    Y10("۱۰ سال", "10y"),
    MAX("حداکثر", "max"),
}

/** Historical daily close-price series, aligned across all included tickers. */
data class PriceSeries(
    val dates: List<kotlinx.datetime.LocalDate>,
    /** tickers in column order matching each row of [values] */
    val tickers: List<String>,
    /** [row=date][col=ticker] */
    val values: Array<DoubleArray>,
) {
    val nAssets get() = tickers.size
    val nDays get() = dates.size

    /** Simple daily returns, one row shorter than [values]. */
    fun dailyReturns(): Array<DoubleArray> = Array(values.size - 1) { i ->
        DoubleArray(nAssets) { j -> values[i + 1][j] / values[i][j] - 1.0 }
    }

    fun column(ticker: String): DoubleArray {
        val idx = tickers.indexOf(ticker)
        require(idx >= 0) { "Unknown ticker $ticker" }
        return DoubleArray(nDays) { values[it][idx] }
    }
}

data class RiskInputs(
    val expectedReturnPct: Double = 0.0,
    val riskGeoPct: Double = 0.0,
    val riskMonPct: Double = 0.0,
    val riskSysPct: Double = 0.0,
)

data class PortfolioMetrics(
    val annualReturn: Double,
    val riskAdjustedReturn: Double,
    val annualVolatility: Double,
    val sharpeRatio: Double,
    val maxDrawdown: Double,
    val cvar95: Double,
    val calmarRatio: Double,
    val recoveryDays: Int,
    val returnGap: Double?,
    val riskDiscountPct: Double,
)

data class WeightsResult(
    val weights: DoubleArray,
    val covariance: Array<DoubleArray>,
)

object TalebSets {
    val SAFE = setOf("GC=F", "GLD", "TLT", "AGG", "EURUSD=X", "GBPUSD=X", "USDCHF=X")
    val RISKY = setOf("BTC-USD", "ETH-USD", "SOL-USD", "AVAX-USD", "NVDA", "TSLA", "ARKK")
}

/** 15-color cyclic chart palette, matching the Python `COLORS` list. */
val ChartPalette = listOf(
    0xFF5B9BD5, 0xFFE8A838, 0xFF3DB87A, 0xFFE05C5C, 0xFF9B72C8,
    0xFF48B8C0, 0xFFD47F3A, 0xFFC45B8E, 0xFF7EB35A, 0xFF5A8FC4,
    0xFFD4A855, 0xFF4DB88C, 0xFFC46060, 0xFF8062B8, 0xFF40A8A8,
).map { it.toInt() }
