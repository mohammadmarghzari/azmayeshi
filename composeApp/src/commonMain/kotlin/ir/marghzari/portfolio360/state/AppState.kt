package ir.marghzari.portfolio360.state

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import ir.marghzari.portfolio360.core.math.CoveredCallResult
import ir.marghzari.portfolio360.core.model.PortfolioMetrics
import ir.marghzari.portfolio360.core.model.PortfolioStyle
import ir.marghzari.portfolio360.core.model.PriceSeries
import ir.marghzari.portfolio360.core.model.RiskInputs
import ir.marghzari.portfolio360.core.network.FearGreedClient
import ir.marghzari.portfolio360.core.network.ImeClient
import ir.marghzari.portfolio360.core.network.NewsRssClient
import ir.marghzari.portfolio360.core.network.WorldCommodityClient
import ir.marghzari.portfolio360.core.network.YahooFinanceClient

/** Portfolio saved by the user in the "ذخیره پرتفوی" tab. */
data class SavedPortfolio(
    val name: String,
    val weights: DoubleArray,
    val tickers: List<String>,
    val metrics: PortfolioMetrics,
    val style: String,
    val risk: RiskInputs,
    val savedAtLabel: String,
)

data class PriceAlert(val symbol: String, val isAbove: Boolean, val threshold: Double, var currentPrice: Double? = null, var triggered: Boolean = false)

data class HedgedAsset(val ticker: String, val strike: Double, val spot: Double, val premium: Double, val days: Int)

data class FearGreedAlertConfig(val lowerBound: Int = 20, val upperBound: Int = 80, var lastScore: Double? = null)

/**
 * The single shared application state, mirroring Streamlit's `st.session_state` in app.py: values
 * computed on one tab (prices, weights, metrics, hedges, saved portfolios) are read by several others.
 */
class AppState {
    // Networking (shared singletons; each has its own internal TTL cache).
    val yahoo = YahooFinanceClient()
    val fearGreed = FearGreedClient()
    val news = NewsRssClient()
    val ime = ImeClient()
    val worldCommodities = WorldCommodityClient()

    // Theme
    var isDarkTheme by mutableStateOf(true)

    // Sidebar / global settings
    var periodCode by mutableStateOf("2y")
    var riskFreeRatePct by mutableStateOf(5.0)
    var selectedTickers by mutableStateOf(listOf<String>())
    var portfolioStyle by mutableStateOf(PortfolioStyle.MAX_SHARPE)
    var riskInputs by mutableStateOf(RiskInputs())

    // Computed portfolio (set after "محاسبه پرتفوی")
    var prices by mutableStateOf<PriceSeries?>(null)
    var fetchFailedTickers by mutableStateOf<List<String>>(emptyList())
    var weights by mutableStateOf<DoubleArray?>(null)
    var covariance by mutableStateOf<Array<DoubleArray>?>(null)
    var metrics by mutableStateOf<PortfolioMetrics?>(null)
    var styleLabelUsed by mutableStateOf("")
    var lastUsedRisk by mutableStateOf(RiskInputs())

    var isFetching by mutableStateOf(false)
    var isCalculating by mutableStateOf(false)
    var lastError by mutableStateOf<String?>(null)

    // Hedges applied from the Protective Put tool onto the main portfolio (tab5 <-> tab2 interaction).
    var hedgedAssets by mutableStateOf<Map<String, HedgedAsset>>(emptyMap())

    // Saved portfolios (tab_save)
    var savedPortfolios by mutableStateOf<Map<String, SavedPortfolio>>(emptyMap())

    // Alerts (tab_alert)
    var priceAlerts by mutableStateOf<List<PriceAlert>>(emptyList())
    var fearGreedAlertConfig by mutableStateOf(FearGreedAlertConfig())

    // Covered-call quick tool (sidebar in the original app)
    var lastCoveredCallResult by mutableStateOf<CoveredCallResult?>(null)

    val rf: Double get() = riskFreeRatePct / 100.0

    fun resetComputedPortfolio() {
        weights = null; covariance = null; metrics = null
    }
}
