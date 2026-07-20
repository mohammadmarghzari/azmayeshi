package ir.marghzari.portfolio360.state

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import ir.marghzari.portfolio360.core.math.CoveredCallResult
import ir.marghzari.portfolio360.core.model.PortfolioMetrics
import ir.marghzari.portfolio360.core.model.PortfolioStyle
import ir.marghzari.portfolio360.core.model.PriceSeries
import ir.marghzari.portfolio360.core.model.RiskInputs
import ir.marghzari.portfolio360.data.MarketRepositories

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

/** One row of the user's manual trade journal ("دفتر تراکنش‌ها"). */
data class Transaction(
    val id: Long,
    val symbol: String,
    val isBuy: Boolean,
    val quantity: Double,
    val price: Double,
    val dateLabel: String,
    val note: String = "",
)

data class FearGreedAlertConfig(val lowerBound: Int = 20, val upperBound: Int = 80, var lastScore: Double? = null)

/**
 * The single shared application state, mirroring Streamlit's `st.session_state` in app.py: values
 * computed on one tab (prices, weights, metrics, hedges, saved portfolios) are read by several others.
 *
 * Structurally this is now a facade: the data layer lives in [MarketRepositories] and the
 * fetch/optimize pipeline's outputs in [PortfolioSession]; the delegating members below keep every
 * existing `appState.x` call site compiling unchanged while new code can take the narrower types.
 */
class AppState(
    val repositories: MarketRepositories = MarketRepositories(),
    val portfolio: PortfolioSession = PortfolioSession(),
) {
    // Data layer accessors (see MarketRepositories for why these live outside the UI state).
    val yahoo get() = repositories.yahoo
    val fearGreed get() = repositories.fearGreed
    val news get() = repositories.news
    val ime get() = repositories.ime
    val worldCommodities get() = repositories.worldCommodities

    // Theme
    var isDarkTheme by mutableStateOf(true)

    // Sidebar / global settings
    var periodCode by mutableStateOf("2y")
    var riskFreeRatePct by mutableStateOf(5.0)
    var selectedTickers by mutableStateOf(listOf<String>())
    var portfolioStyle by mutableStateOf(PortfolioStyle.MAX_SHARPE)
    var riskInputs by mutableStateOf(RiskInputs())

    // Computed portfolio (set after "محاسبه پرتفوی") — held by PortfolioSession.
    var prices: PriceSeries? by portfolio::prices
    var fetchFailedTickers: List<String> by portfolio::fetchFailedTickers
    var weights: DoubleArray? by portfolio::weights
    var covariance: Array<DoubleArray>? by portfolio::covariance
    var metrics: PortfolioMetrics? by portfolio::metrics
    var styleLabelUsed: String by portfolio::styleLabelUsed
    var lastUsedRisk: RiskInputs by portfolio::lastUsedRisk

    var isFetching: Boolean by portfolio::isFetching
    var isCalculating: Boolean by portfolio::isCalculating
    var lastError: String? by portfolio::lastError

    // Hedges applied from the Protective Put tool onto the main portfolio (tab5 <-> tab2 interaction).
    var hedgedAssets by mutableStateOf<Map<String, HedgedAsset>>(emptyMap())

    // Saved portfolios (tab_save)
    var savedPortfolios by mutableStateOf<Map<String, SavedPortfolio>>(emptyMap())

    // Alerts (tab_alert)
    var priceAlerts by mutableStateOf<List<PriceAlert>>(emptyList())
    var fearGreedAlertConfig by mutableStateOf(FearGreedAlertConfig())

    // Covered-call quick tool (sidebar in the original app)
    var lastCoveredCallResult by mutableStateOf<CoveredCallResult?>(null)

    // Starred tickers on the asset price screen
    var favoriteTickers by mutableStateOf(setOf<String>())

    // Manual trade journal (فاز ۵ — دفتر تراکنش‌ها). In-memory for now, like savedPortfolios.
    var transactions by mutableStateOf<List<Transaction>>(emptyList())

    // Global "reduce motion" switch for the app-wide decorative motion system.
    var reducedMotion by mutableStateOf(false)

    val rf: Double get() = riskFreeRatePct / 100.0

    fun resetComputedPortfolio() = portfolio.resetComputed()
}
