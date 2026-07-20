package ir.marghzari.portfolio360.data

import com.russhwolf.settings.Settings
import ir.marghzari.portfolio360.core.model.PortfolioStyle
import ir.marghzari.portfolio360.state.AppState
import ir.marghzari.portfolio360.state.FearGreedAlertConfig
import ir.marghzari.portfolio360.state.PriceAlert
import ir.marghzari.portfolio360.state.Transaction
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json

/**
 * Everything the app promises to remember between launches, captured as one serializable value —
 * user-entered data (trade journal, alerts) and preferences (theme, motion, defaults, selections).
 * Computed results (prices/weights/metrics) are deliberately excluded: they are cheap to recompute
 * and stale market data presented as current would be worse than none. Saved portfolios stay
 * session-scoped for now (their metrics types are not yet serializable) — tracked as roadmap debt.
 */
@Serializable
data class PersistedSnapshot(
    val isDarkTheme: Boolean,
    val reducedMotion: Boolean,
    val periodCode: String,
    val riskFreeRatePct: Double,
    val portfolioStyle: String,
    val selectedTickers: List<String>,
    val favoriteTickers: Set<String>,
    val transactions: List<Transaction>,
    val priceAlerts: List<PriceAlert>,
    val fearGreedAlertConfig: FearGreedAlertConfig,
)

/**
 * Durable storage for [PersistedSnapshot], stored as one JSON document under a versioned key via
 * `multiplatform-settings` (SharedPreferences on Android, java.util.prefs on desktop — the no-arg
 * factory resolves the right backend per platform without any expect/actual, which also keeps the
 * local devpreview sandbox compiling). A corrupt or missing document restores to defaults silently:
 * losing preferences must never crash the app at startup.
 */
class AppPersistence(private val settings: Settings = Settings()) {
    private val json = Json { ignoreUnknownKeys = true; encodeDefaults = true }

    fun capture(state: AppState) = PersistedSnapshot(
        isDarkTheme = state.isDarkTheme,
        reducedMotion = state.reducedMotion,
        periodCode = state.periodCode,
        riskFreeRatePct = state.riskFreeRatePct,
        portfolioStyle = state.portfolioStyle.name,
        selectedTickers = state.selectedTickers,
        favoriteTickers = state.favoriteTickers,
        transactions = state.transactions,
        priceAlerts = state.priceAlerts,
        fearGreedAlertConfig = state.fearGreedAlertConfig,
    )

    fun save(snapshot: PersistedSnapshot) {
        runCatching { settings.putString(KEY, json.encodeToString(PersistedSnapshot.serializer(), snapshot)) }
    }

    fun restore(state: AppState) {
        val raw = settings.getStringOrNull(KEY) ?: return
        val snapshot = runCatching { json.decodeFromString(PersistedSnapshot.serializer(), raw) }.getOrNull() ?: return
        state.isDarkTheme = snapshot.isDarkTheme
        state.reducedMotion = snapshot.reducedMotion
        state.periodCode = snapshot.periodCode
        state.riskFreeRatePct = snapshot.riskFreeRatePct
        state.portfolioStyle = PortfolioStyle.entries.firstOrNull { it.name == snapshot.portfolioStyle } ?: state.portfolioStyle
        state.selectedTickers = snapshot.selectedTickers
        state.favoriteTickers = snapshot.favoriteTickers
        state.transactions = snapshot.transactions
        state.priceAlerts = snapshot.priceAlerts
        state.fearGreedAlertConfig = snapshot.fearGreedAlertConfig
    }

    private companion object {
        const val KEY = "portfolio360_state_v1"
    }
}
