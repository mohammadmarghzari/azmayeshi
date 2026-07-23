package ir.marghzari.portfolio360.data

import ir.marghzari.portfolio360.core.network.YahooFinanceClient
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope

/** One entry of the curated market board; every symbol resolves on Yahoo's daily-close endpoint. */
data class MarketEntry(val symbol: String, val labelFa: String, val category: String)

/** Curated cross-market board shared by the Markets tab and the dashboard's movers section. */
val MARKET_ENTRIES = listOf(
    MarketEntry("BTC-USD", "بیت‌کوین", "ارزهای دیجیتال"),
    MarketEntry("ETH-USD", "اتریوم", "ارزهای دیجیتال"),
    MarketEntry("SOL-USD", "سولانا", "ارزهای دیجیتال"),
    MarketEntry("BNB-USD", "بایننس‌کوین", "ارزهای دیجیتال"),
    MarketEntry("SPY", "شاخص S&P 500", "بازار آمریکا"),
    MarketEntry("QQQ", "شاخص Nasdaq 100", "بازار آمریکا"),
    MarketEntry("AAPL", "اپل", "بازار آمریکا"),
    MarketEntry("NVDA", "انویدیا", "بازار آمریکا"),
    MarketEntry("GLD", "صندوق طلا", "کالاها و ارز"),
    MarketEntry("SLV", "صندوق نقره", "کالاها و ارز"),
    MarketEntry("USO", "صندوق نفت", "کالاها و ارز"),
    MarketEntry("EURUSD=X", "یورو / دلار", "کالاها و ارز"),
)

/** Daily-close history of one board entry with the derived headline numbers screens display. */
data class MarketSnapshot(val entry: MarketEntry, val closes: List<Double>) {
    val last: Double get() = closes.last()
    val dailyPct: Double
        get() {
            val prev = closes.getOrNull(closes.size - 2) ?: return 0.0
            return if (prev != 0.0) (last / prev - 1) * 100 else 0.0
        }
}

/**
 * Fetches all [entries] concurrently, silently dropping symbols that fail or return too little
 * history — an empty result therefore means "nothing reachable" and screens map it to an error
 * state. The Yahoo client's internal TTL cache makes repeated calls (dashboard + markets tab in
 * one session) cost a single network round-trip per symbol.
 */
suspend fun fetchMarketSnapshots(
    yahoo: YahooFinanceClient,
    entries: List<MarketEntry> = MARKET_ENTRIES,
): List<MarketSnapshot> = coroutineScope {
    entries.map { entry ->
        async {
            yahoo.fetchHistory(entry.symbol, "6mo")
                ?.takeIf { it.closes.size >= 2 }
                ?.let { MarketSnapshot(entry, it.closes) }
        }
    }.mapNotNull { it.await() }
}
