package ir.marghzari.portfolio360.data

import ir.marghzari.portfolio360.core.network.FearGreedClient
import ir.marghzari.portfolio360.core.network.ImeClient
import ir.marghzari.portfolio360.core.network.NewsRssClient
import ir.marghzari.portfolio360.core.network.WorldCommodityClient
import ir.marghzari.portfolio360.core.network.YahooFinanceClient

/**
 * The app's data layer in one place: every remote client (each with its own internal TTL cache)
 * lives here instead of inside the UI state object. This is the seam where persistence, request
 * de-duplication, or a test double can be introduced later without touching any screen — screens
 * keep reaching these through [ir.marghzari.portfolio360.state.AppState]'s accessors.
 * Constructed once per app and passed into `AppState` (manual DI; Hilt is Android-only and this
 * module is Kotlin Multiplatform).
 */
class MarketRepositories(
    val yahoo: YahooFinanceClient = YahooFinanceClient(),
    val fearGreed: FearGreedClient = FearGreedClient(),
    val news: NewsRssClient = NewsRssClient(),
    val ime: ImeClient = ImeClient(),
    val worldCommodities: WorldCommodityClient = WorldCommodityClient(),
)
