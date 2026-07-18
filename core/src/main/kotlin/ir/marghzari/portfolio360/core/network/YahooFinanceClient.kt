package ir.marghzari.portfolio360.core.network

import io.ktor.client.call.body
import io.ktor.client.request.get
import io.ktor.client.request.header
import io.ktor.client.statement.HttpResponse
import kotlinx.datetime.Instant
import kotlinx.datetime.TimeZone
import kotlinx.datetime.toLocalDateTime
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable

@Serializable
private data class YahooChartResponse(val chart: YahooChart)

@Serializable
private data class YahooChart(val result: List<YahooResult>? = null, val error: YahooError? = null)

@Serializable
private data class YahooError(val description: String? = null)

@Serializable
private data class YahooResult(
    val timestamp: List<Long>? = null,
    val indicators: YahooIndicators,
)

@Serializable
private data class YahooIndicators(
    val quote: List<YahooQuote> = emptyList(),
    val adjclose: List<YahooAdjClose>? = null,
)

@Serializable
private data class YahooQuote(val close: List<Double?>? = null)

@Serializable
private data class YahooAdjClose(val adjclose: List<Double?>? = null)

data class TickerHistory(val ticker: String, val dates: List<kotlinx.datetime.LocalDate>, val closes: List<Double>)

/**
 * Fetches daily historical prices from Yahoo Finance's public chart endpoint — the same underlying
 * data source the Python app reaches through `yfinance`, with `auto_adjust=True` reproduced by
 * preferring the `adjclose` series when present.
 */
class YahooFinanceClient(
    private val cache: TtlCache<TickerHistory?> = TtlCache(),
) {
    /** @param rangeCode one of 6mo/1y/2y/5y/10y/max/2d, matching Yahoo's `range` query param. */
    suspend fun fetchHistory(ticker: String, rangeCode: String, ttlMs: Long = 3_600_000): TickerHistory? {
        return cache.getOrPut("$ticker|$rangeCode", ttlMs) { fetchNow(ticker, rangeCode) }
    }

    private suspend fun fetchNow(ticker: String, rangeCode: String): TickerHistory? {
        return try {
            val response: HttpResponse = HttpClientProvider.client.get(
                "https://query1.finance.yahoo.com/v8/finance/chart/${ticker}"
            ) {
                url {
                    parameters.append("range", rangeCode)
                    parameters.append("interval", "1d")
                }
                header("Accept", "application/json")
            }
            if (!response.status.value.let { it in 200..299 }) return null
            val body: YahooChartResponse = response.body()
            val result = body.chart.result?.firstOrNull() ?: return null
            val timestamps = result.timestamp ?: return null
            val adjClose = result.indicators.adjclose?.firstOrNull()?.adjclose
            val rawClose = result.indicators.quote.firstOrNull()?.close
            val closesRaw = adjClose ?: rawClose ?: return null

            val dates = mutableListOf<kotlinx.datetime.LocalDate>()
            val closes = mutableListOf<Double>()
            for (i in timestamps.indices) {
                val c = closesRaw.getOrNull(i) ?: continue
                if (c.isNaN()) continue
                dates.add(Instant.fromEpochSeconds(timestamps[i]).toLocalDateTime(TimeZone.UTC).date)
                closes.add(c)
            }
            if (closes.size < 10) return null
            TickerHistory(ticker, dates, closes)
        } catch (e: Exception) {
            null
        }
    }

    /** Latest close over a short lookback window — used for gold/commodity spot references. */
    suspend fun fetchLastClose(ticker: String, ttlMs: Long = 1_800_000): Double? {
        val h = fetchHistory(ticker, "5d", ttlMs)
        return h?.closes?.lastOrNull()
    }
}
