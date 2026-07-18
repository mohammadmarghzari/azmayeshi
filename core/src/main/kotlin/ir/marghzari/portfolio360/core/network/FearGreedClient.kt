package ir.marghzari.portfolio360.core.network

import io.ktor.client.call.body
import io.ktor.client.request.get
import io.ktor.client.request.header
import kotlinx.serialization.Serializable

@Serializable
private data class FearGreedResponse(val fear_and_greed: FearGreedCore)

@Serializable
private data class FearGreedCore(
    val score: Double,
    val rating: String? = null,
    val previous_close: Double? = null,
    val previous_1_week: Double? = null,
    val previous_1_month: Double? = null,
    val previous_1_year: Double? = null,
)

data class FearGreedData(
    val score: Double, val rating: String?,
    val previousClose: Double?, val previousWeek: Double?, val previousMonth: Double?, val previousYear: Double?,
)

/** CNN Fear & Greed index, matching `fetch_fear_greed` in app.py (30-minute cache). */
class FearGreedClient(private val cache: TtlCache<FearGreedData?> = TtlCache()) {
    suspend fun fetch(): FearGreedData? = cache.getOrPut("fear_greed", 1_800_000) {
        try {
            val resp: FearGreedResponse = HttpClientProvider.client.get("https://production.dataviz.cnn.io/index/fearandgreed/graphdata") {
                header("Accept", "application/json")
            }.body()
            FearGreedData(
                resp.fear_and_greed.score, resp.fear_and_greed.rating,
                resp.fear_and_greed.previous_close, resp.fear_and_greed.previous_1_week,
                resp.fear_and_greed.previous_1_month, resp.fear_and_greed.previous_1_year,
            )
        } catch (e: Exception) {
            null
        }
    }
}
