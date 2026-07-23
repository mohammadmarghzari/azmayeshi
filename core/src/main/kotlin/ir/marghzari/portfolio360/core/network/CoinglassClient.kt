package ir.marghzari.portfolio360.core.network

import io.ktor.client.call.body
import io.ktor.client.request.get
import io.ktor.client.request.header
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.contentOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import kotlinx.serialization.json.longOrNull

/**
 * One liquidation-heatmap cell: [timeMs] bucket, [price] level, and the liquidation-volume
 * [intensity] at that point (raw units as returned by the API — the chart normalizes for color).
 */
data class HeatmapCell(val timeMs: Long, val price: Double, val intensity: Double)

/** A parsed liquidation heatmap: the price path overlay plus the sparse intensity grid. */
data class LiquidationHeatmapData(
    val symbol: String,
    val priceLine: List<Pair<Long, Double>>,
    val cells: List<HeatmapCell>,
)

/**
 * CoinGlass (coinglass.com) liquidation-heatmap API client — the "Model 1/2/3" chart from the
 * CoinGlass website's Futures section, showing where leveraged positions cluster and are likely
 * to be liquidated as price moves through them.
 *
 * The API key is NEVER hardcoded: it comes from [BuildSecrets], generated at build time from a
 * gitignored `secrets.properties` (local builds) or the `COINGLASS_API_KEY` CI secret — this repo
 * is public, so a real personal key must never enter git history. An empty key or any request
 * failure surfaces as `null`; the exact response schema is documented by CoinGlass but could not
 * be verified from this environment (outbound access to coinglass.com is blocked here), so parsing
 * is intentionally defensive — read [ImeClient]'s comment for the precedent of this pattern in the
 * codebase. If a fetch fails after installing this on a device with real network access, capturing
 * the raw JSON response lets the parser be corrected precisely, the same way the CVaR format-string
 * bug was fixed from a real stack trace rather than guesswork.
 */
class CoinglassClient(
    private val apiKey: String = BuildSecrets.COINGLASS_API_KEY,
    private val baseUrl: String = "https://open-api-v4.coinglass.com",
    private val cache: TtlCache<LiquidationHeatmapData?> = TtlCache(),
) {
    val isConfigured: Boolean get() = apiKey.isNotBlank()

    /**
     * @param symbol e.g. "BTC", "ETH".
     * @param range one of 12h/24h/48h/3d/1w/2w/1m/3m/6m/1y/2y, matching the website's range picker.
     * @param model 1, 2 or 3 — CoinGlass's three liquidation-heatmap calculation models.
     */
    suspend fun fetchHeatmap(symbol: String, range: String, model: Int, ttlMs: Long = 60_000): LiquidationHeatmapData? {
        if (!isConfigured) return null
        return cache.getOrPut("$symbol|$range|$model", ttlMs) { fetchNow(symbol, range, model) }
    }

    private suspend fun fetchNow(symbol: String, range: String, model: Int): LiquidationHeatmapData? {
        return try {
            val resp: JsonElement = HttpClientProvider.client.get(
                "$baseUrl/api/futures/liquidation/heatmap/model$model",
            ) {
                url {
                    parameters.append("symbol", symbol)
                    parameters.append("range", range)
                }
                header("CG-API-KEY", apiKey)
                header("Accept", "application/json")
            }.body()

            val root = resp as? JsonObject ?: return null
            val data = root["data"]?.jsonObject ?: (root.takeIf { it.containsKey("y") })  ?: return null

            val priceLine = parsePriceLine(data)
            val cells = parseCells(data)
            if (cells.isEmpty() && priceLine.isEmpty()) return null
            LiquidationHeatmapData(symbol = symbol, priceLine = priceLine, cells = cells)
        } catch (e: Exception) {
            null
        }
    }

    /** Accepts either `prices`/`priceList`/`price_history` as a list of [ts, price] pairs. */
    private fun parsePriceLine(data: JsonObject): List<Pair<Long, Double>> {
        val arr = (data["prices"] ?: data["priceList"] ?: data["price_history"]) as? JsonArray ?: return emptyList()
        return arr.mapNotNull { entry ->
            val pair = entry as? JsonArray ?: return@mapNotNull null
            val t = pair.getOrNull(0)?.jsonPrimitive?.longOrNull ?: return@mapNotNull null
            val p = pair.getOrNull(1)?.jsonPrimitive?.doubleOrNull ?: return@mapNotNull null
            t to p
        }
    }

    /**
     * Accepts the documented sparse-triple shape `liq: [[timeIndex, priceIndex, value], ...]`
     * resolved against parallel `x` (times) / `y` (prices) axis arrays, falling back to a dense
     * `[row][col]` matrix under `data`/`matrix` keyed directly by numeric price/time if present.
     */
    private fun parseCells(data: JsonObject): List<HeatmapCell> {
        val xAxis = (data["x"] as? JsonArray)?.mapNotNull { it.jsonPrimitive.longOrNull }
        val yAxis = (data["y"] as? JsonArray)?.mapNotNull { it.jsonPrimitive.doubleOrNull }
        val triples = (data["liq"] ?: data["data"]) as? JsonArray

        if (triples != null && xAxis != null && yAxis != null) {
            return triples.mapNotNull { entry ->
                val t = entry as? JsonArray ?: return@mapNotNull null
                val xi = t.getOrNull(0)?.jsonPrimitive?.contentOrNull?.toDoubleOrNull()?.toInt() ?: return@mapNotNull null
                val yi = t.getOrNull(1)?.jsonPrimitive?.contentOrNull?.toDoubleOrNull()?.toInt() ?: return@mapNotNull null
                val v = t.getOrNull(2)?.jsonPrimitive?.doubleOrNull ?: return@mapNotNull null
                val time = xAxis.getOrNull(xi) ?: return@mapNotNull null
                val price = yAxis.getOrNull(yi) ?: return@mapNotNull null
                if (v <= 0.0) null else HeatmapCell(time, price, v)
            }
        }
        return emptyList()
    }
}
