package ir.marghzari.portfolio360.core.network

import io.ktor.client.call.body
import io.ktor.client.request.get
import io.ktor.client.request.header
import ir.marghzari.portfolio360.core.model.ImeCandle
import ir.marghzari.portfolio360.core.model.ImeQuote
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive

/**
 * Iran Mercantile Exchange (بورس کالای ایران) commodity-deposit-certificate API client,
 * a direct port of the `ime_fetch` / `ime_fetch_history` / `ime_parse_rows` helpers in app.py.
 *
 * The API key below is the one already committed in the source Python app (`app.py`, IME Live tab) —
 * it is not a secret introduced by this port.
 */
class ImeClient(
    private val apiKey: String = "B3dExgPIwrJFhn47YH9hxDrRnmAq6BQU",
    private val baseUrl: String = "https://api.ime.co.ir/api/v1",
    private val boardCache: TtlCache<ImeApiResult> = TtlCache(),
    private val historyCache: TtlCache<ImeApiResult> = TtlCache(),
) {
    sealed class ImeApiResult {
        data class Ok(val rows: List<JsonObject>) : ImeApiResult()
        data class Err(val message: String) : ImeApiResult()
    }

    private suspend fun rawGet(endpoint: String, params: Map<String, String> = emptyMap()): ImeApiResult {
        return try {
            val resp: JsonElement = HttpClientProvider.client.get("$baseUrl$endpoint") {
                url {
                    parameters.append("api_key", apiKey)
                    params.forEach { (k, v) -> parameters.append(k, v) }
                }
                header("Accept", "application/json")
                header("X-Api-Key", apiKey)
            }.body()
            val obj = resp as? JsonObject
            val rowsElement: JsonElement? = obj?.get("data") ?: obj?.get("result") ?: (resp as? JsonArray)
            val rows: List<JsonObject> = when (rowsElement) {
                is JsonArray -> rowsElement.jsonArray.mapNotNull { it as? JsonObject }
                is JsonObject -> listOf(rowsElement)
                else -> emptyList()
            }
            ImeApiResult.Ok(rows)
        } catch (e: Exception) {
            ImeApiResult.Err(e.message ?: "unknown error")
        }
    }

    suspend fun fetchLastChanges(): ImeApiResult =
        boardCache.getOrPut("last-changes", 60_000) { rawGet("/commodity-deposit-certificate/last-changes") }

    suspend fun fetchHistory(contractCode: String, fromDateShamsi: String, toDateShamsi: String): ImeApiResult =
        historyCache.getOrPut("history|$contractCode|$fromDateShamsi|$toDateShamsi", 300_000) {
            rawGet("/commodity-deposit-certificate/$contractCode/history", mapOf("from" to fromDateShamsi, "to" to toDateShamsi))
        }

    suspend fun testEndpoint(endpoint: String, params: Map<String, String>): ImeApiResult = rawGet(endpoint, params)

    companion object {
        private fun JsonObject.str(key: String, fallbackKey: String? = null): String {
            val v = this[key] ?: fallbackKey?.let { this[it] }
            return (v as? JsonPrimitive)?.contentOrNull() ?: ""
        }
        private fun JsonObject.num(key: String, fallbackKey: String? = null): Double {
            val v = this[key] ?: fallbackKey?.let { this[it] }
            val p = v as? JsonPrimitive ?: return 0.0
            return p.contentOrNull()?.toDoubleOrNull() ?: 0.0
        }
        private fun JsonPrimitive.contentOrNull(): String? = try { content } catch (e: Exception) { null }

        fun parseQuote(o: JsonObject): ImeQuote = ImeQuote(
            commodity = o.str("commodity"),
            contractCode = o.str("contract_code"),
            contractDescription = o.str("contract_description"),
            contractSize = o.num("contract_size"),
            contractSizeUnit = o.str("contract_size_unit"),
            contractCurrency = o.str("contract_currency"),
            py = o.num("py"),
            pf = o.num("pf"), pfChange = o.num("pfc"), pfChangePct = o.num("pfp"),
            pMax = o.num("pmax"), pMaxChange = o.num("pmaxc"), pMaxChangePct = o.num("pmaxp"),
            pMin = o.num("pmin"), pMinChange = o.num("pminc"), pMinChangePct = o.num("pminp"),
            pl = o.num("pl"), plChange = o.num("plc"), plChangePct = o.num("plp"),
            plTime = o.str("pl_time"), pfTime = o.str("pf_time"),
            tradeCount = o.num("tno").toLong(), volume = o.num("tvol"), tradeValue = o.num("tval"), tradeValueUnit = o.str("tval_unit"),
            dateOrder = o.str("date_order"), timeOrder = o.str("time_order"),
            bidPrices = listOf(o.num("pd1"), o.num("pd2"), o.num("pd3")),
            bidQtys = listOf(o.num("qd1"), o.num("qd2"), o.num("qd3")),
            askPrices = listOf(o.num("po1"), o.num("po2"), o.num("po3")),
            askQtys = listOf(o.num("qo1"), o.num("qo2"), o.num("qo3")),
            dateUpdate = o.str("date_update"), timeUpdate = o.str("time_update"),
        )

        fun parseCandle(o: JsonObject): ImeCandle = ImeCandle(
            dateLabel = o.str("date", "date_y"),
            open = o.num("open", "pf"),
            high = o.num("high", "pmax"),
            low = o.num("low", "pmin"),
            close = o.num("close", "pl"),
            volume = o.num("volume", "tvol"),
        )
    }
}
