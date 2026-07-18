package ir.marghzari.portfolio360.core.network

import io.ktor.client.HttpClient
import io.ktor.client.engine.cio.CIO
import io.ktor.client.plugins.HttpTimeout
import io.ktor.client.plugins.contentnegotiation.ContentNegotiation
import io.ktor.client.plugins.defaultRequest
import io.ktor.client.request.header
import io.ktor.serialization.kotlinx.json.json
import kotlinx.serialization.json.Json

/** Single shared Ktor client for every remote data source (Yahoo Finance, CNN F&G, IME, RSS news). */
object HttpClientProvider {
    val json = Json { ignoreUnknownKeys = true; isLenient = true; coerceInputValues = true }

    val client: HttpClient by lazy {
        HttpClient(CIO) {
            expectSuccess = false
            install(ContentNegotiation) { json(json) }
            install(HttpTimeout) {
                requestTimeoutMillis = 12_000
                connectTimeoutMillis = 8_000
            }
            defaultRequest {
                header("User-Agent", "Portfolio360/1.0 (+desktop/android)")
            }
        }
    }
}
