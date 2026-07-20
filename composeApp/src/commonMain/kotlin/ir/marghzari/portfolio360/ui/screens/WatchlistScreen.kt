package ir.marghzari.portfolio360.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Star
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import ir.marghzari.portfolio360.charts.Sparkline
import ir.marghzari.portfolio360.state.AppState
import ir.marghzari.portfolio360.theme.LocalChartColors
import ir.marghzari.portfolio360.ui.components.CoinAvatar
import ir.marghzari.portfolio360.ui.components.Card
import ir.marghzari.portfolio360.ui.components.EmptyState
import ir.marghzari.portfolio360.ui.components.ScreenHeader
import ir.marghzari.portfolio360.ui.motion.SkeletonCard
import ir.marghzari.portfolio360.ui.motion.StaggerIn
import ir.marghzari.portfolio360.ui.state.StateHost
import ir.marghzari.portfolio360.ui.state.UiState
import ir.marghzari.portfolio360.util.money
import ir.marghzari.portfolio360.util.pct
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch

private data class WatchQuote(val symbol: String, val closes: List<Double>) {
    val last: Double get() = closes.last()
    val dailyPct: Double
        get() {
            val prev = closes.getOrNull(closes.size - 2) ?: return 0.0
            return if (prev != 0.0) (last / prev - 1) * 100 else 0.0
        }
}

/**
 * Watchlist tab: every starred symbol (starred on the price-chart hero or a Markets row) as a
 * live card with latest price, daily change and a 30-day sparkline. Reloads whenever the set of
 * favorites changes; un-starring here removes the row in place.
 */
@Composable
fun WatchlistScreen(appState: AppState) {
    val colors = LocalChartColors.current
    val scope = rememberCoroutineScope()
    val favorites = appState.favoriteTickers.sorted()
    var state by remember { mutableStateOf<UiState<List<WatchQuote>>>(UiState.Loading) }

    suspend fun load() {
        if (favorites.isEmpty()) {
            state = UiState.Empty
            return
        }
        state = UiState.Loading
        val quotes = coroutineScope {
            favorites.map { symbol ->
                async {
                    appState.yahoo.fetchHistory(symbol, "6mo")
                        ?.takeIf { it.closes.size >= 2 }
                        ?.let { WatchQuote(symbol, it.closes) }
                }
            }.map { it.await() }
        }.filterNotNull()
        state = if (quotes.isEmpty()) {
            UiState.Error("داده هیچ‌کدام از نمادهای ستاره‌دار دریافت نشد — اتصال اینترنت (یا VPN) را بررسی کنید.")
        } else {
            UiState.Success(quotes)
        }
    }

    LaunchedEffect(favorites) { load() }

    LazyColumn(verticalArrangement = Arrangement.spacedBy(16.dp), modifier = Modifier.fillMaxWidth()) {
        item {
            ScreenHeader(
                "واچ‌لیست",
                "نمادهای ستاره‌دار شما — از تب «بازارها» یا صفحه «نمودار قیمت» ستاره کنید.",
            )
        }
        item {
            StateHost(
                state = state,
                onRetry = { scope.launch { load() } },
                emptyTitle = "واچ‌لیست خالی است",
                emptyHint = "در تب «بازارها» یا بالای صفحه «نمودار قیمت»، ستاره هر نماد را بزنید تا اینجا با قیمت زنده دنبالش کنید.",
                skeleton = { SkeletonCard(lines = 4) },
            ) { quotes ->
                Card(modifier = Modifier.fillMaxWidth()) {
                    quotes.forEachIndexed { idx, q ->
                        StaggerIn(index = idx) { WatchRow(appState, q) }
                    }
                }
            }
        }
    }
}

@Composable
private fun WatchRow(appState: AppState, q: WatchQuote) {
    val colors = LocalChartColors.current
    val positive = q.dailyPct >= 0
    Row(
        modifier = Modifier.fillMaxWidth().padding(vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        CoinAvatar(q.symbol)
        Column(modifier = Modifier.weight(1f)) {
            Text(q.symbol, style = MaterialTheme.typography.bodyMedium, color = colors.textPrimary)
        }
        Sparkline(
            values = q.closes.takeLast(30),
            color = if (positive) colors.green else colors.red,
        )
        Column(horizontalAlignment = Alignment.End) {
            Text(q.last.money(), style = MaterialTheme.typography.labelLarge, color = colors.textPrimary)
            Text(
                q.dailyPct.pct(2, signed = true),
                style = MaterialTheme.typography.labelSmall,
                color = if (positive) colors.green else colors.red,
            )
        }
        IconButton(onClick = { appState.favoriteTickers = appState.favoriteTickers - q.symbol }) {
            Icon(Icons.Rounded.Star, contentDescription = "حذف از واچ‌لیست", tint = colors.gold)
        }
    }
}
