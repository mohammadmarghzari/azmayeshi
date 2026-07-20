package ir.marghzari.portfolio360.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.AddCircleOutline
import androidx.compose.material.icons.rounded.CheckCircle
import androidx.compose.material.icons.rounded.Star
import androidx.compose.material.icons.rounded.StarBorder
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
import ir.marghzari.portfolio360.ui.components.ScreenHeader
import ir.marghzari.portfolio360.ui.components.SectionHeader
import ir.marghzari.portfolio360.ui.motion.SkeletonCard
import ir.marghzari.portfolio360.ui.motion.StaggerIn
import ir.marghzari.portfolio360.ui.state.StateHost
import ir.marghzari.portfolio360.ui.state.UiState
import ir.marghzari.portfolio360.util.money
import ir.marghzari.portfolio360.util.pct
import kotlinx.coroutines.async
import kotlinx.coroutines.coroutineScope
import kotlinx.coroutines.launch

private data class MarketEntry(val symbol: String, val labelFa: String, val category: String)

/** Curated watch board; every symbol resolves on Yahoo Finance's daily-close endpoint. */
private val MARKET_ENTRIES = listOf(
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

private data class MarketQuote(
    val entry: MarketEntry,
    val closes: List<Double>,
) {
    val last: Double get() = closes.last()
    val dailyPct: Double
        get() {
            val prev = closes.getOrNull(closes.size - 2) ?: return 0.0
            return if (prev != 0.0) (last / prev - 1) * 100 else 0.0
        }
}

/**
 * Markets tab: a curated cross-market board (crypto, US equities, commodities, FX) with latest
 * price, daily change and a 30-day sparkline per symbol; one tap adds/removes the symbol from the
 * portfolio's ticker list so the discovery → allocation loop stays inside the app.
 */
@Composable
fun MarketsScreen(appState: AppState) {
    val colors = LocalChartColors.current
    val scope = rememberCoroutineScope()
    var state by remember { mutableStateOf<UiState<List<MarketQuote>>>(UiState.Loading) }

    suspend fun load() {
        state = UiState.Loading
        val quotes = coroutineScope {
            MARKET_ENTRIES.map { entry ->
                async {
                    appState.yahoo.fetchHistory(entry.symbol, "6mo")
                        ?.takeIf { it.closes.size >= 2 }
                        ?.let { MarketQuote(entry, it.closes) }
                }
            }.map { it.await() }
        }.filterNotNull()
        state = if (quotes.isEmpty()) {
            UiState.Error("هیچ بازاری در دسترس نبود — اتصال اینترنت (یا VPN) را بررسی کنید.")
        } else {
            UiState.Success(quotes)
        }
    }

    LaunchedEffect(Unit) { load() }

    LazyColumn(verticalArrangement = Arrangement.spacedBy(16.dp), modifier = Modifier.fillMaxWidth()) {
        item {
            ScreenHeader(
                "بازارها",
                "نمای زنده بازارهای جهانی — با یک ضربه هر نماد را به پرتفوی اضافه کنید.",
            )
        }
        item {
            StateHost(
                state = state,
                onRetry = { scope.launch { load() } },
                skeleton = { SkeletonCard(lines = 6) },
            ) { quotes ->
                Column(verticalArrangement = Arrangement.spacedBy(12.dp)) {
                    quotes.groupBy { it.entry.category }.forEach { (category, rows) ->
                        SectionHeader(category)
                        Card(modifier = Modifier.fillMaxWidth()) {
                            rows.forEachIndexed { idx, q ->
                                StaggerIn(index = idx) { MarketRow(appState, q) }
                            }
                        }
                    }
                    Text(
                        "قیمت پایانی روزانه از Yahoo Finance؛ برای داده کامل و بهینه‌سازی، نماد را اضافه و در تب «تخصیص پرتفوی» محاسبه کنید.",
                        style = MaterialTheme.typography.labelSmall, color = colors.muted,
                    )
                }
            }
        }
    }
}

@Composable
private fun MarketRow(appState: AppState, q: MarketQuote) {
    val colors = LocalChartColors.current
    val positive = q.dailyPct >= 0
    val inPortfolio = q.entry.symbol in appState.selectedTickers
    Row(
        modifier = Modifier.fillMaxWidth().padding(vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        CoinAvatar(q.entry.symbol)
        Column(modifier = Modifier.weight(1f)) {
            Text(q.entry.labelFa, style = MaterialTheme.typography.bodyMedium, color = colors.textPrimary)
            Text(q.entry.symbol, style = MaterialTheme.typography.labelSmall, color = colors.muted)
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
        val isFavorite = q.entry.symbol in appState.favoriteTickers
        IconButton(
            onClick = {
                appState.favoriteTickers = if (isFavorite) {
                    appState.favoriteTickers - q.entry.symbol
                } else {
                    appState.favoriteTickers + q.entry.symbol
                }
            },
        ) {
            Icon(
                if (isFavorite) Icons.Rounded.Star else Icons.Rounded.StarBorder,
                contentDescription = if (isFavorite) "حذف از واچ‌لیست" else "افزودن به واچ‌لیست",
                tint = if (isFavorite) colors.gold else colors.muted,
            )
        }
        IconButton(
            onClick = {
                appState.selectedTickers = if (inPortfolio) {
                    appState.selectedTickers - q.entry.symbol
                } else {
                    appState.selectedTickers + q.entry.symbol
                }
            },
        ) {
            Icon(
                if (inPortfolio) Icons.Rounded.CheckCircle else Icons.Rounded.AddCircleOutline,
                contentDescription = if (inPortfolio) "حذف از پرتفوی" else "افزودن به پرتفوی",
                tint = if (inPortfolio) colors.green else colors.muted,
            )
        }
    }
}
