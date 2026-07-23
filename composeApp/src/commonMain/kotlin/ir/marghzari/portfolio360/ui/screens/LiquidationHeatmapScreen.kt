package ir.marghzari.portfolio360.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import ir.marghzari.portfolio360.charts.LiquidationHeatmapChart
import ir.marghzari.portfolio360.core.network.LiquidationHeatmapData
import ir.marghzari.portfolio360.state.AppState
import ir.marghzari.portfolio360.theme.LocalChartColors
import ir.marghzari.portfolio360.ui.components.AppChip
import ir.marghzari.portfolio360.ui.components.Card
import ir.marghzari.portfolio360.ui.components.EmptyState
import ir.marghzari.portfolio360.ui.components.ScreenHeader
import ir.marghzari.portfolio360.ui.components.SimpleDropdown
import ir.marghzari.portfolio360.ui.motion.SkeletonCard
import ir.marghzari.portfolio360.ui.state.StateHost
import ir.marghzari.portfolio360.ui.state.UiState
import kotlinx.coroutines.launch

private val HEATMAP_SYMBOLS = listOf("BTC", "ETH", "SOL", "BNB")
private data class RangeOption(val label: String, val code: String)
private val HEATMAP_RANGES = listOf(
    RangeOption("۱۲ ساعت", "12h"), RangeOption("۲۴ ساعت", "24h"), RangeOption("۴۸ ساعت", "48h"),
    RangeOption("۳ روز", "3d"), RangeOption("۱ هفته", "1w"), RangeOption("۲ هفته", "2w"),
    RangeOption("۱ ماه", "1m"), RangeOption("۳ ماه", "3m"), RangeOption("۶ ماه", "6m"),
    RangeOption("۱ سال", "1y"), RangeOption("۲ سال", "2y"),
)

/**
 * Liquidation heatmap tab — where leveraged futures positions cluster and are likely to be
 * liquidated as price moves through them, mirroring CoinGlass's own "Model 1/2/3" chart. Requires
 * a CoinGlass API key (see `CoinglassClient` for why it is never hardcoded in this public repo);
 * an unconfigured key renders a clear guidance state instead of a silent failure.
 */
@Composable
fun LiquidationHeatmapScreen(appState: AppState) {
    val colors = LocalChartColors.current
    val scope = rememberCoroutineScope()

    var symbol by remember { mutableStateOf(HEATMAP_SYMBOLS.first()) }
    var range by remember { mutableStateOf(HEATMAP_RANGES[2]) }
    var model by remember { mutableStateOf(3) }
    var state by remember { mutableStateOf<UiState<LiquidationHeatmapData>>(UiState.Loading) }

    suspend fun load() {
        if (!appState.coinglass.isConfigured) {
            state = UiState.Empty
            return
        }
        state = UiState.Loading
        val data = appState.coinglass.fetchHeatmap(symbol, range.code, model)
        state = if (data == null) {
            UiState.Error("دریافت نقشه نقدشوندگی ناموفق بود — کلید API، اتصال اینترنت یا سطح اشتراک CoinGlass را بررسی کنید.")
        } else {
            UiState.Success(data)
        }
    }

    LaunchedEffect(symbol, range, model) { load() }

    LazyColumn(verticalArrangement = Arrangement.spacedBy(16.dp), modifier = Modifier.fillMaxWidth()) {
        item {
            ScreenHeader(
                "نقشه حرارتی نقدشوندگی",
                "تراکم موقعیت‌های اهرمی که با رسیدن قیمت به آن‌ها لیکویید می‌شوند — داده از CoinGlass.",
            )
        }
        item {
            Card(modifier = Modifier.fillMaxWidth()) {
                Row(horizontalArrangement = Arrangement.spacedBy(8.dp)) {
                    listOf(1, 2, 3).forEach { m ->
                        AppChip("مدل $m", selected = model == m, onClick = { model = m })
                    }
                }
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.padding(top = 10.dp)) {
                    SimpleDropdown(
                        "نماد", symbol, HEATMAP_SYMBOLS, { it }, { symbol = it },
                        modifier = Modifier.weight(1f),
                    )
                    SimpleDropdown(
                        "بازه زمانی", range, HEATMAP_RANGES, { it.label }, { range = it },
                        modifier = Modifier.weight(1f),
                    )
                }
            }
        }
        item {
            StateHost(
                state = state,
                onRetry = { scope.launch { load() } },
                emptyTitle = "کلید CoinGlass تنظیم نشده",
                emptyHint = "این نمودار به کلید API معتبر CoinGlass نیاز دارد. برای نسخه‌های دستی، کلید را در secrets.properties قرار دهید؛ برای APK ساخته‌شده در گیت‌هاب، کلید باید به‌صورت GitHub Actions secret اضافه شود.",
                skeleton = { SkeletonCard(lines = 6) },
            ) { data ->
                Card(modifier = Modifier.fillMaxWidth()) {
                    LiquidationHeatmapChart(
                        cells = data.cells,
                        priceLine = data.priceLine,
                        title = "${data.symbol} — نقدشوندگی (مدل $model)",
                    )
                    Text(
                        "رنگ روشن‌تر = تراکم بیشتر سفارش‌های اهرمی در آن سطح قیمتی؛ خط سفید مسیر قیمت واقعی است.",
                        style = MaterialTheme.typography.labelSmall, color = colors.muted,
                        modifier = Modifier.padding(top = 8.dp),
                    )
                }
            }
        }
    }
}
