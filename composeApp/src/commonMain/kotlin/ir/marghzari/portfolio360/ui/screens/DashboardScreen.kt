package ir.marghzari.portfolio360.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
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
import ir.marghzari.portfolio360.data.MarketEntry
import ir.marghzari.portfolio360.data.MarketSnapshot
import ir.marghzari.portfolio360.data.fetchMarketSnapshots
import ir.marghzari.portfolio360.nav.Destination
import ir.marghzari.portfolio360.state.AppState
import ir.marghzari.portfolio360.theme.LocalChartColors
import ir.marghzari.portfolio360.ui.components.AppButton
import ir.marghzari.portfolio360.ui.components.AssetRow
import ir.marghzari.portfolio360.ui.components.ButtonStyle
import ir.marghzari.portfolio360.ui.components.Card
import ir.marghzari.portfolio360.ui.components.CoinAvatar
import ir.marghzari.portfolio360.ui.components.EmptyState
import ir.marghzari.portfolio360.ui.components.HeroMetric
import ir.marghzari.portfolio360.ui.components.MetricTile
import ir.marghzari.portfolio360.ui.components.ScreenHeader
import ir.marghzari.portfolio360.ui.components.SectionHeader
import ir.marghzari.portfolio360.ui.motion.SkeletonCard
import ir.marghzari.portfolio360.ui.motion.StaggerIn
import ir.marghzari.portfolio360.ui.state.StateHost
import ir.marghzari.portfolio360.ui.state.UiState
import ir.marghzari.portfolio360.util.money
import ir.marghzari.portfolio360.util.pct
import kotlinx.coroutines.launch

/**
 * Home dashboard — the app's landing tab: one screen answering "how is my money doing?" with
 * the recorded capital, portfolio health, a watchlist glance, today's market movers and the
 * latest journal activity, each section deep-linking into its full tab via [onNavigate].
 */
@Composable
fun DashboardScreen(appState: AppState, onNavigate: (Destination) -> Unit) {
    val colors = LocalChartColors.current
    val scope = rememberCoroutineScope()

    var marketState by remember { mutableStateOf<UiState<List<MarketSnapshot>>>(UiState.Loading) }
    val favorites = appState.favoriteTickers.sorted().take(4)
    var watchState by remember { mutableStateOf<UiState<List<MarketSnapshot>>>(UiState.Empty) }

    suspend fun loadMarkets() {
        marketState = UiState.Loading
        val snapshots = fetchMarketSnapshots(appState.yahoo)
        marketState = if (snapshots.isEmpty()) {
            UiState.Error("بازارها در دسترس نیستند — اتصال اینترنت (یا VPN) را بررسی کنید.")
        } else {
            UiState.Success(snapshots)
        }
    }

    suspend fun loadWatchlist() {
        if (favorites.isEmpty()) {
            watchState = UiState.Empty
            return
        }
        watchState = UiState.Loading
        val snapshots = fetchMarketSnapshots(
            appState.yahoo,
            favorites.map { MarketEntry(it, it, "واچ‌لیست") },
        )
        watchState = if (snapshots.isEmpty()) UiState.Empty else UiState.Success(snapshots)
    }

    LaunchedEffect(Unit) { loadMarkets() }
    LaunchedEffect(favorites) { loadWatchlist() }

    val txs = appState.transactions
    val invested = txs.sumOf { if (it.isBuy) it.quantity * it.price else -it.quantity * it.price }
    val metrics = appState.metrics
    val prices = appState.prices
    val weights = appState.weights

    LazyColumn(verticalArrangement = Arrangement.spacedBy(16.dp), modifier = Modifier.fillMaxWidth()) {
        item {
            ScreenHeader("داشبورد", "نمای کلی سرمایه، بازار و فعالیت‌های شما در یک نگاه")
        }

        // Hero: recorded capital + quick actions.
        item {
            Card(modifier = Modifier.fillMaxWidth(), highlighted = true) {
                if (txs.isNotEmpty()) {
                    HeroMetric(
                        label = "سرمایه ثبت‌شده در دفتر تراکنش‌ها",
                        value = invested.money(),
                        delta = "${txs.size} تراکنش",
                    )
                } else {
                    HeroMetric(label = "سرمایه ثبت‌شده", value = "—")
                    Text(
                        "هنوز تراکنشی ثبت نشده — با «ثبت تراکنش» شروع کنید تا این عدد زنده شود.",
                        style = MaterialTheme.typography.bodySmall, color = colors.muted,
                        modifier = Modifier.padding(top = 6.dp),
                    )
                }
                Row(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    modifier = Modifier.fillMaxWidth().padding(top = 14.dp),
                ) {
                    AppButton("📥 دریافت داده", onClick = { onNavigate(Destination.ALLOCATION) }, style = ButtonStyle.Secondary)
                    AppButton("🧾 ثبت تراکنش", onClick = { onNavigate(Destination.TRANSACTIONS) }, style = ButtonStyle.Secondary)
                    AppButton("🛰 بازارها", onClick = { onNavigate(Destination.MARKETS) }, style = ButtonStyle.Secondary)
                }
            }
        }

        // Portfolio health snapshot.
        item {
            SectionHeader("وضعیت پرتفوی")
            if (metrics != null) {
                LazyRow(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    items(
                        listOf(
                            Triple("بازده سالانه", (metrics.annualReturn * 100).pct(2, signed = true), null),
                            Triple("نوسان سالانه", (metrics.annualVolatility * 100).pct(2), null),
                            Triple("نسبت شارپ", "%.2f".format(metrics.sharpeRatio), null),
                            Triple("حداکثر افت", (metrics.maxDrawdown * 100).pct(2), null),
                        ),
                    ) { (label, value, _) -> MetricTile(label, value) }
                }
            } else {
                EmptyState(
                    title = "هنوز پرتفویی محاسبه نشده",
                    hint = "داده را دریافت و پرتفوی را بهینه کنید تا سلامت آن اینجا خلاصه شود.",
                    actionText = "برو به تخصیص پرتفوی",
                    onAction = { onNavigate(Destination.ALLOCATION) },
                )
            }
        }

        // Top positions from the computed portfolio.
        if (weights != null && prices != null) {
            item {
                SectionHeader("موقعیت‌های اصلی")
                Card(modifier = Modifier.fillMaxWidth()) {
                    prices.tickers.indices.sortedByDescending { weights[it] }.take(3)
                        .forEachIndexed { idx, i ->
                            StaggerIn(index = idx) {
                                AssetRow(
                                    symbol = prices.tickers[i],
                                    title = prices.tickers[i],
                                    caption = "وزن ${(weights[i] * 100).pct(1)}",
                                    value = prices.column(prices.tickers[i]).lastOrNull()?.money() ?: "—",
                                )
                            }
                        }
                    AppButton("مشاهده کامل تخصیص", onClick = { onNavigate(Destination.ALLOCATION) }, style = ButtonStyle.Ghost)
                }
            }
        }

        // Watchlist glance.
        item {
            SectionHeader("واچ‌لیست")
            StateHost(
                state = watchState,
                onRetry = { scope.launch { loadWatchlist() } },
                emptyTitle = "واچ‌لیست خالی است",
                emptyHint = "در تب بازارها ستاره نمادها را بزنید تا اینجا دنبال‌شان کنید.",
                emptyActionText = "برو به بازارها",
                onEmptyAction = { onNavigate(Destination.MARKETS) },
                skeleton = { SkeletonCard(lines = 3) },
            ) { rows ->
                Card(modifier = Modifier.fillMaxWidth()) {
                    rows.forEachIndexed { idx, q ->
                        StaggerIn(index = idx) { DashboardMarketRow(q) }
                    }
                    AppButton("واچ‌لیست کامل", onClick = { onNavigate(Destination.WATCHLIST) }, style = ButtonStyle.Ghost)
                }
            }
        }

        // Today's movers across the curated board.
        item {
            SectionHeader("تحرکات امروز بازار")
            StateHost(
                state = marketState,
                onRetry = { scope.launch { loadMarkets() } },
                skeleton = { SkeletonCard(lines = 4) },
            ) { snapshots ->
                val sorted = snapshots.sortedByDescending { it.dailyPct }
                Card(modifier = Modifier.fillMaxWidth()) {
                    Text("برترین‌ها", style = MaterialTheme.typography.labelMedium, color = colors.green)
                    sorted.take(3).forEachIndexed { idx, q ->
                        StaggerIn(index = idx) { DashboardMarketRow(q) }
                    }
                    Text(
                        "ضعیف‌ترین‌ها", style = MaterialTheme.typography.labelMedium, color = colors.red,
                        modifier = Modifier.padding(top = 10.dp),
                    )
                    sorted.takeLast(3).reversed().forEachIndexed { idx, q ->
                        StaggerIn(index = idx) { DashboardMarketRow(q) }
                    }
                    AppButton("همه بازارها", onClick = { onNavigate(Destination.MARKETS) }, style = ButtonStyle.Ghost)
                }
            }
        }

        // Recent journal activity.
        if (txs.isNotEmpty()) {
            item {
                SectionHeader("فعالیت اخیر")
                Card(modifier = Modifier.fillMaxWidth()) {
                    txs.sortedByDescending { it.id }.take(3).forEachIndexed { idx, t ->
                        StaggerIn(index = idx) {
                            AssetRow(
                                symbol = t.symbol,
                                title = t.symbol,
                                caption = listOfNotNull(
                                    if (t.isBuy) "خرید" else "فروش",
                                    t.dateLabel.ifBlank { null },
                                ).joinToString(" · "),
                                value = (t.quantity * t.price).money(),
                                delta = if (t.isBuy) "خرید" else "فروش",
                                deltaPositive = t.isBuy,
                            )
                        }
                    }
                    AppButton("دفتر کامل تراکنش‌ها", onClick = { onNavigate(Destination.TRANSACTIONS) }, style = ButtonStyle.Ghost)
                }
            }
        }
    }
}

@Composable
private fun DashboardMarketRow(q: MarketSnapshot) {
    val colors = LocalChartColors.current
    val positive = q.dailyPct >= 0
    Row(
        modifier = Modifier.fillMaxWidth().padding(vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        CoinAvatar(q.entry.symbol, size = 32.dp)
        Column(modifier = Modifier.weight(1f)) {
            Text(q.entry.labelFa, style = MaterialTheme.typography.bodyMedium, color = colors.textPrimary)
        }
        Sparkline(
            values = q.closes.takeLast(30),
            color = if (positive) colors.green else colors.red,
            width = 56.dp,
            height = 22.dp,
        )
        Column(horizontalAlignment = Alignment.End) {
            Text(q.last.money(), style = MaterialTheme.typography.labelLarge, color = colors.textPrimary)
            Text(
                q.dailyPct.pct(2, signed = true),
                style = MaterialTheme.typography.labelSmall,
                color = if (positive) colors.green else colors.red,
            )
        }
    }
}
