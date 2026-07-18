package ir.marghzari.portfolio360.ui.screens

import androidx.compose.animation.animateColorAsState
import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.background
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.ArrowDownward
import androidx.compose.material.icons.filled.ArrowUpward
import androidx.compose.material.icons.filled.Refresh
import androidx.compose.material.icons.filled.Star
import androidx.compose.material.icons.filled.StarBorder
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Text
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.scale
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.clipRect
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import ir.marghzari.portfolio360.charts.HeatmapChart
import ir.marghzari.portfolio360.charts.LineChart
import ir.marghzari.portfolio360.charts.LineSeries
import ir.marghzari.portfolio360.core.math.Stats
import ir.marghzari.portfolio360.core.network.TickerHistory
import ir.marghzari.portfolio360.state.AppState
import ir.marghzari.portfolio360.theme.LocalChartColors
import ir.marghzari.portfolio360.theme.chartColor
import ir.marghzari.portfolio360.ui.components.Card
import ir.marghzari.portfolio360.ui.components.GlowButton
import ir.marghzari.portfolio360.ui.components.InfoBanner
import ir.marghzari.portfolio360.ui.components.MetricTile
import ir.marghzari.portfolio360.ui.components.SectionHeader
import ir.marghzari.portfolio360.ui.components.SimpleDropdown
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch
import kotlinx.datetime.LocalDate

private enum class RangeFilter(val label: String, val days: Int?) {
    D7("7D", 7), M1("1M", 30), M3("3M", 90), M6("6M", 182), Y1("1Y", 365), ALL("ALL", null),
}

private fun sliceRange(dates: List<LocalDate>, closes: List<Double>, range: RangeFilter): Pair<List<LocalDate>, List<Double>> {
    if (range.days == null || dates.size <= range.days) return dates to closes
    val from = (dates.size - range.days).coerceAtLeast(0)
    return dates.subList(from, dates.size) to closes.subList(from, closes.size)
}

/**
 * Single-asset detail view: pick a ticker from the portfolio and see its live-feeling price, an
 * animated interactive chart with range filters, and the real metrics our Yahoo daily-close data
 * source actually has (no market cap / intraday volume / brokerage holdings are available here,
 * so a "weight in portfolio" and "period return" stand in for holdings/P&L instead of faking them).
 * The previous multi-ticker overlay + correlation matrix is preserved below, collapsed by default.
 */
@OptIn(ExperimentalMaterial3Api::class)
@Composable
fun PriceChartScreen(appState: AppState) {
    val colors = LocalChartColors.current
    val prices = appState.prices ?: run { InfoBanner("ابتدا پرتفوی را محاسبه کنید."); return }
    val scope = rememberCoroutineScope()

    var selectedTicker by remember(prices) { mutableStateOf(prices.tickers.first()) }
    var range by remember { mutableStateOf(RangeFilter.ALL) }
    var refreshed by remember(selectedTicker) { mutableStateOf<TickerHistory?>(null) }
    var isRefreshing by remember { mutableStateOf(false) }
    var showCompareAll by remember { mutableStateOf(false) }

    val fullDates = refreshed?.dates ?: prices.dates
    val fullCloses = refreshed?.closes ?: prices.column(selectedTicker).toList()
    val (dates, closes) = remember(fullDates, fullCloses, range) { sliceRange(fullDates, fullCloses, range) }

    val current = closes.lastOrNull() ?: 0.0
    val previous = closes.getOrNull(closes.size - 2) ?: current
    val change = current - previous
    val changePct = if (previous != 0.0) change / previous * 100 else 0.0
    val periodHigh = closes.maxOrNull() ?: current
    val periodLow = closes.minOrNull() ?: current
    val periodReturnPct = if (closes.isNotEmpty() && closes.first() != 0.0) (closes.last() / closes.first() - 1) * 100 else 0.0
    val weightPct = appState.weights?.let { w -> prices.tickers.indexOf(selectedTicker).takeIf { it in w.indices }?.let { w[it] * 100 } }

    fun refresh() {
        scope.launch {
            isRefreshing = true
            refreshed = appState.yahoo.fetchHistory(selectedTicker, "max", ttlMs = 0)
            isRefreshing = false
        }
    }

    Box(modifier = Modifier.fillMaxSize()) {
        LazyColumn(verticalArrangement = Arrangement.spacedBy(16.dp), modifier = Modifier.fillMaxWidth()) {
            item {
                AssetHeroCard(
                    tickers = prices.tickers, selected = selectedTicker, onSelect = { selectedTicker = it },
                    price = current, change = change, changePct = changePct,
                    isFavorite = selectedTicker in appState.favoriteTickers,
                    onToggleFavorite = {
                        appState.favoriteTickers = if (selectedTicker in appState.favoriteTickers) {
                            appState.favoriteTickers - selectedTicker
                        } else {
                            appState.favoriteTickers + selectedTicker
                        }
                    },
                    isRefreshing = isRefreshing,
                    onRefresh = { refresh() },
                )
            }

            item { RangeFilterRow(selected = range, onSelect = { range = it }) }

            item {
                Card(modifier = Modifier.fillMaxWidth()) {
                    AssetLineChart(
                        dates = dates, closes = closes, positive = change >= 0,
                        modifier = Modifier.fillMaxWidth().height(260.dp),
                    )
                }
            }

            item {
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.fillMaxWidth()) {
                    MetricTile("بیشترین (بازه)", "$%,.2f".format(periodHigh), modifier = Modifier.weight(1f))
                    MetricTile("کمترین (بازه)", "$%,.2f".format(periodLow), modifier = Modifier.weight(1f))
                    MetricTile(
                        "بازده کل بازه", "%+.2f%%".format(periodReturnPct),
                        valueColor = if (periodReturnPct >= 0) colors.green else colors.red,
                        modifier = Modifier.weight(1f),
                    )
                }
            }

            if (weightPct != null) {
                item { MetricTile("وزن در پرتفوی", "%.1f%%".format(weightPct)) }
            }

            item {
                val inPortfolio = selectedTicker in appState.selectedTickers
                GlowButton(
                    text = if (inPortfolio) "➖ حذف از پرتفوی" else "➕ افزودن به پرتفوی",
                    onClick = {
                        appState.selectedTickers = if (inPortfolio) {
                            appState.selectedTickers - selectedTicker
                        } else {
                            appState.selectedTickers + selectedTicker
                        }
                    },
                    modifier = Modifier.fillMaxWidth(),
                )
            }

            item {
                SectionHeader(if (showCompareAll) "▼ مقایسه همه نمادها" else "▶ مقایسه همه نمادها")
                androidx.compose.material3.TextButton(onClick = { showCompareAll = !showCompareAll }) {
                    Text(if (showCompareAll) "بستن مقایسه" else "نمایش مقایسه همه نمادها و ماتریس همبستگی")
                }
            }

            if (showCompareAll) {
                item {
                    var normalized by remember { mutableStateOf(true) }
                    Row(verticalAlignment = Alignment.CenterVertically) {
                        RadioButton(selected = normalized, onClick = { normalized = true })
                        Text("نرمال‌شده (base=100)", modifier = Modifier.padding(end = 12.dp))
                        RadioButton(selected = !normalized, onClick = { normalized = false })
                        Text("قیمت خام")
                    }
                    val series = prices.tickers.mapIndexed { i, ticker ->
                        val col = prices.column(ticker)
                        val values = if (normalized) col.map { it / col[0] * 100 } else col.toList()
                        LineSeries(ticker, chartColor(i), values)
                    }
                    Card(modifier = Modifier.fillMaxWidth().padding(top = 10.dp)) {
                        LineChart(
                            series = series, title = "PRICE CHART",
                            yFormatter = { if (normalized) "%.0f".format(it) else "$%.2f".format(it) },
                            height = 380.dp,
                        )
                    }
                }
                if (prices.nAssets >= 2) {
                    item {
                        SectionHeader("Correlation Matrix")
                        val returns = prices.dailyReturns()
                        val n = prices.nAssets
                        val corr = Array(n) { i ->
                            DoubleArray(n) { j ->
                                Stats.correlation(DoubleArray(returns.size) { returns[it][i] }, DoubleArray(returns.size) { returns[it][j] })
                            }
                        }
                        Card(modifier = Modifier.fillMaxWidth()) {
                            HeatmapChart(
                                rowLabels = prices.tickers, colLabels = prices.tickers, values = corr,
                                title = "CORRELATION MATRIX", height = (60 + n * 34).dp,
                            )
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun AssetHeroCard(
    tickers: List<String>,
    selected: String,
    onSelect: (String) -> Unit,
    price: Double,
    change: Double,
    changePct: Double,
    isFavorite: Boolean,
    onToggleFavorite: () -> Unit,
    isRefreshing: Boolean,
    onRefresh: () -> Unit,
) {
    val colors = LocalChartColors.current
    val positive = change >= 0
    val scope = rememberCoroutineScope()

    val animatedPrice by animateFloatAsState(
        targetValue = price.toFloat(), animationSpec = tween(650, easing = FastOutSlowInEasing), label = "asset-price",
    )

    var flashTarget by remember { mutableStateOf(Color.Transparent) }
    val flash by animateColorAsState(flashTarget, tween(700), label = "asset-price-flash")
    var prevPrice by remember { mutableStateOf(price) }
    LaunchedEffect(price) {
        if (price > prevPrice) flashTarget = colors.green.copy(alpha = 0.22f)
        else if (price < prevPrice) flashTarget = colors.red.copy(alpha = 0.22f)
        prevPrice = price
        delay(150)
        flashTarget = Color.Transparent
    }

    val starScale = remember { Animatable(1f) }

    Box(modifier = Modifier.fillMaxWidth()) {
        Card(modifier = Modifier.fillMaxWidth(), highlighted = true) {
            Row(
                modifier = Modifier.fillMaxWidth(),
                horizontalArrangement = Arrangement.SpaceBetween,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                SimpleDropdown("دارایی", selected, tickers, { it }, onSelect, modifier = Modifier.weight(1f))
                if (isRefreshing) {
                    CircularProgressIndicator(modifier = Modifier.size(22.dp).padding(start = 8.dp), strokeWidth = 2.dp)
                } else {
                    IconButton(onClick = onRefresh, modifier = Modifier.padding(start = 4.dp)) {
                        Icon(Icons.Filled.Refresh, contentDescription = "بروزرسانی", tint = colors.muted)
                    }
                }
                IconButton(
                    onClick = {
                        onToggleFavorite()
                        scope.launch { starScale.animateTo(1.35f, tween(120)); starScale.animateTo(1f, tween(150)) }
                    },
                    modifier = Modifier.scale(starScale.value).padding(start = 4.dp),
                ) {
                    Icon(
                        if (isFavorite) Icons.Filled.Star else Icons.Filled.StarBorder,
                        contentDescription = "Favorite",
                        tint = if (isFavorite) colors.gold else colors.muted,
                    )
                }
            }
            Text(
                "$%,.2f".format(animatedPrice),
                style = MaterialTheme.typography.titleLarge.copy(fontSize = 34.sp),
                color = colors.textPrimary,
                modifier = Modifier.padding(top = 10.dp),
            )
            Row(verticalAlignment = Alignment.CenterVertically, modifier = Modifier.padding(top = 4.dp)) {
                Icon(
                    if (positive) Icons.Filled.ArrowUpward else Icons.Filled.ArrowDownward,
                    contentDescription = null,
                    tint = if (positive) colors.green else colors.red,
                    modifier = Modifier.size(16.dp),
                )
                Text(
                    "%s%,.2f (%s%.2f%%)".format(if (positive) "+" else "", change, if (positive) "+" else "", changePct),
                    color = if (positive) colors.green else colors.red,
                    style = MaterialTheme.typography.bodyMedium,
                    modifier = Modifier.padding(start = 4.dp),
                )
            }
        }
        Box(
            modifier = Modifier.matchParentSize().clip(RoundedCornerShape(18.dp)).background(flash),
        )
    }
}

@Composable
private fun RangeFilterRow(selected: RangeFilter, onSelect: (RangeFilter) -> Unit) {
    LazyRow(horizontalArrangement = Arrangement.spacedBy(8.dp), modifier = Modifier.fillMaxWidth()) {
        items(RangeFilter.entries) { r ->
            val isSelected = r == selected
            val scale by animateFloatAsState(if (isSelected) 1.08f else 1f, label = "range-chip-scale")
            FilterChip(
                selected = isSelected,
                onClick = { onSelect(r) },
                label = { Text(r.label) },
                modifier = Modifier.scale(scale),
            )
        }
    }
}

/** Animated line-draw chart with a tap/drag crosshair + tooltip; scoped to this screen only. */
@Composable
private fun AssetLineChart(dates: List<LocalDate>, closes: List<Double>, positive: Boolean, modifier: Modifier = Modifier) {
    val colors = LocalChartColors.current
    val lineColor = if (positive) colors.green else colors.red

    if (closes.size < 2) {
        Box(modifier.fillMaxWidth().height(220.dp), contentAlignment = Alignment.Center) {
            Text("داده کافی برای این بازه موجود نیست.", color = colors.muted, style = MaterialTheme.typography.bodySmall)
        }
        return
    }

    val progress = remember { Animatable(0f) }
    LaunchedEffect(dates, closes) {
        progress.snapTo(0f)
        progress.animateTo(1f, animationSpec = tween(900, easing = FastOutSlowInEasing))
    }

    var crosshairIndex by remember(dates) { mutableStateOf<Int?>(null) }
    val minV = closes.min()
    val maxV = closes.max()
    val valueRange = (maxV - minV).takeIf { it > 0 } ?: 1.0

    Box(modifier = modifier) {
        Canvas(
            modifier = Modifier.fillMaxSize()
                .pointerInput(dates, closes) {
                    detectDragGestures(
                        onDragStart = { pos ->
                            crosshairIndex = ((pos.x / size.width) * (closes.size - 1)).toInt().coerceIn(0, closes.size - 1)
                        },
                        onDrag = { change, _ ->
                            crosshairIndex = ((change.position.x / size.width) * (closes.size - 1)).toInt().coerceIn(0, closes.size - 1)
                        },
                        onDragEnd = { crosshairIndex = null },
                        onDragCancel = { crosshairIndex = null },
                    )
                },
        ) {
            val stepX = size.width / (closes.size - 1).coerceAtLeast(1)
            fun yFor(v: Double) = size.height - ((v - minV) / valueRange * size.height).toFloat()

            val linePath = Path().apply {
                closes.forEachIndexed { i, v ->
                    val x = i * stepX
                    val y = yFor(v)
                    if (i == 0) moveTo(x, y) else lineTo(x, y)
                }
            }
            val fillPath = Path().apply {
                addPath(linePath)
                lineTo((closes.size - 1) * stepX, size.height)
                lineTo(0f, size.height)
                close()
            }

            clipRect(right = size.width * progress.value) {
                drawPath(fillPath, brush = Brush.verticalGradient(listOf(lineColor.copy(alpha = 0.30f), Color.Transparent)))
                drawPath(linePath, color = lineColor, style = Stroke(width = 2.6.dp.toPx(), cap = StrokeCap.Round, join = StrokeJoin.Round))
            }

            crosshairIndex?.let { idx ->
                val x = idx * stepX
                val y = yFor(closes[idx])
                drawLine(colors.plotGrid, Offset(x, 0f), Offset(x, size.height), strokeWidth = 1.dp.toPx())
                drawCircle(lineColor, radius = 5.dp.toPx(), center = Offset(x, y))
                drawCircle(Color.White, radius = 2.dp.toPx(), center = Offset(x, y))
            }
        }

        crosshairIndex?.let { idx ->
            val isLeftHalf = idx < closes.size / 2
            Text(
                "${dates.getOrNull(idx)} · $%,.2f".format(closes[idx]),
                style = MaterialTheme.typography.labelSmall,
                color = colors.textPrimary,
                modifier = Modifier
                    .align(if (isLeftHalf) Alignment.TopStart else Alignment.TopEnd)
                    .padding(8.dp)
                    .background(colors.bg2.copy(alpha = 0.95f), RoundedCornerShape(8.dp))
                    .padding(horizontal = 8.dp, vertical = 4.dp),
            )
        }
    }
}
