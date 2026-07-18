package ir.marghzari.portfolio360.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
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
import ir.marghzari.portfolio360.charts.BarChart
import ir.marghzari.portfolio360.charts.BarSeries
import ir.marghzari.portfolio360.charts.GaugeChart
import ir.marghzari.portfolio360.charts.GaugeZone
import ir.marghzari.portfolio360.charts.HeatmapChart
import ir.marghzari.portfolio360.core.math.Seasonality
import ir.marghzari.portfolio360.core.network.FearGreedData
import ir.marghzari.portfolio360.core.network.NewsItem
import ir.marghzari.portfolio360.state.AppState
import ir.marghzari.portfolio360.theme.LocalChartColors
import ir.marghzari.portfolio360.ui.components.Card
import ir.marghzari.portfolio360.ui.components.InfoBanner
import ir.marghzari.portfolio360.ui.components.MetricTile
import ir.marghzari.portfolio360.ui.components.SectionHeader
import ir.marghzari.portfolio360.ui.components.SimpleDropdown
import kotlinx.coroutines.launch

private fun fgZoneLabel(score: Double): Triple<String, String, Boolean> = when {
    score <= 25 -> Triple("ترس شدید", "😱", true)
    score <= 45 -> Triple("ترس", "😨", true)
    score <= 55 -> Triple("خنثی", "😐", false)
    score <= 75 -> Triple("طمع", "😏", false)
    else -> Triple("طمع شدید", "🤑", false)
}

@Composable
fun LiveDataScreen(appState: AppState) {
    val colors = LocalChartColors.current
    val scope = rememberCoroutineScope()

    var fg by remember { mutableStateOf<FearGreedData?>(null) }
    var fgError by remember { mutableStateOf<String?>(null) }
    var fgLoading by remember { mutableStateOf(false) }

    val prices = appState.prices
    var newsTicker by remember(prices) { mutableStateOf(prices?.tickers?.firstOrNull().orEmpty()) }
    var news by remember { mutableStateOf<List<NewsItem>>(emptyList()) }
    var seasTicker by remember(prices) { mutableStateOf(prices?.tickers?.firstOrNull().orEmpty()) }

    LaunchedEffect(Unit) {
        fgLoading = true
        fg = appState.fearGreed.fetch()
        if (fg == null) fgError = "دریافت Fear & Greed ناموفق بود."
        fgLoading = false
    }

    LazyColumn(verticalArrangement = Arrangement.spacedBy(16.dp), modifier = Modifier.fillMaxWidth()) {
        item {
            SectionHeader("Fear & Greed Index")
            Text("شاخص ترس و طمع CNN — بدون API، مستقیم از سایت.", style = MaterialTheme.typography.bodySmall, color = colors.muted)
        }
        item {
            if (fgLoading) {
                CircularProgressIndicator()
            } else if (fg != null) {
                val (label, emoji, _) = fgZoneLabel(fg!!.score)
                Card(modifier = Modifier.fillMaxWidth()) {
                    GaugeChart(
                        value = fg!!.score, valueRange = 0.0..100.0,
                        zones = listOf(
                            GaugeZone(0.0, 25.0, colors.red.copy(alpha = 0.5f)), GaugeZone(25.0, 45.0, colors.riskGeo.copy(alpha = 0.5f)),
                            GaugeZone(45.0, 55.0, colors.gold.copy(alpha = 0.5f)), GaugeZone(55.0, 75.0, colors.green.copy(alpha = 0.4f)),
                            GaugeZone(75.0, 100.0, colors.green.copy(alpha = 0.7f)),
                        ),
                        needleColor = colors.textPrimary, centerLabel = "$emoji $label",
                    )
                }
                item2Row(fg!!)
                Text(
                    "📌 0–25 ترس شدید — فرصت خرید احتمالی | 26–45 ترس | 46–55 خنثی | 56–75 طمع | 76–100 طمع شدید — احتیاط در خرید",
                    style = MaterialTheme.typography.labelSmall, color = colors.muted, modifier = Modifier.padding(top = 6.dp),
                )
            } else if (fgError != null) {
                InfoBanner(fgError!!)
            }
        }

        item {
            SectionHeader("اخبار نمادهای پرتفوی")
            Text("آخرین اخبار از Yahoo Finance RSS.", style = MaterialTheme.typography.bodySmall, color = colors.muted)
        }
        if (prices == null) {
            item { InfoBanner("ابتدا داده پرتفوی را دانلود کنید.") }
        } else {
            item {
                SimpleDropdown("نماد", newsTicker, prices.tickers, { it }, { t -> newsTicker = t; scope.launch { news = appState.news.fetch(t) } })
            }
            item {
                Button(onClick = { scope.launch { news = appState.news.fetch(newsTicker) } }, colors = ButtonDefaults.buttonColors(containerColor = colors.blueAccent)) {
                    Text("دریافت اخبار $newsTicker")
                }
            }
            if (news.isNotEmpty()) {
                items(news) { item ->
                    Card(modifier = Modifier.fillMaxWidth()) {
                        Text(item.title, color = colors.blueAccent, style = MaterialTheme.typography.bodyMedium)
                        Text(item.pubDate.take(22), color = colors.muted, style = MaterialTheme.typography.labelSmall, modifier = Modifier.padding(top = 4.dp))
                    }
                }
            }

            item {
                SectionHeader("Seasonality — بازده ماهانه تاریخی")
                Text("میانگین بازده هر ماه در سال‌های گذشته.", style = MaterialTheme.typography.bodySmall, color = colors.muted)
            }
            item { SimpleDropdown("نماد برای Seasonality", seasTicker, prices.tickers, { it }, { seasTicker = it }) }
            item {
                val monthly = Seasonality.monthlyReturns(prices.dates, prices.column(seasTicker))
                val stats = Seasonality.byMonth(monthly)
                Card(modifier = Modifier.fillMaxWidth()) {
                    BarChart(
                        categories = Seasonality.MONTH_ABBREV_EN,
                        series = listOf(BarSeries("میانگین", colors.blueAccent, stats.map { it.avgReturnPct })),
                        title = "SEASONALITY — $seasTicker", showLegend = false,
                        perBarColorOverride = { _, i, v -> if (v >= 0) colors.green else colors.red },
                    )
                }
                if (monthly.isNotEmpty()) {
                    val years = monthly.map { it.year }.distinct().sorted()
                    val grid = Array(years.size) { y -> DoubleArray(12) { m -> monthly.firstOrNull { it.year == years[y] && it.month == m + 1 }?.returnPct ?: 0.0 } }
                    Card(modifier = Modifier.fillMaxWidth().padding(top = 10.dp)) {
                        HeatmapChart(
                            rowLabels = years.map { it.toString() }, colLabels = Seasonality.MONTH_ABBREV_EN, values = grid,
                            title = "MONTHLY RETURN HEATMAP — $seasTicker", valueRange = -10.0 to 10.0,
                            valueFormatter = { "%.1f".format(it) },
                        )
                    }
                }
                val best = stats.maxByOrNull { it.avgReturnPct }
                val worst = stats.minByOrNull { it.avgReturnPct }
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.padding(top = 10.dp)) {
                    best?.let { MetricTile("بهترین ماه تاریخی", Seasonality.MONTH_ABBREV_EN[it.month - 1], caption = "%.0f%% سال‌های مثبت".format(it.positiveRatePct)) }
                    worst?.let { MetricTile("بدترین ماه تاریخی", Seasonality.MONTH_ABBREV_EN[it.month - 1], caption = "%.0f%% سال‌های مثبت".format(it.positiveRatePct)) }
                }
            }
        }
    }
}

@Composable
private fun item2Row(fg: FearGreedData) {
    LazyRow(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
        items(
            listOfNotNull(
                fg.previousClose?.let { "دیروز" to "%.0f".format(it) },
                fg.previousWeek?.let { "هفته پیش" to "%.0f".format(it) },
                fg.previousMonth?.let { "ماه پیش" to "%.0f".format(it) },
                fg.previousYear?.let { "سال پیش" to "%.0f".format(it) },
            ),
        ) { (l, v) -> MetricTile(l, v) }
    }
}
