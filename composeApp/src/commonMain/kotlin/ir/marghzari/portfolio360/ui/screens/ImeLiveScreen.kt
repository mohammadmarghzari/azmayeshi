package ir.marghzari.portfolio360.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.Checkbox
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
import ir.marghzari.portfolio360.charts.Candle
import ir.marghzari.portfolio360.charts.CandlestickChart
import ir.marghzari.portfolio360.charts.RadarChart
import ir.marghzari.portfolio360.charts.RadarSeries
import ir.marghzari.portfolio360.charts.VolumeBarChart
import ir.marghzari.portfolio360.core.math.ImeSignalAnalyzer
import ir.marghzari.portfolio360.core.model.ImeQuote
import ir.marghzari.portfolio360.core.network.ImeClient
import ir.marghzari.portfolio360.state.AppState
import ir.marghzari.portfolio360.theme.LocalChartColors
import ir.marghzari.portfolio360.theme.chartColor
import ir.marghzari.portfolio360.ui.components.Card
import ir.marghzari.portfolio360.ui.components.InfoBanner
import ir.marghzari.portfolio360.ui.components.SectionHeader
import ir.marghzari.portfolio360.ui.components.SimpleDropdown
import kotlinx.coroutines.launch

@Composable
fun ImeLiveScreen(appState: AppState) {
    val colors = LocalChartColors.current
    val scope = rememberCoroutineScope()

    var quotes by remember { mutableStateOf<List<ImeQuote>>(emptyList()) }
    var loadError by remember { mutableStateOf<String?>(null) }
    var loading by remember { mutableStateOf(false) }

    suspend fun load() {
        loading = true
        loadError = null
        when (val result = appState.ime.fetchLastChanges()) {
            is ImeClient.ImeApiResult.Ok -> quotes = result.rows.map { ImeClient.parseQuote(it) }
            is ImeClient.ImeApiResult.Err -> loadError = result.message
        }
        loading = false
    }

    LaunchedEffect(Unit) { load() }

    LazyColumn(verticalArrangement = Arrangement.spacedBy(16.dp), modifier = Modifier.fillMaxWidth()) {
        item {
            SectionHeader("📡 IME Live — گواهی سپرده کالایی")
            Text("بورس کالای ایران · API رسمی · نرخ تازه‌سازی: ۶۰ ثانیه · منبع: api.ime.co.ir", style = MaterialTheme.typography.bodySmall, color = colors.muted)
            Button(
                onClick = { scope.launch { load() } }, modifier = Modifier.fillMaxWidth().padding(top = 8.dp),
                colors = ButtonDefaults.buttonColors(containerColor = colors.blueAccent),
            ) { Text("🔄 بروزرسانی") }
        }

        if (loading) {
            item { CircularProgressIndicator() }
        } else if (loadError != null) {
            item {
                InfoBanner("❌ خطا در اتصال به API: ${loadError}")
                Text(
                    "راه‌حل‌های احتمالی: ① دامنه api.ime.co.ir را در egress settings اضافه کنید. ② برنامه را روی سرور ایرانی اجرا کنید.",
                    style = MaterialTheme.typography.labelSmall, color = colors.muted, modifier = Modifier.padding(top = 6.dp),
                )
            }
        } else if (quotes.isEmpty()) {
            item { InfoBanner("داده‌ای یافت نشد.") }
        } else {
            item {
                SectionHeader("① تابلوی زنده — آخرین قیمت‌ها")
                Card(modifier = Modifier.fillMaxWidth()) {
                    quotes.forEach { q ->
                        Row(modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp), horizontalArrangement = Arrangement.SpaceBetween) {
                            Text(q.commodity, color = colors.textPrimary, style = MaterialTheme.typography.bodyMedium)
                            Text("%,.0f ریال".format(q.pl), color = colors.textPrimary, style = MaterialTheme.typography.bodyMedium)
                            Text(
                                "%s%.2f%%".format(if (q.plChangePct >= 0) "▲" else "▼", q.plChangePct),
                                color = if (q.plChangePct >= 0) colors.green else colors.red, style = MaterialTheme.typography.bodyMedium,
                            )
                        }
                    }
                    BarChart(
                        categories = quotes.map { it.commodity }, series = listOf(BarSeries("تغییر %", colors.blueAccent, quotes.map { it.plChangePct })),
                        title = "نمودار تغییر درصدی", showLegend = false,
                        perBarColorOverride = { _, i, v -> if (v >= 0) colors.green else colors.red },
                    )
                }
            }

            item {
                SectionHeader("② جزئیات نماد — دفتر سفارشات")
                var selected by remember(quotes) { mutableStateOf(quotes.first()) }
                SimpleDropdown("انتخاب نماد", selected, quotes, { "${it.commodity} · ${it.contractCode}" }, { selected = it })
                Card(modifier = Modifier.fillMaxWidth().padding(top = 8.dp)) {
                    Text("📋 دفتر سفارشات — ${selected.commodity}", style = MaterialTheme.typography.titleSmall, color = colors.textPrimary)
                    for (i in 0..2) {
                        Row(modifier = Modifier.fillMaxWidth().padding(vertical = 2.dp), horizontalArrangement = Arrangement.SpaceBetween) {
                            Text("%,.0f".format(selected.bidQtys.getOrElse(i) { 0.0 }), color = colors.green, style = MaterialTheme.typography.labelSmall)
                            Text("%,.0f".format(selected.bidPrices.getOrElse(i) { 0.0 }), color = colors.green, style = MaterialTheme.typography.labelSmall)
                            Text("%,.0f".format(selected.askPrices.getOrElse(i) { 0.0 }), color = colors.red, style = MaterialTheme.typography.labelSmall)
                            Text("%,.0f".format(selected.askQtys.getOrElse(i) { 0.0 }), color = colors.red, style = MaterialTheme.typography.labelSmall)
                        }
                    }
                    Text(
                        "کد قرارداد: ${selected.contractCode} · اندازه قرارداد: ${selected.contractSize} · حجم: %,.0f · ارزش: %,.0f %s".format(selected.volume, selected.tradeValue, selected.tradeValueUnit),
                        style = MaterialTheme.typography.labelSmall, color = colors.muted, modifier = Modifier.padding(top = 8.dp),
                    )
                }
            }

            if (quotes.size >= 2) {
                item {
                    SectionHeader("④ مقایسه رادار — نمادها")
                    val symbols = quotes.take(6)
                    val maxChange = symbols.maxOf { kotlin.math.abs(it.plChangePct) }.takeIf { it > 0 } ?: 1.0
                    val maxVol = symbols.maxOf { it.volume }.takeIf { it > 0 } ?: 1.0
                    val maxVal = symbols.maxOf { it.tradeValue }.takeIf { it > 0 } ?: 1.0
                    val maxTno = symbols.maxOf { it.tradeCount }.takeIf { it > 0 } ?: 1
                    Card(modifier = Modifier.fillMaxWidth()) {
                        RadarChart(
                            axisLabels = listOf("تغییر %", "دامنه روز", "حجم", "ارزش", "تعداد معامله"),
                            series = symbols.mapIndexed { i, q ->
                                RadarSeries(
                                    q.commodity, chartColor(i),
                                    listOf(
                                        kotlin.math.abs(q.plChangePct) / maxChange * 100,
                                        (q.pMaxChangePct - q.pMinChangePct).let { kotlin.math.abs(it) }.coerceAtMost(100.0),
                                        q.volume / maxVol * 100, q.tradeValue / maxVal * 100, q.tradeCount / maxTno.toDouble() * 100,
                                    ),
                                )
                            },
                            title = "RADAR — مقایسه نمادها (نرمال‌شده)",
                        )
                    }
                }
            }

            item {
                SectionHeader("⑤ تحلیل‌گر هوشمند — سیگنال خودکار")
                Card(modifier = Modifier.fillMaxWidth()) {
                    quotes.forEach { q ->
                        val verdict = ImeSignalAnalyzer.analyze(q)
                        Row(modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp), horizontalArrangement = Arrangement.SpaceBetween) {
                            Text(q.commodity, color = colors.textPrimary, style = MaterialTheme.typography.bodySmall)
                            Text(verdict.labelFa, style = MaterialTheme.typography.bodySmall)
                            Text("امتیاز: %+d".format(verdict.score), color = colors.muted, style = MaterialTheme.typography.labelSmall)
                        }
                    }
                    Text(
                        "⚠ این سیگنال‌ها صرفاً بر اساس داده لحظه‌ای API و قوانین ساده هستند و توصیه سرمایه‌گذاری نیستند.",
                        style = MaterialTheme.typography.labelSmall, color = colors.muted, modifier = Modifier.padding(top = 8.dp),
                    )
                }
            }

            item { HistorySection(appState, quotes) }
        }
    }
}

@Composable
private fun HistorySection(appState: AppState, quotes: List<ImeQuote>) {
    val colors = LocalChartColors.current
    val scope = rememberCoroutineScope()
    SectionHeader("③ تاریخچه قیمت — نمودار شمعی")
    var histSymbol by remember(quotes) { mutableStateOf(quotes.first()) }
    var fromDate by remember { mutableStateOf("1404-01-01") }
    var toDate by remember { mutableStateOf("1404-07-02") }
    var candles by remember { mutableStateOf<List<Candle>>(emptyList()) }
    var histError by remember { mutableStateOf<String?>(null) }

    SimpleDropdown("نماد تاریخچه", histSymbol, quotes, { "${it.commodity} · ${it.contractCode}" }, { histSymbol = it })
    Button(
        onClick = {
            scope.launch {
                when (val result = appState.ime.fetchHistory(histSymbol.contractCode, fromDate, toDate)) {
                    is ImeClient.ImeApiResult.Ok -> {
                        candles = result.rows.map { row -> ImeClient.parseCandle(row).let { Candle(it.open, it.high, it.low, it.close) } }
                        histError = if (candles.isEmpty()) "داده‌ای برای این بازه یافت نشد." else null
                    }
                    is ImeClient.ImeApiResult.Err -> histError = "خطا در دریافت تاریخچه: ${result.message}"
                }
            }
        },
        modifier = Modifier.fillMaxWidth().padding(top = 8.dp), colors = ButtonDefaults.buttonColors(containerColor = colors.blueAccent),
    ) { Text("دریافت تاریخچه") }

    histError?.let { Text(it, color = colors.red, modifier = Modifier.padding(top = 6.dp)) }

    if (candles.isNotEmpty()) {
        val closes = candles.map { it.close }
        fun ma(period: Int): List<Double?> = closes.indices.map { i -> if (i >= period - 1) closes.subList(i - period + 1, i + 1).average() else null }
        Card(modifier = Modifier.fillMaxWidth().padding(top = 8.dp)) {
            CandlestickChart(candles = candles, ma20 = if (closes.size >= 20) ma(20) else null, ma50 = if (closes.size >= 50) ma(50) else null, title = "نمودار شمعی — ${histSymbol.commodity}")
            val upFlags = candles.mapIndexed { i, c -> if (i == 0) true else c.close >= candles[i - 1].close }
            VolumeBarChart(volumes = List(candles.size) { 1.0 }, upFlags = upFlags, title = "حجم معاملات تاریخی")

            val totalReturn = (closes.last() / closes.first() - 1) * 100
            val dailyReturns = (1 until closes.size).map { closes[it] / closes[it - 1] - 1 }
            val vol = if (dailyReturns.isNotEmpty()) kotlin.math.sqrt(dailyReturns.map { it * it }.average()) * kotlin.math.sqrt(252.0) * 100 else 0.0
            Text("بازده کل دوره: %.2f%% · نوسان سالانه‌شده: %.2f%%".format(totalReturn, vol), style = MaterialTheme.typography.labelSmall, color = colors.textPrimary, modifier = Modifier.padding(top = 8.dp))
        }
    }
}
