package ir.marghzari.portfolio360.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import ir.marghzari.portfolio360.charts.BarChart
import ir.marghzari.portfolio360.charts.BarSeries
import ir.marghzari.portfolio360.charts.LineChart
import ir.marghzari.portfolio360.charts.LineSeries
import ir.marghzari.portfolio360.charts.RefLine
import ir.marghzari.portfolio360.core.math.MonteCarloEngine
import ir.marghzari.portfolio360.core.math.MonteCarloResult
import ir.marghzari.portfolio360.core.math.StressTest
import ir.marghzari.portfolio360.core.math.StressTestRow
import ir.marghzari.portfolio360.state.AppState
import ir.marghzari.portfolio360.theme.LocalChartColors
import ir.marghzari.portfolio360.ui.components.EmptyState
import ir.marghzari.portfolio360.ui.components.Card
import ir.marghzari.portfolio360.ui.components.MetricTile
import ir.marghzari.portfolio360.ui.components.SectionHeader

@Composable
fun StressMonteCarloScreen(appState: AppState) {
    val colors = LocalChartColors.current
    val weights = appState.weights
    val prices = appState.prices
    if (weights == null || prices == null) {
        EmptyState(title = "هنوز پرتفویی محاسبه نشده", hint = "از تب «تخصیص دارایی» داده را دریافت و پرتفوی را محاسبه کنید.")
        return
    }

    var stressRows by remember { mutableStateOf<List<StressTestRow>>(emptyList()) }
    var mcResult by remember { mutableStateOf<MonteCarloResult?>(null) }
    var horizonYears by remember { mutableStateOf(3f) }
    var initialCapital by remember { mutableStateOf(10000.0) }

    LazyColumn(verticalArrangement = Arrangement.spacedBy(16.dp), modifier = Modifier.fillMaxWidth()) {
        item {
            SectionHeader("Stress Test — مقاومت در برابر بحران‌های تاریخی")
            Text("شبیه‌سازی می‌کند اگه پرتفوی فعلی در هر بحران تاریخی بود، چه اتفاقی می‌افتاد.", style = MaterialTheme.typography.bodySmall, color = colors.muted)
            Button(
                onClick = { stressRows = StressTest.run(prices.dates, prices.values, prices.tickers, weights) },
                modifier = Modifier.fillMaxWidth().padding(top = 10.dp), colors = ButtonDefaults.buttonColors(containerColor = colors.red),
            ) { Text("اجرای Stress Test") }
        }
        if (stressRows.isNotEmpty()) {
            item {
                Card(modifier = Modifier.fillMaxWidth()) {
                    stressRows.forEach { r ->
                        Text("${r.crisisName}: بازده %.1f%% · نوسان %.1f%% · افت %.1f%% · %d روز".format(r.totalReturnPct, r.annualVolPct, r.maxDrawdownPct, r.days), color = colors.textPrimary, style = MaterialTheme.typography.bodySmall)
                    }
                    BarChart(
                        categories = stressRows.map { it.crisisName },
                        series = listOf(
                            BarSeries("بازده کل (%)", colors.blueAccent, stressRows.map { it.totalReturnPct }),
                            BarSeries("حداکثر افت (%)", colors.red, stressRows.map { it.maxDrawdownPct }),
                        ),
                        title = "STRESS TEST — عملکرد در بحران‌ها",
                    )
                }
            }
        }

        item {
            SectionHeader("Monte Carlo — شبیه‌سازی آینده")
            Text("افق زمانی: ${horizonYears.toInt()} سال", style = MaterialTheme.typography.labelMedium)
            Slider(value = horizonYears, onValueChange = { horizonYears = it }, valueRange = 1f..10f, steps = 8)
            Button(
                onClick = { mcResult = MonteCarloEngine.simulate(weights, prices.dailyReturns(), horizonYears = horizonYears.toDouble()) },
                modifier = Modifier.fillMaxWidth().padding(top = 10.dp), colors = ButtonDefaults.buttonColors(containerColor = colors.blueAccent),
            ) { Text("اجرای Monte Carlo") }
        }
        mcResult?.let { mc ->
            item {
                LazyRow(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    items(
                        listOf(
                            "احتمال سود" to "%.1f%%".format(mc.probProfitPct), "احتمال دو برابر شدن" to "%.1f%%".format(mc.prob2xPct),
                            "میانه سرمایه نهایی" to "$%,.0f".format(mc.median * initialCapital), "بدترین ۵٪" to "$%,.0f".format(mc.worst5 * initialCapital),
                        ),
                    ) { (l, v) -> MetricTile(l, v) }
                }
            }
            item {
                Card(modifier = Modifier.fillMaxWidth()) {
                    LineChart(
                        series = listOf(
                            LineSeries("۹۵٪", colors.green, mc.pct95.map { it * initialCapital }, dashed = true),
                            LineSeries("۷۵٪", colors.green, mc.pct75.map { it * initialCapital }),
                            LineSeries("میانه", colors.blueAccent, mc.pct50.map { it * initialCapital }, strokeWidth = 3.dp),
                            LineSeries("۲۵٪", colors.red, mc.pct25.map { it * initialCapital }),
                            LineSeries("کف ۵٪", colors.red, mc.pct5.map { it * initialCapital }, dashed = true),
                        ),
                        horizontalRefLines = listOf(RefLine(initialCapital, colors.muted, "سرمایه اولیه")),
                        title = "MONTE CARLO — ${horizonYears.toInt()} سال آینده (400 مسیر)", height = 380.dp,
                    )
                }
            }
            item {
                val finalVals = mc.final.map { it * initialCapital }
                val bins = 40
                val min = finalVals.min(); val max = finalVals.max()
                val width = ((max - min) / bins).takeIf { it > 0 } ?: 1.0
                val counts = DoubleArray(bins)
                finalVals.forEach { v -> val idx = ((v - min) / width).toInt().coerceIn(0, bins - 1); counts[idx]++ }
                Card(modifier = Modifier.fillMaxWidth()) {
                    BarChart(
                        categories = (0 until bins).map { formatShort(min + width * it) },
                        series = listOf(BarSeries("تعداد مسیر", colors.blueAccent, counts.toList())),
                        title = "توزیع سرمایه نهایی", showLegend = false, zeroLine = false,
                    )
                }
            }
        }
    }
}

private fun formatShort(v: Double): String = if (v >= 1000) "%.0fK".format(v / 1000) else "%.0f".format(v)
