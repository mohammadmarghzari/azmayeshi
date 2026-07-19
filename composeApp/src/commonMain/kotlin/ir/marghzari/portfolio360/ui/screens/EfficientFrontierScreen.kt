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
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import ir.marghzari.portfolio360.charts.BarChart
import ir.marghzari.portfolio360.charts.BarSeries
import ir.marghzari.portfolio360.charts.LineChart
import ir.marghzari.portfolio360.charts.LineSeries
import ir.marghzari.portfolio360.charts.RefLine
import ir.marghzari.portfolio360.charts.ScatterChart
import ir.marghzari.portfolio360.charts.ScatterMarker
import ir.marghzari.portfolio360.charts.ScatterPoint
import ir.marghzari.portfolio360.core.math.EfficientFrontier
import ir.marghzari.portfolio360.core.math.EfficientFrontierResult
import ir.marghzari.portfolio360.core.math.RollingMetrics
import ir.marghzari.portfolio360.core.math.Stats
import ir.marghzari.portfolio360.state.AppState
import ir.marghzari.portfolio360.theme.LocalChartColors
import ir.marghzari.portfolio360.ui.components.EmptyState
import ir.marghzari.portfolio360.ui.components.Card
import ir.marghzari.portfolio360.ui.components.MetricTile
import ir.marghzari.portfolio360.ui.components.SectionHeader
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

@Composable
fun EfficientFrontierScreen(appState: AppState) {
    val colors = LocalChartColors.current
    val weights = appState.weights
    val prices = appState.prices
    if (weights == null || prices == null) {
        EmptyState(title = "هنوز پرتفویی محاسبه نشده", hint = "از تب «تخصیص دارایی» داده را دریافت و پرتفوی را محاسبه کنید.")
        return
    }
    val scope = rememberCoroutineScope()
    var ef by remember { mutableStateOf<EfficientFrontierResult?>(null) }
    var running by remember { mutableStateOf(false) }

    LazyColumn(verticalArrangement = Arrangement.spacedBy(16.dp), modifier = Modifier.fillMaxWidth()) {
        item {
            SectionHeader("Efficient Frontier — مرز کارایی پرتفوی")
            Text(
                "هر نقطه یک ترکیب ممکن از دارایی‌هاست. مرز کارایی بالاترین بازده به ازای هر سطح ریسک را نشان می‌دهد.",
                style = MaterialTheme.typography.bodySmall, color = colors.muted,
            )
        }
        item {
            Button(
                onClick = {
                    scope.launch {
                        running = true
                        ef = withContext(Dispatchers.Default) {
                            EfficientFrontier.compute(weights, prices.dailyReturns(), prices.tickers, appState.rf)
                        }
                        running = false
                    }
                },
                enabled = !running,
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.buttonColors(containerColor = colors.blueAccent),
            ) {
                if (running) CircularProgressIndicator(modifier = Modifier.padding(2.dp), color = MaterialTheme.colorScheme.onPrimary, strokeWidth = 2.dp)
                else Text("▶ رسم Efficient Frontier")
            }
        }

        ef?.let { result ->
            item {
                Card(modifier = Modifier.fillMaxWidth()) {
                    ScatterChart(
                        points = result.simulated.map { ScatterPoint(it.volPct, it.retPct, it.sharpe) },
                        lineOverlay = result.frontier.map { it.volPct to it.retPct },
                        markers = listOfNotNull(
                            ScatterMarker(result.currentVolPct, result.currentRetPct, "پرتفوی شما", colors.gold),
                            result.bestSharpePoint?.let { ScatterMarker(it.volPct, it.retPct, "بیشترین شارپ", colors.green) },
                        ),
                        title = "EFFICIENT FRONTIER",
                        xFormatter = { "%.1f%%".format(it) }, yFormatter = { "%.1f%%".format(it) },
                    )
                }
            }
            item {
                LazyRow(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    items(
                        listOf(
                            "بازده پرتفوی فعلی" to "%.2f%%".format(result.currentRetPct),
                            "نوسان پرتفوی فعلی" to "%.2f%%".format(result.currentVolPct),
                            "نسبت شارپ فعلی" to "%.3f".format(result.currentSharpe),
                        ),
                    ) { (l, v) -> MetricTile(l, v) }
                }
            }
        }

        item {
            SectionHeader("Rolling Metrics — تغییرات شارپ و نوسان در طول زمان")
            var window by remember { mutableStateOf(63f) }
            Text("پنجره Rolling: ${window.toInt()} روز", style = MaterialTheme.typography.labelMedium, color = colors.textPrimary)
            Slider(value = window, onValueChange = { window = it }, valueRange = 21f..252f, steps = 10)

            val returns = prices.dailyReturns()
            val portRet = Stats.portfolioReturns(returns, weights)
            val rolling = remember(window) { RollingMetrics.compute(portRet, window.toInt(), appState.rf) }
            val validSharpe = rolling.rollingSharpe.filter { !it.isNaN() }

            Card(modifier = Modifier.fillMaxWidth()) {
                LineChart(
                    series = listOf(LineSeries("Sharpe", colors.blueAccent, rolling.rollingSharpe.map { if (it.isNaN()) 0.0 else it }, fillToZero = true)),
                    horizontalRefLines = listOf(RefLine(0.0, colors.plotTick), RefLine(1.0, colors.green, "Sharpe=1")),
                    title = "ROLLING SHARPE", showLegend = false,
                )
            }
            Card(modifier = Modifier.fillMaxWidth()) {
                LineChart(
                    series = listOf(
                        LineSeries("Volatility", colors.red, rolling.rollingVolPct.map { if (it.isNaN()) 0.0 else it }, fillToZero = true),
                        LineSeries("Return", colors.green, rolling.rollingReturnPct.map { if (it.isNaN()) 0.0 else it }),
                    ),
                    title = "ROLLING VOLATILITY & RETURN",
                )
            }

            if (validSharpe.size > 10) {
                val pctPositive = validSharpe.count { it > 0 } * 100.0 / validSharpe.size
                val pctAbove1 = validSharpe.count { it > 1 } * 100.0 / validSharpe.size
                val bins = 30
                val min = validSharpe.min(); val max = validSharpe.max()
                val width = ((max - min) / bins).takeIf { it > 0 } ?: 1.0
                val counts = DoubleArray(bins)
                validSharpe.forEach { v -> val idx = ((v - min) / width).toInt().coerceIn(0, bins - 1); counts[idx]++ }
                Card(modifier = Modifier.fillMaxWidth()) {
                    BarChart(
                        categories = (0 until bins).map { "%.1f".format(min + width * it) },
                        series = listOf(BarSeries("Frequency", colors.blueAccent, counts.toList())),
                        title = "ROLLING SHARPE DISTRIBUTION", showLegend = false, zeroLine = false,
                    )
                }
                Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    MetricTile("درصد دوره‌های Sharpe > 0", "%.1f%%".format(pctPositive))
                    MetricTile("درصد دوره‌های Sharpe > 1", "%.1f%%".format(pctAbove1))
                }
            }
        }
    }
}
