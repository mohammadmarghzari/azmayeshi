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
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import ir.marghzari.portfolio360.charts.LineChart
import ir.marghzari.portfolio360.charts.LineSeries
import ir.marghzari.portfolio360.charts.RefLine
import ir.marghzari.portfolio360.core.math.Benchmark
import ir.marghzari.portfolio360.core.math.BenchmarkComparison
import ir.marghzari.portfolio360.core.math.Stats
import ir.marghzari.portfolio360.core.model.HistoryPeriod
import ir.marghzari.portfolio360.state.AppState
import ir.marghzari.portfolio360.theme.LocalChartColors
import ir.marghzari.portfolio360.ui.components.EmptyState
import ir.marghzari.portfolio360.ui.components.Card
import ir.marghzari.portfolio360.ui.components.MetricTile
import ir.marghzari.portfolio360.ui.components.SectionHeader
import ir.marghzari.portfolio360.ui.components.SimpleDropdown
import ir.marghzari.portfolio360.ui.components.VerdictCard
import ir.marghzari.portfolio360.ui.components.VerdictTone
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private val BENCHMARKS = listOf("SPY", "QQQ", "DIA", "IWM", "BTC-USD", "GLD", "TLT")

@Composable
fun BenchmarkScreen(appState: AppState) {
    val colors = LocalChartColors.current
    val weights = appState.weights
    val prices = appState.prices
    if (weights == null || prices == null) {
        EmptyState(title = "هنوز پرتفویی محاسبه نشده", hint = "از تب «تخصیص دارایی» داده را دریافت و پرتفوی را محاسبه کنید.")
        return
    }
    val scope = rememberCoroutineScope()
    var benchmark by remember { mutableStateOf("SPY") }
    var period by remember { mutableStateOf(HistoryPeriod.Y2) }
    var comparison by remember { mutableStateOf<BenchmarkComparison?>(null) }
    var running by remember { mutableStateOf(false) }
    var error by remember { mutableStateOf<String?>(null) }

    LazyColumn(verticalArrangement = Arrangement.spacedBy(16.dp), modifier = Modifier.fillMaxWidth()) {
        item {
            SectionHeader("📊 Benchmark Comparison — مقایسه با شاخص")
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                SimpleDropdown("بنچمارک", benchmark, BENCHMARKS, { it }, { benchmark = it }, modifier = Modifier.weight(1f))
                SimpleDropdown("بازه", period, HistoryPeriod.entries, { it.faLabel }, { period = it }, modifier = Modifier.weight(1f))
            }
            Button(
                onClick = {
                    scope.launch {
                        running = true
                        error = null
                        val bench = withContext(Dispatchers.Default) { appState.yahoo.fetchHistory(benchmark, period.apiCode) }
                        if (bench == null) {
                            error = "دانلود داده بنچمارک $benchmark ناموفق بود."
                        } else {
                            val portRet = Stats.portfolioReturns(prices.dailyReturns(), weights)
                            val portDates = prices.dates.drop(1)
                            val benchDateIdx = bench.dates.withIndex().associate { (i, d) -> d to i }
                            val commonDates = portDates.filter { it in benchDateIdx }
                            if (commonDates.size < 20) {
                                error = "داده کافی برای مقایسه وجود ندارد."
                            } else {
                                val portMap = portDates.withIndex().associate { (i, d) -> d to portRet[i] }
                                val benchReturns = DoubleArray(bench.closes.size - 1) { i -> bench.closes[i + 1] / bench.closes[i] - 1.0 }
                                val benchRetMap = bench.dates.drop(1).withIndex().associate { (i, d) -> d to benchReturns[i] }
                                val alignedDates = commonDates.filter { it in benchRetMap }
                                val p = DoubleArray(alignedDates.size) { portMap.getValue(alignedDates[it]) }
                                val b = DoubleArray(alignedDates.size) { benchRetMap.getValue(alignedDates[it]) }
                                comparison = Benchmark.compare(alignedDates, p, b)
                            }
                        }
                        running = false
                    }
                },
                enabled = !running,
                modifier = Modifier.fillMaxWidth().padding(top = 10.dp), colors = ButtonDefaults.buttonColors(containerColor = colors.blueAccent),
            ) {
                if (running) CircularProgressIndicator(modifier = Modifier.padding(2.dp), color = MaterialTheme.colorScheme.onPrimary, strokeWidth = 2.dp)
                else Text("مقایسه با بنچمارک")
            }
            error?.let { Text(it, color = colors.red, style = MaterialTheme.typography.labelSmall, modifier = Modifier.padding(top = 6.dp)) }
        }

        comparison?.let { c ->
            item {
                VerdictCard(
                    title = if (c.outperform) "✅ پرتفوی بهتر از $benchmark" else "❌ پرتفوی ضعیف‌تر از $benchmark",
                    body = "بازده پرتفو: %.2f%% در مقابل بازده %s: %.2f%%".format(c.portfolioAnnualReturnPct, benchmark, c.benchmarkAnnualReturnPct),
                    tone = if (c.outperform) VerdictTone.POSITIVE else VerdictTone.NEGATIVE,
                )
            }
            item {
                LazyRow(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                    items(
                        listOf(
                            "Alpha (α)" to "%.2f%%".format(c.alphaPct), "Beta (β)" to "%.3f".format(c.beta),
                            "Tracking Error" to "%.2f%%".format(c.trackingErrorPct), "Information Ratio" to "%.3f".format(c.informationRatio),
                        ),
                    ) { (l, v) -> MetricTile(l, v) }
                }
            }
            item {
                Card(modifier = Modifier.fillMaxWidth()) {
                    LineChart(
                        series = listOf(
                            LineSeries("پرتفوی", colors.blueAccent, c.portfolioCumulative.toList()),
                            LineSeries(benchmark, colors.gold, c.benchmarkCumulative.toList(), dashed = true),
                        ),
                        horizontalRefLines = listOf(RefLine(1.0, colors.plotTick, dashed = true)),
                        title = "PORTFOLIO vs $benchmark — رشد تجمعی", height = 380.dp,
                    )
                }
            }
            item {
                val alphaCum = c.portfolioCumulative.indices.map { c.portfolioCumulative[it] - c.benchmarkCumulative[it] }
                Card(modifier = Modifier.fillMaxWidth()) {
                    LineChart(
                        series = listOf(LineSeries("Cumulative Alpha", if (alphaCum.last() >= 0) colors.green else colors.red, alphaCum, fillToZero = true)),
                        horizontalRefLines = listOf(RefLine(0.0, colors.plotTick)),
                        title = "CUMULATIVE ALPHA — پرتفو منهای بنچمارک", showLegend = false,
                    )
                }
            }
        }
    }
}
