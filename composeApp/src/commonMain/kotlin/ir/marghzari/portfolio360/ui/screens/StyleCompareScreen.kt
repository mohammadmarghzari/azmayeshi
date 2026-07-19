package ir.marghzari.portfolio360.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
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
import ir.marghzari.portfolio360.charts.BarChart
import ir.marghzari.portfolio360.charts.BarSeries
import ir.marghzari.portfolio360.charts.LineChart
import ir.marghzari.portfolio360.charts.LineSeries
import ir.marghzari.portfolio360.core.math.PortfolioEngine
import ir.marghzari.portfolio360.core.math.Stats
import ir.marghzari.portfolio360.core.model.PortfolioMetrics
import ir.marghzari.portfolio360.core.model.PortfolioStyle
import ir.marghzari.portfolio360.state.AppState
import ir.marghzari.portfolio360.theme.LocalChartColors
import ir.marghzari.portfolio360.theme.chartColor
import ir.marghzari.portfolio360.ui.components.EmptyState
import ir.marghzari.portfolio360.ui.components.Card
import ir.marghzari.portfolio360.ui.components.SectionHeader
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

private data class StyleResult(val style: PortfolioStyle, val metrics: PortfolioMetrics, val weights: DoubleArray)

@Composable
fun StyleCompareScreen(appState: AppState) {
    val colors = LocalChartColors.current
    val prices = appState.prices
    if (prices == null || prices.nAssets < 2) {
        EmptyState(title = "حداقل ۲ نماد نیاز است", hint = "برای مقایسه سبک‌ها، پرتفوی را با دست‌کم دو نماد بسازید.")
        return
    }
    val scope = rememberCoroutineScope()
    var results by remember { mutableStateOf<List<StyleResult>>(emptyList()) }
    var running by remember { mutableStateOf(false) }

    LazyColumn(verticalArrangement = Arrangement.spacedBy(16.dp), modifier = Modifier.fillMaxWidth()) {
        item {
            Button(
                onClick = {
                    scope.launch {
                        running = true
                        results = withContext(Dispatchers.Default) {
                            val returns = prices.dailyReturns()
                            PortfolioStyle.entries.mapNotNull { style ->
                                try {
                                    val w = PortfolioEngine.calcWeights(style, returns, appState.rf, prices.tickers, appState.riskInputs)
                                    val m = PortfolioEngine.portfolioMetrics(w.weights, returns, appState.rf, appState.riskInputs)
                                    StyleResult(style, m, w.weights)
                                } catch (e: Exception) { null }
                            }
                        }
                        running = false
                    }
                },
                enabled = !running,
                modifier = Modifier.fillMaxWidth(),
                colors = ButtonDefaults.buttonColors(containerColor = colors.blueAccent),
            ) {
                if (running) CircularProgressIndicator(modifier = Modifier.height(16.dp), color = MaterialTheme.colorScheme.onPrimary, strokeWidth = 2.dp)
                else Text("▶ اجرای مقایسه همه سبک‌ها")
            }
        }

        if (results.isNotEmpty()) {
            item {
                Card(modifier = Modifier.fillMaxWidth()) {
                    Text("سبک | بازده تعدیل‌شده% | نوسان% | شارپ | MDD%", style = MaterialTheme.typography.labelSmall, color = colors.muted)
                    results.forEach { r ->
                        Row(modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp), horizontalArrangement = Arrangement.SpaceBetween) {
                            Text(r.style.faLabel, style = MaterialTheme.typography.bodySmall, color = colors.textPrimary)
                            Text("%.2f".format(r.metrics.riskAdjustedReturn * 100), style = MaterialTheme.typography.bodySmall)
                            Text("%.2f".format(r.metrics.annualVolatility * 100), style = MaterialTheme.typography.bodySmall)
                            Text("%.3f".format(r.metrics.sharpeRatio), style = MaterialTheme.typography.bodySmall)
                            Text("%.2f".format(r.metrics.maxDrawdown * 100), style = MaterialTheme.typography.bodySmall)
                        }
                    }
                }
            }
            item {
                Card(modifier = Modifier.fillMaxWidth()) {
                    BarChart(
                        categories = results.map { it.style.faLabel },
                        series = listOf(
                            BarSeries("بازده تعدیل‌شده (%)", chartColor(0), results.map { it.metrics.riskAdjustedReturn * 100 }),
                            BarSeries("نوسان (%)", chartColor(1), results.map { it.metrics.annualVolatility * 100 }),
                            BarSeries("شارپ", chartColor(2), results.map { it.metrics.sharpeRatio }),
                            BarSeries("MDD (%)", chartColor(3), results.map { it.metrics.maxDrawdown * 100 }),
                        ),
                        title = "STRATEGY COMPARISON (RISK-ADJUSTED)",
                    )
                }
            }
            item {
                SectionHeader("Growth Curves — All Strategies")
                val returns = prices.dailyReturns()
                Card(modifier = Modifier.fillMaxWidth()) {
                    LineChart(
                        series = results.mapIndexed { i, r ->
                            val portRet = Stats.portfolioReturns(returns, r.weights)
                            LineSeries(r.style.faLabel, chartColor(i), Stats.cumulative(portRet).toList())
                        },
                        title = "GROWTH CURVES", height = 380.dp,
                    )
                }
            }
        }
    }
}
