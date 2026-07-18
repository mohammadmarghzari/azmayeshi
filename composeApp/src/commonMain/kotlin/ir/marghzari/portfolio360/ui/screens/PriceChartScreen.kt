package ir.marghzari.portfolio360.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.RadioButton
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import ir.marghzari.portfolio360.charts.HeatmapChart
import ir.marghzari.portfolio360.charts.LineChart
import ir.marghzari.portfolio360.charts.LineSeries
import ir.marghzari.portfolio360.core.math.Stats
import ir.marghzari.portfolio360.state.AppState
import ir.marghzari.portfolio360.theme.LocalChartColors
import ir.marghzari.portfolio360.theme.chartColor
import ir.marghzari.portfolio360.ui.components.Card
import ir.marghzari.portfolio360.ui.components.InfoBanner
import ir.marghzari.portfolio360.ui.components.SectionHeader

@Composable
fun PriceChartScreen(appState: AppState) {
    val colors = LocalChartColors.current
    val prices = appState.prices ?: run { InfoBanner("ابتدا پرتفوی را محاسبه کنید."); return }

    var normalized by remember { mutableStateOf(true) }

    LazyColumn(verticalArrangement = Arrangement.spacedBy(16.dp), modifier = Modifier.fillMaxWidth()) {
        item {
            Row(verticalAlignment = androidx.compose.ui.Alignment.CenterVertically) {
                RadioButton(selected = normalized, onClick = { normalized = true })
                Text("نرمال‌شده (base=100)", modifier = Modifier.padding(end = 12.dp))
                RadioButton(selected = !normalized, onClick = { normalized = false })
                Text("قیمت خام")
            }
        }
        item {
            val series = prices.tickers.mapIndexed { i, ticker ->
                val col = prices.column(ticker)
                val values = if (normalized) col.map { it / col[0] * 100 } else col.toList()
                LineSeries(ticker, chartColor(i), values)
            }
            Card(modifier = Modifier.fillMaxWidth()) {
                LineChart(
                    series = series, title = "PRICE CHART",
                    yFormatter = { if (normalized) "%.0f".format(it) else "$%.2f".format(it) },
                    height = 420.dp,
                )
            }
        }

        if (prices.nAssets >= 2) {
            item {
                SectionHeader("Correlation Matrix")
                val returns = prices.dailyReturns()
                val n = prices.nAssets
                val corr = Array(n) { i -> DoubleArray(n) { j ->
                    Stats.correlation(DoubleArray(returns.size) { returns[it][i] }, DoubleArray(returns.size) { returns[it][j] })
                } }
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
