package ir.marghzari.portfolio360.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import ir.marghzari.portfolio360.charts.DonutChart
import ir.marghzari.portfolio360.charts.PieSlice
import ir.marghzari.portfolio360.charts.TreemapChart
import ir.marghzari.portfolio360.charts.TreemapItem
import ir.marghzari.portfolio360.state.AppState
import ir.marghzari.portfolio360.theme.LocalChartColors
import ir.marghzari.portfolio360.theme.chartColor
import ir.marghzari.portfolio360.ui.components.Card
import ir.marghzari.portfolio360.ui.components.InfoBanner
import ir.marghzari.portfolio360.ui.components.PortfolioSetupPanel
import ir.marghzari.portfolio360.ui.components.SectionHeader

@Composable
fun AllocationScreen(appState: AppState) {
    val colors = LocalChartColors.current
    LazyColumn(modifier = Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(16.dp)) {
        item { PortfolioSetupPanel(appState) }

        val weights = appState.weights
        val prices = appState.prices
        if (weights == null || prices == null) {
            item { InfoBanner("ابتدا پرتفوی را محاسبه کنید.") }
            return@LazyColumn
        }

        item {
            SectionHeader("تخصیص پرتفوی")
            val sortedIdx = prices.tickers.indices.sortedByDescending { weights[it] }
            var totalCapital by remember { mutableStateOf(10000.0) }

            Row(horizontalArrangement = Arrangement.spacedBy(16.dp), modifier = Modifier.fillMaxWidth()) {
                Card(modifier = Modifier.weight(1f)) {
                    Text("نماد   وزن (%)   مبلغ ($)", style = MaterialTheme.typography.labelMedium, color = colors.muted)
                    sortedIdx.forEach { i ->
                        val w = weights[i] * 100
                        Row(modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp), horizontalArrangement = Arrangement.SpaceBetween) {
                            Text(prices.tickers[i], style = MaterialTheme.typography.bodyMedium, color = colors.textPrimary)
                            Text("%.2f%%".format(w), style = MaterialTheme.typography.bodyMedium, color = colors.textPrimary)
                            Text("$%,.2f".format(w / 100 * totalCapital), style = MaterialTheme.typography.bodyMedium, color = colors.blueAccent)
                        }
                    }
                }
                Card(modifier = Modifier.weight(1f)) {
                    DonutChart(
                        slices = sortedIdx.mapIndexed { rank, i -> PieSlice(prices.tickers[i], weights[i] * 100, chartColor(rank)) },
                        title = "ALLOCATION — ${appState.styleLabelUsed}",
                    )
                }
            }
        }

        item {
            SectionHeader("Treemap — نمایش بصری وزن‌ها")
            Card(modifier = Modifier.fillMaxWidth()) {
                val sortedIdx = prices.tickers.indices.sortedByDescending { weights[it] }
                TreemapChart(
                    items = sortedIdx.mapIndexed { rank, i -> TreemapItem(prices.tickers[i], weights[i] * 100, chartColor(rank)) },
                    title = "TREEMAP — PORTFOLIO WEIGHTS",
                )
            }
        }
    }
}
