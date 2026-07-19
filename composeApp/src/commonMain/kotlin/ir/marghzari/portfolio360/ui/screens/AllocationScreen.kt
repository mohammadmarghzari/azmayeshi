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
import ir.marghzari.portfolio360.ui.components.AssetRow
import ir.marghzari.portfolio360.ui.motion.StaggerIn
import ir.marghzari.portfolio360.ui.components.EmptyState
import ir.marghzari.portfolio360.ui.components.Card
import ir.marghzari.portfolio360.ui.components.HeroMetric
import ir.marghzari.portfolio360.ui.components.PortfolioSetupPanel
import ir.marghzari.portfolio360.ui.components.ScreenHeader
import ir.marghzari.portfolio360.ui.components.SectionHeader
import ir.marghzari.portfolio360.util.money
import ir.marghzari.portfolio360.util.pct

@Composable
fun AllocationScreen(appState: AppState) {
    val colors = LocalChartColors.current
    LazyColumn(modifier = Modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(16.dp)) {
        item { PortfolioSetupPanel(appState) }

        val weights = appState.weights
        val prices = appState.prices
        if (weights == null || prices == null) {
            item { EmptyState(title = "هنوز پرتفویی محاسبه نشده", hint = "از تب «تخصیص دارایی» داده را دریافت و پرتفوی را محاسبه کنید.") }
            return@LazyColumn
        }

        item {
            val sortedIdx = prices.tickers.indices.sortedByDescending { weights[it] }
            var totalCapital by remember { mutableStateOf(10000.0) }
            val topIdx = sortedIdx.first()

            ScreenHeader("تخصیص پرتفوی", "وزن بهینه هر دارایی بر اساس سبک «${appState.styleLabelUsed}»")
            Card(modifier = Modifier.fillMaxWidth(), highlighted = true) {
                HeroMetric(
                    label = "ارزش کل پرتفوی",
                    value = totalCapital.money(),
                    delta = "${prices.tickers.size} دارایی",
                )
                Text(
                    "بزرگ‌ترین موقعیت: ${prices.tickers[topIdx]} با وزن ${(weights[topIdx] * 100).pct(1)}",
                    style = MaterialTheme.typography.bodySmall, color = colors.muted,
                    modifier = Modifier.padding(top = 8.dp),
                )
            }

            SectionHeader("دارایی‌ها")
            Row(horizontalArrangement = Arrangement.spacedBy(16.dp), modifier = Modifier.fillMaxWidth()) {
                Card(modifier = Modifier.weight(1f)) {
                    sortedIdx.forEachIndexed { rank, i ->
                        val w = weights[i] * 100
                        StaggerIn(index = rank) {
                            AssetRow(
                                symbol = prices.tickers[i],
                                title = prices.tickers[i],
                                caption = "وزن ${w.pct(2)}",
                                value = (w / 100 * totalCapital).money(),
                            )
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
