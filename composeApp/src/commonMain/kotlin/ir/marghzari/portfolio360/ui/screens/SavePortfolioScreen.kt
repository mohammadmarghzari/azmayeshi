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
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
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
import ir.marghzari.portfolio360.charts.DonutChart
import ir.marghzari.portfolio360.charts.PieSlice
import ir.marghzari.portfolio360.state.AppState
import ir.marghzari.portfolio360.state.SavedPortfolio
import ir.marghzari.portfolio360.theme.LocalChartColors
import ir.marghzari.portfolio360.theme.chartColor
import ir.marghzari.portfolio360.ui.components.Card
import ir.marghzari.portfolio360.ui.components.InfoBanner
import ir.marghzari.portfolio360.ui.components.MetricTile
import ir.marghzari.portfolio360.ui.components.SectionHeader

@Composable
fun SavePortfolioScreen(appState: AppState) {
    val colors = LocalChartColors.current
    val weights = appState.weights
    val metrics = appState.metrics
    val prices = appState.prices

    LazyColumn(verticalArrangement = Arrangement.spacedBy(16.dp), modifier = Modifier.fillMaxWidth()) {
        item {
            SectionHeader("💾 ذخیره پرتفوی فعلی")
            if (weights != null && metrics != null && prices != null) {
                var name by remember { mutableStateOf("") }
                var message by remember { mutableStateOf<String?>(null) }
                OutlinedTextField(
                    value = name, onValueChange = { name = it }, placeholder = { Text("مثلاً: پرتفوی محافظه‌کار ۱۴۰۳") },
                    modifier = Modifier.fillMaxWidth(),
                )
                Button(
                    onClick = {
                        if (name.isBlank()) {
                            message = "یک نام برای پرتفوی وارد کنید."
                        } else {
                            appState.savedPortfolios = appState.savedPortfolios + (name.trim() to SavedPortfolio(
                                name.trim(), weights, prices.tickers, metrics, appState.styleLabelUsed, appState.lastUsedRisk,
                                "ذخیره‌شده",
                            ))
                            message = "پرتفوی «${name.trim()}» ذخیره شد."
                            name = ""
                        }
                    },
                    modifier = Modifier.fillMaxWidth().padding(top = 8.dp), colors = ButtonDefaults.buttonColors(containerColor = colors.blueAccent),
                ) { Text("💾 ذخیره") }
                message?.let { Text(it, color = colors.green, style = MaterialTheme.typography.labelSmall, modifier = Modifier.padding(top = 6.dp)) }
            } else {
                InfoBanner("ابتدا پرتفوی را محاسبه کنید تا بتوانید آن را ذخیره کنید.")
            }
        }

        item { SectionHeader("پرتفوی‌های ذخیره‌شده") }
        if (appState.savedPortfolios.isEmpty()) {
            item { InfoBanner("هنوز پرتفویی ذخیره نشده.") }
        } else {
            items(appState.savedPortfolios.values.toList()) { sp ->
                Card(modifier = Modifier.fillMaxWidth()) {
                    Text("📁 ${sp.name} · ${sp.style}", style = MaterialTheme.typography.titleSmall, color = colors.textPrimary)
                    LazyRow(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.padding(top = 8.dp)) {
                        items(
                            listOf(
                                "بازده تعدیل‌شده" to "%.2f%%".format(sp.metrics.riskAdjustedReturn * 100),
                                "نوسان" to "%.2f%%".format(sp.metrics.annualVolatility * 100),
                                "شارپ" to "%.3f".format(sp.metrics.sharpeRatio),
                                "MDD" to "%.2f%%".format(sp.metrics.maxDrawdown * 100),
                            ),
                        ) { (l, v) -> MetricTile(l, v) }
                    }
                    val sortedIdx = sp.tickers.indices.sortedByDescending { sp.weights[it] }
                    DonutChart(
                        slices = sortedIdx.mapIndexed { rank, i -> PieSlice(sp.tickers[i], sp.weights[i] * 100, chartColor(rank)) },
                        modifier = Modifier.padding(top = 10.dp),
                    )
                    Button(
                        onClick = { appState.savedPortfolios = appState.savedPortfolios - sp.name },
                        colors = ButtonDefaults.buttonColors(containerColor = colors.red), modifier = Modifier.padding(top = 8.dp),
                    ) { Text("🗑 حذف «${sp.name}»") }
                }
            }
        }

        if (appState.savedPortfolios.size >= 2) {
            item {
                SectionHeader("مقایسه پرتفوی‌های ذخیره‌شده")
                val list = appState.savedPortfolios.values.toList()
                Card(modifier = Modifier.fillMaxWidth()) {
                    BarChart(
                        categories = list.map { it.name },
                        series = listOf(
                            BarSeries("بازده تعدیل‌شده (%)", chartColor(0), list.map { it.metrics.riskAdjustedReturn * 100 }),
                            BarSeries("نوسان (%)", chartColor(1), list.map { it.metrics.annualVolatility * 100 }),
                            BarSeries("شارپ", chartColor(2), list.map { it.metrics.sharpeRatio }),
                            BarSeries("MDD (%)", chartColor(3), list.map { it.metrics.maxDrawdown * 100 }),
                        ),
                        title = "PORTFOLIO COMPARISON",
                    )
                }
            }
        }
    }
}
