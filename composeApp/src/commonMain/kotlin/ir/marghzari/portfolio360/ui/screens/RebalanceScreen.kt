package ir.marghzari.portfolio360.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
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
import ir.marghzari.portfolio360.core.math.CorrelationRegime
import ir.marghzari.portfolio360.core.math.CorrelationRegimeResult
import ir.marghzari.portfolio360.core.math.Rebalancing
import ir.marghzari.portfolio360.state.AppState
import ir.marghzari.portfolio360.theme.LocalChartColors
import ir.marghzari.portfolio360.ui.components.Card
import ir.marghzari.portfolio360.ui.components.InfoBanner
import ir.marghzari.portfolio360.ui.components.SectionHeader
import ir.marghzari.portfolio360.ui.components.VerdictCard
import ir.marghzari.portfolio360.ui.components.VerdictTone

@Composable
fun RebalanceScreen(appState: AppState) {
    val colors = LocalChartColors.current
    val weights = appState.weights
    val prices = appState.prices
    if (weights == null || prices == null) {
        InfoBanner("ابتدا پرتفوی را محاسبه کنید.")
        return
    }

    var capital by remember { mutableStateOf(10000.0) }
    var thresholdPct by remember { mutableStateOf(5f) }
    val lastPrices = DoubleArray(prices.nAssets) { prices.values.last()[it] }
    val rows = remember(weights, thresholdPct, capital) {
        Rebalancing.compute(prices.tickers, lastPrices, weights, capital, thresholdPct / 100.0)
    }
    val needsRebalance = rows.count { it.needsRebalance }

    var regime by remember { mutableStateOf<CorrelationRegimeResult?>(null) }

    LazyColumn(verticalArrangement = Arrangement.spacedBy(16.dp), modifier = Modifier.fillMaxWidth()) {
        item {
            SectionHeader("Rebalancing Alert — نیاز به ری‌بالانس")
            Text("ارزش فعلی پرتفو: $%,.0f".format(capital), style = MaterialTheme.typography.labelMedium)
            Slider(value = capital.toFloat(), onValueChange = { capital = it.toDouble() }, valueRange = 1000f..1_000_000f)
            Text("آستانه ری‌بالانس: ${thresholdPct.toInt()}%", style = MaterialTheme.typography.labelMedium)
            Slider(value = thresholdPct, onValueChange = { thresholdPct = it }, valueRange = 1f..20f, steps = 18)
        }
        item {
            if (needsRebalance > 0) {
                VerdictCard("⚠ $needsRebalance نماد نیاز به ری‌بالانس دارد", "انحراف بیش از ${thresholdPct.toInt()}٪ از وزن هدف", VerdictTone.NEGATIVE)
            } else {
                VerdictCard("✅ همه نمادها در محدوده هدف هستند", "انحراف زیر ${thresholdPct.toInt()}٪", VerdictTone.POSITIVE)
            }
        }
        item {
            Card(modifier = Modifier.fillMaxWidth()) {
                rows.forEach { r ->
                    Row(modifier = Modifier.fillMaxWidth().padding(vertical = 3.dp), horizontalArrangement = Arrangement.SpaceBetween) {
                        Text(r.ticker, color = colors.textPrimary, style = MaterialTheme.typography.bodySmall)
                        Text("هدف %.1f%%".format(r.targetWeightPct), color = colors.textPrimary, style = MaterialTheme.typography.bodySmall)
                        Text("فعلی %.1f%%".format(r.currentWeightPct), color = colors.textPrimary, style = MaterialTheme.typography.bodySmall)
                        Text("معامله $%,.0f".format(r.tradeDollar), color = colors.blueAccent, style = MaterialTheme.typography.bodySmall)
                        Text(if (r.needsRebalance) "⚠" else "✓", color = if (r.needsRebalance) colors.red else colors.green)
                    }
                }
                BarChart(
                    categories = rows.map { it.ticker },
                    series = listOf(BarSeries("وزن هدف", colors.blueAccent, rows.map { it.targetWeightPct }), BarSeries("وزن فعلی", colors.gold, rows.map { it.currentWeightPct })),
                    title = "REBALANCING — وزن هدف vs فعلی",
                )
            }
        }

        if (prices.nAssets >= 2) {
            item {
                SectionHeader("Correlation Regime — تشخیص رژیم بازار")
                Text("اگه همبستگی‌ها ناگهان افزایش یابد، معمولاً نشانه استرس بازار یا بحران است.", style = MaterialTheme.typography.bodySmall, color = colors.muted)
                Button(
                    onClick = { regime = CorrelationRegime.detect(prices.dates.drop(1), prices.dailyReturns(), prices.nAssets) },
                    modifier = Modifier.fillMaxWidth().padding(top = 10.dp), colors = ButtonDefaults.buttonColors(containerColor = colors.blueAccent),
                ) { Text("تشخیص رژیم همبستگی") }
            }
            regime?.let { r ->
                item {
                    VerdictCard("رژیم فعلی بازار", r.regimeLabelFa, if (r.regimeLabelFa.startsWith("🔴")) VerdictTone.NEGATIVE else VerdictTone.POSITIVE)
                }
                if (r.points.isNotEmpty()) {
                    item {
                        Card(modifier = Modifier.fillMaxWidth()) {
                            LineChart(
                                series = listOf(
                                    LineSeries("همبستگی ۳۰ روزه", colors.red, r.points.map { it.corrShort }),
                                    LineSeries("همبستگی ۱۲۶ روزه", colors.blueAccent, r.points.map { it.corrLong }, dashed = true),
                                ),
                                title = "CORRELATION REGIME DETECTION",
                            )
                        }
                    }
                }
            }
        }
    }
}
