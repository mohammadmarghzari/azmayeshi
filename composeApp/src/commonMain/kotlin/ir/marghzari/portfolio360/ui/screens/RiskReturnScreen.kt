package ir.marghzari.portfolio360.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import ir.marghzari.portfolio360.charts.BarChart
import ir.marghzari.portfolio360.charts.BarSeries
import ir.marghzari.portfolio360.charts.LineChart
import ir.marghzari.portfolio360.charts.LineSeries
import ir.marghzari.portfolio360.charts.RadarChart
import ir.marghzari.portfolio360.charts.RadarSeries
import ir.marghzari.portfolio360.charts.RefLine
import ir.marghzari.portfolio360.core.math.OptionsStrategies
import ir.marghzari.portfolio360.core.math.PortfolioEngine
import ir.marghzari.portfolio360.core.math.Stats
import ir.marghzari.portfolio360.state.AppState
import ir.marghzari.portfolio360.theme.LocalChartColors
import ir.marghzari.portfolio360.ui.components.Card
import ir.marghzari.portfolio360.ui.components.InfoBanner
import ir.marghzari.portfolio360.ui.components.MetricTile
import ir.marghzari.portfolio360.ui.components.SectionHeader
import kotlin.math.abs

@Composable
fun RiskReturnScreen(appState: AppState) {
    val colors = LocalChartColors.current
    val metrics = appState.metrics
    val weights = appState.weights
    val prices = appState.prices

    if (metrics == null || weights == null || prices == null) {
        InfoBanner("ابتدا پرتفوی را محاسبه کنید.")
        return
    }

    val returns = prices.dailyReturns()
    val hedged = appState.hedgedAssets
    val hedgedReturns = if (hedged.isNotEmpty()) {
        var r = returns
        hedged.values.forEach { h ->
            val idx = prices.tickers.indexOf(h.ticker)
            if (idx >= 0) {
                val col = OptionsStrategies.applyHedgeToColumn(DoubleArray(r.size) { r[it][idx] }, h.strike, h.spot, h.premium)
                r = Array(r.size) { i -> DoubleArray(r[i].size) { j -> if (j == idx) col[i] else r[i][j] } }
            }
        }
        r
    } else null
    val hedgedMetrics = hedgedReturns?.let { PortfolioEngine.portfolioMetrics(weights, it, appState.rf, appState.lastUsedRisk) }
    val displayMetrics = hedgedMetrics ?: metrics

    LazyColumn(verticalArrangement = Arrangement.spacedBy(16.dp), modifier = Modifier.fillMaxWidth()) {
        if (hedged.isNotEmpty()) {
            item {
                SectionHeader("🛡️ پرتفو با هج فعال — ${hedged.keys.joinToString("، ")}")
                Text("متریک‌های زیر با احتساب Protective Put محاسبه شده‌اند.", style = MaterialTheme.typography.bodySmall, color = colors.muted)
            }
        }

        item {
            LazyRow(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                items(
                    listOf(
                        Triple("بازده سالانه", "%.2f%%".format(displayMetrics.annualReturn * 100), null),
                        Triple("بازده تعدیل‌شده ریسک", "%.2f%%".format(displayMetrics.riskAdjustedReturn * 100), null),
                        Triple("نوسان سالانه", "%.2f%%".format(displayMetrics.annualVolatility * 100), null),
                        Triple("نسبت شارپ", "%.3f".format(displayMetrics.sharpeRatio), null),
                        Triple("حداکثر افت (MDD)", "%.2f%%".format(displayMetrics.maxDrawdown * 100), null),
                        Triple("CVaR 95%", "%.2f%%".format(displayMetrics.cvar95 * 100), null),
                        Triple("نسبت کالمار", "%.3f".format(displayMetrics.calmarRatio), null),
                        Triple("ریکاوری", if (displayMetrics.recoveryDays >= 30) "${displayMetrics.recoveryDays / 30} ماه" else "${displayMetrics.recoveryDays} روز", null),
                    ),
                ) { (label, value, _) -> MetricTile(label, value) }
            }
        }

        val risk = appState.lastUsedRisk
        if (risk.expectedReturnPct > 0 || risk.riskGeoPct > 0 || risk.riskMonPct > 0 || risk.riskSysPct > 0) {
            item {
                SectionHeader("Risk-Adjusted Analysis")
                LazyRow(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                    items(
                        listOf(
                            "تنزل کل ریسک‌ها" to "%.1f%%".format(metrics.riskDiscountPct),
                            "واگرایی از هدف بازده" to (metrics.returnGap?.let { "%+.2f%%".format(it * 100) } ?: "—"),
                            "نرخ بدون‌ریسک مؤثر" to "%.2f%%".format((appState.rf + PortfolioEngine.riskPenalty(risk)) * 100),
                            "تعداد نمادها" to prices.tickers.size.toString(),
                        ),
                    ) { (label, value) -> MetricTile(label, value) }
                }
            }
            item {
                SectionHeader("Risk Radar")
                Card(modifier = Modifier.fillMaxWidth()) {
                    RadarChart(
                        axisLabels = listOf("ریسک ژئوپولیتیک", "ریسک سیاست پولی", "ریسک سیستماتیک"),
                        series = listOf(RadarSeries("Risk", colors.red, listOf(risk.riskGeoPct, risk.riskMonPct, risk.riskSysPct))),
                        showLegend = false,
                    )
                }
            }
        }

        item {
            SectionHeader("Drawdown Chart")
            val portRet = Stats.portfolioReturns(hedgedReturns ?: returns, weights)
            val cum = Stats.cumulative(portRet)
            val dd = Stats.drawdownSeries(cum).map { it * 100 }
            Card(modifier = Modifier.fillMaxWidth()) {
                LineChart(
                    series = listOf(LineSeries("Drawdown", colors.red, dd, fillToZero = true)),
                    title = "DRAWDOWN (%)", showLegend = false,
                )
            }
        }

        item {
            SectionHeader("Underwater Chart — مدت زمان زیر Peak")
            val portRet = Stats.portfolioReturns(hedgedReturns ?: returns, weights)
            val cum = Stats.cumulative(portRet)
            val dd = Stats.drawdownSeries(cum).map { it * 100 }
            val maxStreak = Stats.longestUnderwaterStreak(dd.map { it / 100 }.toDoubleArray())
            Card(modifier = Modifier.fillMaxWidth()) {
                LineChart(
                    series = listOf(LineSeries("Underwater", colors.gold.copy(alpha = 0.9f), dd, fillToZero = true)),
                    horizontalRefLines = listOfNotNull(
                        if (dd.any { it < -10 }) RefLine(-10.0, colors.gold, "-10%") else null,
                        if (dd.any { it < -20 }) RefLine(-20.0, colors.red, "-20%") else null,
                    ),
                    title = "UNDERWATER CHART", showLegend = false,
                )
            }
            Row(horizontalArrangement = Arrangement.spacedBy(12.dp)) {
                MetricTile("بدترین افت", "%.2f%%".format(dd.minOrNull() ?: 0.0))
                MetricTile("طولانی‌ترین دوره زیر Peak", "$maxStreak روز")
            }
        }

        item {
            val portRet = Stats.portfolioReturns(hedgedReturns ?: returns, weights)
            val cum = Stats.cumulative(portRet)
            val expected = risk.expectedReturnPct
            val target = if (expected > 0) {
                val dailyTarget = Math.pow(1 + expected / 100.0, 1.0 / 252.0)
                var acc = 1.0
                cum.indices.map { acc *= dailyTarget; acc }
            } else null
            Card(modifier = Modifier.fillMaxWidth()) {
                LineChart(
                    series = listOfNotNull(
                        LineSeries("Portfolio", colors.blueAccent, cum.toList()),
                        target?.let { LineSeries("هدف ${expected.toInt()}%", colors.gold, it, dashed = true) },
                    ),
                    horizontalRefLines = listOf(RefLine(1.0, colors.plotTick, dashed = true)),
                    title = "PORTFOLIO GROWTH (BASE=1)",
                )
            }
        }

        item {
            SectionHeader("Daily Return Distribution")
            val portRet = Stats.portfolioReturns(hedgedReturns ?: returns, weights).map { it * 100 }
            if (portRet.isEmpty()) return@item
            val bins = 30
            val min = portRet.min(); val max = portRet.max()
            val width = ((max - min) / bins).takeIf { it > 0 } ?: 1.0
            val counts = DoubleArray(bins)
            portRet.forEach { v -> val idx = ((v - min) / width).toInt().coerceIn(0, bins - 1); counts[idx]++ }
            val labels = (0 until bins).map { "%.1f".format(min + width * it) }
            val cvarLine = Stats.percentile(portRet.toDoubleArray(), 5.0)
            Card(modifier = Modifier.fillMaxWidth()) {
                BarChart(
                    categories = labels, series = listOf(BarSeries("Frequency", colors.blueAccent, counts.toList())),
                    title = "DAILY RETURN DISTRIBUTION", showLegend = false, zeroLine = false,
                )
                Text("CVaR 95%%: %.2f%%".format(cvarLine), style = MaterialTheme.typography.labelSmall, color = colors.red)
            }
        }
    }
}
