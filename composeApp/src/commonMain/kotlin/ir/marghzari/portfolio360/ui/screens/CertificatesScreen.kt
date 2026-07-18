package ir.marghzari.portfolio360.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
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
import ir.marghzari.portfolio360.core.math.IranTools
import ir.marghzari.portfolio360.state.AppState
import ir.marghzari.portfolio360.theme.LocalChartColors
import ir.marghzari.portfolio360.ui.components.Card
import ir.marghzari.portfolio360.ui.components.MetricTile
import ir.marghzari.portfolio360.ui.components.SectionHeader
import ir.marghzari.portfolio360.ui.components.SimpleDropdown
import ir.marghzari.portfolio360.ui.motion.motionColorsFor

/** Split out of the Iran tools screen into its own tab so it's a separate, focused partition. */
@Composable
fun CertificatesScreen(appState: AppState) {
    val colors = LocalChartColors.current
    var marketDollar by remember { mutableStateOf(90000.0) }

    LazyColumn(verticalArrangement = Arrangement.spacedBy(16.dp), modifier = Modifier.fillMaxWidth()) {
        item {
            SectionHeader("🏭 گواهی سپرده کالایی — بورس کالای ایران")
            NumberInput("قیمت دلار بازار آزاد (تومان)", marketDollar, { marketDollar = it })
        }
        item {
            var world by remember { mutableStateOf<IranTools.WorldPrices?>(null) }
            LaunchedEffect(Unit) { world = appState.worldCommodities.fetch() }
            val effectiveWorld = world ?: IranTools.WorldPrices(2350.0, 30.0, 4.5 * 2.20462, 2800.0 / 1000.0)
            val marketPrices = remember { IranTools.IRAN_COMMODITIES.associate { it.key to mutableStateOf(it.defaultPriceToman.toDouble()) } }

            IranTools.IRAN_COMMODITIES.forEach { c ->
                val state = marketPrices.getValue(c.key)
                NumberInput("${c.icon} ${c.labelFa} (${c.unitFa})", state.value, { state.value = it }, modifier = Modifier.padding(top = 6.dp))
            }

            val bubbles = IranTools.IRAN_COMMODITIES.map { c ->
                val fair = IranTools.fairPrice(c.key, effectiveWorld, marketDollar)
                c to IranTools.bubble(marketPrices.getValue(c.key).value, fair)
            }
            bubbles.forEach { (c, b) ->
                Card(modifier = Modifier.fillMaxWidth().padding(top = 6.dp), motionColors = motionColorsFor(c.key)) {
                    Row(modifier = Modifier.fillMaxWidth(), horizontalArrangement = Arrangement.SpaceBetween) {
                        Text("${c.icon} ${c.labelFa}", color = colors.textPrimary, style = MaterialTheme.typography.bodySmall)
                        Text(b.fairPriceToman?.let { "%,.0f".format(it) } ?: "—", color = colors.muted, style = MaterialTheme.typography.bodySmall)
                        Text(
                            b.bubblePct?.let { "%+.1f%%".format(it) } ?: "بدون مرجع",
                            color = b.bubblePct?.let { if (it > 5) colors.red else if (it < -5) colors.green else colors.gold } ?: colors.muted,
                            style = MaterialTheme.typography.bodySmall,
                        )
                    }
                }
            }
            val validBubbles = bubbles.mapNotNull { (c, b) -> b.bubblePct?.let { c.labelFa to it } }
            if (validBubbles.isNotEmpty()) {
                Card(modifier = Modifier.fillMaxWidth().padding(top = 10.dp)) {
                    BarChart(
                        categories = validBubbles.map { it.first },
                        series = listOf(BarSeries("حباب (%)", colors.gold, validBubbles.map { it.second })),
                        title = "حباب گواهی‌های سپرده بورس کالا", showLegend = false,
                        perBarColorOverride = { _, i, v -> if (v > 5) colors.red else if (v < -5) colors.green else colors.gold },
                    )
                }
            }

            SectionHeader("📊 نمودار سود و زیان — قیمت تارگت", modifier = Modifier.padding(top = 16.dp))
            Text("قیمت خرید، تعداد و تارگت را برای یک گواهی وارد کنید.", style = MaterialTheme.typography.bodySmall, color = colors.muted)
            var pnlKey by remember { mutableStateOf(IranTools.IRAN_COMMODITIES.first().key) }
            val pnlCommodity = IranTools.IRAN_COMMODITIES.first { it.key == pnlKey }
            val currentPrice = marketPrices.getValue(pnlKey).value
            var buyPrice by remember(pnlKey) { mutableStateOf(currentPrice * 0.95) }
            var qty by remember(pnlKey) { mutableStateOf(10.0) }
            var targetPrice by remember(pnlKey) { mutableStateOf(currentPrice * 1.20) }
            var stopPrice by remember(pnlKey) { mutableStateOf(currentPrice * 0.85) }

            SimpleDropdown(
                "گواهی", pnlCommodity, IranTools.IRAN_COMMODITIES, { "${it.icon} ${it.labelFa}" }, { pnlKey = it.key },
                modifier = Modifier.padding(top = 10.dp),
            )
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.padding(top = 8.dp)) {
                NumberInput("قیمت خرید (تومان)", buyPrice, { buyPrice = it }, modifier = Modifier.weight(1f))
                NumberInput("تعداد (${pnlCommodity.unitFa})", qty, { qty = it }, modifier = Modifier.weight(1f))
            }
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.padding(top = 8.dp)) {
                NumberInput("قیمت تارگت (تومان)", targetPrice, { targetPrice = it }, modifier = Modifier.weight(1f))
                NumberInput("حد ضرر (تومان)", stopPrice, { stopPrice = it }, modifier = Modifier.weight(1f))
            }

            val pnl = IranTools.pnlTarget(buyPrice, qty, currentPrice, targetPrice, stopPrice)
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.padding(top = 10.dp)) {
                MetricTile("P&L فعلی", "%,.0f (%.1f%%)".format(pnl.pnlCurrentToman, pnl.pnlCurrentPct))
                MetricTile("سود در تارگت", "%,.0f (%.1f%%)".format(pnl.pnlTargetToman, pnl.pnlTargetPct), valueColor = colors.green)
                MetricTile("زیان در استاپ", "%,.0f (%.1f%%)".format(pnl.pnlStopToman, pnl.pnlStopPct), valueColor = colors.red)
                MetricTile("R/R", if (pnl.riskRewardRatio.isFinite()) "%.2f".format(pnl.riskRewardRatio) else "∞")
            }
            Card(modifier = Modifier.fillMaxWidth().padding(top = 10.dp)) {
                val lo = minOf(stopPrice, buyPrice) * 0.9
                val hi = targetPrice * 1.1
                val range = (0 until 300).map { lo + (hi - lo) * it / 299.0 }
                val pnlLine = range.map { (it - buyPrice) * qty }
                LineChart(
                    series = listOf(LineSeries("P&L", colors.blueAccent, pnlLine, fillToZero = true)),
                    xValues = range,
                    verticalRefLines = listOf(
                        RefLine(buyPrice, colors.gold, "خرید"), RefLine(currentPrice, colors.blueAccent, "فعلی"),
                        RefLine(targetPrice, colors.green, "تارگت"), RefLine(stopPrice, colors.red, "استاپ"),
                    ),
                    horizontalRefLines = listOf(RefLine(0.0, colors.plotTick)),
                    title = "P&L — ${pnlCommodity.icon} ${pnlCommodity.labelFa}", showLegend = false,
                )
            }
        }
    }
}
