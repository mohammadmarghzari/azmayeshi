package ir.marghzari.portfolio360.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
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
import kotlin.math.abs

@Composable
private fun NumberInput(label: String, value: Double, onChange: (Double) -> Unit, modifier: Modifier = Modifier) {
    var text by remember(value) { mutableStateOf(if (value == value.toLong().toDouble()) value.toLong().toString() else value.toString()) }
    Card(modifier = modifier) {
        Text(label, style = MaterialTheme.typography.labelSmall, color = LocalChartColors.current.muted)
        OutlinedTextField(
            value = text, onValueChange = { text = it; it.toDoubleOrNull()?.let(onChange) },
            modifier = Modifier.fillMaxWidth().padding(top = 4.dp), singleLine = true,
        )
    }
}

@Composable
fun IranToolsScreen(appState: AppState) {
    val colors = LocalChartColors.current
    var marketDollar by remember { mutableStateOf(90000.0) }

    // Method 1: inflation
    var baseDollar by remember { mutableStateOf(31000.0) }
    var iranInflation by remember { mutableStateOf(48.0) }
    var usInflation by remember { mutableStateOf(7.0) }

    // Method 2: gold
    var sekkePrice by remember { mutableStateOf(20_400_000.0) }
    var goldOz by remember { mutableStateOf(1866.0) }
    var goldAutoFetched by remember { mutableStateOf(false) }

    LaunchedEffect(Unit) {
        appState.worldCommodities.fetchGoldOz()?.let { goldOz = it; goldAutoFetched = true }
    }

    val inf = IranTools.fairValueByInflation(baseDollar, iranInflation, usInflation, marketDollar)
    val gold = IranTools.fairValueByGold(sekkePrice, goldOz, marketDollar)
    val avgReal = (inf.fairDollarToman + gold.fairDollarToman) / 2

    LazyColumn(verticalArrangement = Arrangement.spacedBy(16.dp), modifier = Modifier.fillMaxWidth()) {
        item {
            SectionHeader("🇮🇷 ابزار ایران — نرخ واقعی دلار")
            NumberInput("قیمت دلار بازار آزاد (تومان)", marketDollar, { marketDollar = it })
        }

        item {
            SectionHeader("روش ۱ — محاسبه به کمک تورم")
            Text("قیمت پایه دلار × (۱ + اختلاف تورم ایران و آمریکا) = قیمت واقعی تخمینی", style = MaterialTheme.typography.bodySmall, color = colors.muted)
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.padding(top = 8.dp)) {
                NumberInput("قیمت پایه دلار", baseDollar, { baseDollar = it }, modifier = Modifier.weight(1f))
                NumberInput("تورم ایران (%)", iranInflation, { iranInflation = it }, modifier = Modifier.weight(1f))
                NumberInput("تورم آمریکا (%)", usInflation, { usInflation = it }, modifier = Modifier.weight(1f))
            }
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.padding(top = 10.dp)) {
                MetricTile("اختلاف تورم", "%.1f%%".format(inf.inflationDiffPct))
                MetricTile("قیمت واقعی تخمینی", "%,.0f تومان".format(inf.fairDollarToman))
                MetricTile("فاصله قیمتی", "%,.0f تومان (%.1f%%)".format(inf.gapToman, inf.gapPct), valueColor = if (inf.gapToman >= 0) colors.green else colors.red)
            }
        }

        item {
            SectionHeader("روش ۲ — محاسبه به کمک قیمت طلا")
            Text("قیمت سکه بهار آزادی ÷ (انس جهانی × ۴) = قیمت واقعی دلار", style = MaterialTheme.typography.bodySmall, color = colors.muted)
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.padding(top = 8.dp)) {
                NumberInput("قیمت سکه بهار آزادی (تومان)", sekkePrice, { sekkePrice = it }, modifier = Modifier.weight(1f))
                NumberInput("قیمت انس جهانی طلا ($)", goldOz, { goldOz = it }, modifier = Modifier.weight(1f))
            }
            if (goldAutoFetched) Text("✓ قیمت خودکار از Yahoo Finance دریافت شد", style = MaterialTheme.typography.labelSmall, color = colors.green, modifier = Modifier.padding(top = 4.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.padding(top = 10.dp)) {
                MetricTile("قیمت واقعی تخمینی", "%,.0f تومان".format(gold.fairDollarToman))
                MetricTile("فاصله قیمتی", "%,.0f تومان (%.1f%%)".format(gold.gapToman, gold.gapPct), valueColor = if (gold.gapToman >= 0) colors.green else colors.red)
            }
        }

        item {
            Card(modifier = Modifier.fillMaxWidth()) {
                BarChart(
                    categories = listOf("بازار آزاد", "روش تورم", "روش طلا", "میانگین"),
                    series = listOf(BarSeries("قیمت", colors.blueAccent, listOf(marketDollar, inf.fairDollarToman, gold.fairDollarToman, avgReal))),
                    title = "مقایسه قیمت بازار با قیمت واقعی", showLegend = false,
                )
                val gapAvgPct = (avgReal - marketDollar) / marketDollar * 100
                Text(
                    if (gapAvgPct > 0) "دلار در بازار ارزان‌تر از ارزش واقعی تخمینی معامله می‌شود (%.1f%%)".format(gapAvgPct)
                    else "دلار در بازار گران‌تر از ارزش واقعی تخمینی معامله می‌شود (%.1f%%)".format(-gapAvgPct),
                    style = MaterialTheme.typography.bodySmall, color = colors.gold, modifier = Modifier.padding(top = 8.dp),
                )
                Text("⚠ این محاسبات تخمینی هستند و برای تصمیم‌گیری مالی باید با سایر عوامل ترکیب شوند.", style = MaterialTheme.typography.labelSmall, color = colors.muted, modifier = Modifier.padding(top = 6.dp))
            }
        }

        item { SectionHeader("🏭 گواهی سپرده کالایی — بورس کالای ایران") }
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
            Card(modifier = Modifier.fillMaxWidth().padding(top = 10.dp)) {
                bubbles.forEach { (c, b) ->
                    Row(modifier = Modifier.fillMaxWidth().padding(vertical = 3.dp), horizontalArrangement = Arrangement.SpaceBetween) {
                        Text("${c.icon} ${c.labelFa}", color = colors.textPrimary, style = MaterialTheme.typography.bodySmall)
                        Text(b.fairPriceToman?.let { "%,.0f".format(it) } ?: "—", color = colors.muted, style = MaterialTheme.typography.bodySmall)
                        Text(
                            b.bubblePct?.let { "%+.1f%%".format(it) } ?: "بدون مرجع",
                            color = b.bubblePct?.let { if (it > 5) colors.red else if (it < -5) colors.green else colors.gold } ?: colors.muted,
                            style = MaterialTheme.typography.bodySmall,
                        )
                    }
                }
                val validBubbles = bubbles.mapNotNull { (c, b) -> b.bubblePct?.let { c.labelFa to it } }
                if (validBubbles.isNotEmpty()) {
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

            ir.marghzari.portfolio360.ui.components.SimpleDropdown(
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
