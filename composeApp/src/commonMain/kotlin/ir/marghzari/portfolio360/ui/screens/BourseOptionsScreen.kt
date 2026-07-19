package ir.marghzari.portfolio360.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.LazyRow
import androidx.compose.foundation.lazy.items
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.RadioButton
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
import ir.marghzari.portfolio360.charts.RefLine
import ir.marghzari.portfolio360.core.math.BlackScholes
import ir.marghzari.portfolio360.core.math.BourseStrategies
import ir.marghzari.portfolio360.core.math.OptionType
import ir.marghzari.portfolio360.core.math.OptionsStrategies
import ir.marghzari.portfolio360.state.AppState
import ir.marghzari.portfolio360.theme.LocalChartColors
import ir.marghzari.portfolio360.ui.components.ScreenHeader
import ir.marghzari.portfolio360.ui.components.AppButton
import ir.marghzari.portfolio360.ui.components.Card
import ir.marghzari.portfolio360.ui.components.MetricTile
import ir.marghzari.portfolio360.ui.components.SectionHeader
import ir.marghzari.portfolio360.ui.components.SimpleDropdown
import kotlin.math.abs
import kotlin.math.max
import kotlin.math.min

@Composable
private fun NumField(label: String, value: Double, onChange: (Double) -> Unit, modifier: Modifier = Modifier) {
    val colors = LocalChartColors.current
    var text by remember(value) { mutableStateOf(if (value == value.toLong().toDouble()) value.toLong().toString() else value.toString()) }
    Card(modifier = modifier) {
        Text(label, style = MaterialTheme.typography.labelSmall, color = colors.muted)
        OutlinedTextField(value = text, onValueChange = { text = it; it.toDoubleOrNull()?.let(onChange) }, modifier = Modifier.fillMaxWidth().padding(top = 4.dp), singleLine = true)
    }
}

private enum class BourseStrategyType(val label: String) {
    BULL_CALL("Bull Call Spread"), BEAR_PUT("Bear Put Spread"), STRADDLE("Long Straddle"),
    STRANGLE("Long Strangle"), COVERED_CALL("Covered Call (با دارایی پایه)"), PROTECTIVE_PUT("Protective Put (بیمه دارایی)"),
}

@Composable
fun BourseOptionsScreen(appState: AppState) {
    val colors = LocalChartColors.current
    var underlying by remember { mutableStateOf(BourseStrategies.UNDERLYINGS.first()) }
    var spot by remember { mutableStateOf(700000.0) }
    var strike by remember { mutableStateOf(700000.0) }
    var dte by remember { mutableStateOf(30.0) }
    var iv by remember { mutableStateOf(35.0) }
    var rf by remember { mutableStateOf(25.0) }
    var nContracts by remember { mutableStateOf(1.0) }
    var isCall by remember { mutableStateOf(true) }
    val lot = underlying.lotSize

    val T = dte / 365.0
    val r = rf / 100.0
    val sigma = iv / 100.0
    val greeks = BlackScholes.price(spot, strike, T, r, sigma, if (isCall) OptionType.CALL else OptionType.PUT)
    val totalPremium = greeks.price * lot * nContracts.toInt()
    val breakeven = if (isCall) spot + greeks.price else spot - greeks.price

    LazyColumn(verticalArrangement = Arrangement.spacedBy(16.dp), modifier = Modifier.fillMaxWidth()) {
        item {
            ScreenHeader("🏛 اختیار معامله — بورس کالای ایران", "قیمت‌گذاری · Greeks · P&L · استراتژی‌های ترکیبی · مقایسه سررسید")
        }
        item {
            SectionHeader("① دارایی پایه و پارامترهای قرارداد")
            SimpleDropdown("دارایی پایه", underlying, BourseStrategies.UNDERLYINGS, { "${it.icon} ${it.labelFa}" }, {
                underlying = it; spot = spot; strike = strike
            })
            Text("📦 اندازه قرارداد: ${lot} واحد · 💱 ${underlying.unitFa}", style = MaterialTheme.typography.labelSmall, color = colors.muted, modifier = Modifier.padding(top = 4.dp))
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.padding(top = 10.dp)) {
                NumField("قیمت لحظه‌ای (ریال)", spot, { spot = it }, modifier = Modifier.weight(1f))
                NumField("قیمت اعمال Strike (ریال)", strike, { strike = it }, modifier = Modifier.weight(1f))
            }
            Text("روز تا سررسید: ${dte.toInt()}", style = MaterialTheme.typography.labelMedium, modifier = Modifier.padding(top = 10.dp))
            Slider(value = dte.toFloat(), onValueChange = { dte = it.toDouble() }, valueRange = 1f..365f)
            Text("نوسان ضمنی IV: ${iv.toInt()}%", style = MaterialTheme.typography.labelMedium)
            Slider(value = iv.toFloat(), onValueChange = { iv = it.toDouble() }, valueRange = 5f..150f)
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                NumField("نرخ بدون ریسک (%)", rf, { rf = it }, modifier = Modifier.weight(1f))
                NumField("تعداد قرارداد", nContracts, { nContracts = it }, modifier = Modifier.weight(1f))
            }
            Row(verticalAlignment = androidx.compose.ui.Alignment.CenterVertically, modifier = Modifier.padding(top = 8.dp)) {
                RadioButton(selected = isCall, onClick = { isCall = true }); Text("اختیار خرید (Call)", modifier = Modifier.padding(end = 16.dp))
                RadioButton(selected = !isCall, onClick = { isCall = false }); Text("اختیار فروش (Put)")
            }
        }

        item {
            SectionHeader("② قیمت‌گذاری و Greeks")
            LazyRow(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
                items(
                    listOf(
                        "قیمت منصفانه (BS)" to "%,.0f ریال".format(greeks.price),
                        "Delta (Δ)" to "%.4f".format(greeks.delta),
                        "Theta روزانه" to "%,.0f ریال".format(greeks.theta * greeks.price),
                        "Vega per 1%%IV" to "%.4f".format(greeks.vega * spot),
                        "نقطه سربه‌سر" to "%,.0f ریال".format(breakeven),
                    ),
                ) { (l, v) -> MetricTile(l, v) }
            }
            Text("کل پرمیوم قراردادها: %,.0f ریال · وثیقه اولیه (تخمین): %,.0f ریال".format(totalPremium, totalPremium * 1.2), style = MaterialTheme.typography.labelSmall, color = colors.muted, modifier = Modifier.padding(top = 6.dp))
        }

        item {
            SectionHeader("③ نوسان ضمنی از قیمت بازار")
            var marketPrice by remember { mutableStateOf(0.0) }
            var ivCalc by remember { mutableStateOf<Double?>(null) }
            NumField("قیمت معامله‌شده در تابلو (ریال/واحد)", marketPrice, { marketPrice = it })
            AppButton(
                text = "محاسبه IV از قیمت بازار",
                onClick = { ivCalc = BlackScholes.impliedVolatility(marketPrice, spot, strike, T, r, if (isCall) OptionType.CALL else OptionType.PUT) },
                modifier = Modifier.fillMaxWidth().padding(top = 8.dp),
            )
            ivCalc?.let {
                val diff = it * 100 - iv
                Text(
                    "IV محاسبه‌شده: %.1f%% (%s نسبت به BS)".format(it * 100, if (diff > 5) "گران‌تر" else if (diff < -5) "ارزان‌تر" else "منصفانه"),
                    color = if (diff > 5) colors.red else if (diff < -5) colors.green else colors.gold, modifier = Modifier.padding(top = 6.dp),
                )
            }
        }

        item {
            SectionHeader("④ نمودار P&L در سررسید")
            var isLong by remember { mutableStateOf(true) }
            Row {
                RadioButton(selected = isLong, onClick = { isLong = true }); Text("خریدار (Long)", modifier = Modifier.padding(end = 16.dp))
                RadioButton(selected = !isLong, onClick = { isLong = false }); Text("فروشنده (Short)")
            }
            val sRange = (0 until 300).map { spot * 0.6 + spot * 0.8 * it / 299.0 }
            val pnl = sRange.map { s -> OptionsStrategies.pnlAtExpiry(s, strike, greeks.price, lot, nContracts.toInt().coerceAtLeast(1), if (isCall) OptionType.CALL else OptionType.PUT, isLong) }
            Card(modifier = Modifier.fillMaxWidth().padding(top = 8.dp)) {
                LineChart(
                    series = listOf(LineSeries("P&L", colors.blueAccent, pnl, fillToZero = true)),
                    xValues = sRange, verticalRefLines = listOf(RefLine(spot, colors.blueAccent, "Spot"), RefLine(strike, colors.gold, "Strike")),
                    horizontalRefLines = listOf(RefLine(0.0, colors.plotTick)),
                    title = "P&L در سررسید", showLegend = false,
                )
            }
        }

        item {
            SectionHeader("⑤ Theta Decay")
            val days = (dte.toInt() downTo 1).toList()
            val values = days.map { d -> BlackScholes.price(spot, strike, d / 365.0, r, sigma, if (isCall) OptionType.CALL else OptionType.PUT).price }
            Card(modifier = Modifier.fillMaxWidth()) {
                LineChart(
                    series = listOf(LineSeries("ارزش اختیار", colors.gold, values, fillToZero = true)),
                    xValues = days.map { it.toDouble() },
                    horizontalRefLines = listOf(RefLine(greeks.price, colors.muted, "Premium فعلی", dashed = true)),
                    title = "THETA DECAY", showLegend = false, height = 260.dp,
                )
            }
        }

        item {
            SectionHeader("⑥ مقایسه Strike — Option Chain")
            val pctSteps = listOf(-20, -15, -10, -5, 0, 5, 10, 15, 20)
            val chainStrikes = pctSteps.map { spot * (1 + it / 100.0) }
            Card(modifier = Modifier.fillMaxWidth()) {
                chainStrikes.forEach { k ->
                    val call = BlackScholes.price(spot, k, T, r, sigma, OptionType.CALL)
                    val put = BlackScholes.price(spot, k, T, r, sigma, OptionType.PUT)
                    Row(modifier = Modifier.fillMaxWidth().padding(vertical = 2.dp), horizontalArrangement = Arrangement.SpaceBetween) {
                        Text("%,.0f".format(k), style = MaterialTheme.typography.labelSmall, color = colors.textPrimary)
                        Text("C:%,.0f".format(call.price), style = MaterialTheme.typography.labelSmall, color = colors.green)
                        Text("P:%,.0f".format(put.price), style = MaterialTheme.typography.labelSmall, color = colors.red)
                    }
                }
                LineChart(
                    series = listOf(
                        LineSeries("Call", colors.green, chainStrikes.map { BlackScholes.price(spot, it, T, r, sigma, OptionType.CALL).price }),
                        LineSeries("Put", colors.red, chainStrikes.map { BlackScholes.price(spot, it, T, r, sigma, OptionType.PUT).price }),
                    ),
                    xValues = chainStrikes, verticalRefLines = listOf(RefLine(spot, colors.blueAccent, "Spot")),
                    title = "Option Chain", height = 260.dp,
                )
            }
        }

        item { CombinedStrategiesSection(spot, strike, T, r, sigma, lot, nContracts.toInt().coerceAtLeast(1)) }
        item { ExpiryComparisonSection(spot, strike, r, sigma, isCall, dte.toInt()) }
        item { RoiCalculatorSection(spot, strike, T, r, sigma, lot, nContracts.toInt().coerceAtLeast(1), isCall, greeks.price, dte.toInt()) }
        item { ChecklistSection(dte.toInt(), iv, greeks.delta) }
    }
}

@Composable
private fun CombinedStrategiesSection(spot: Double, strike: Double, T: Double, r: Double, sigma: Double, lot: Int, n: Int) {
    val colors = LocalChartColors.current
    var strategy by remember { mutableStateOf(BourseStrategyType.BULL_CALL) }
    SectionHeader("⑦ استراتژی‌های ترکیبی")
    SimpleDropdown("استراتژی", strategy, BourseStrategyType.entries, { it.label }, { strategy = it })

    val sRange = (0 until 300).map { spot * 0.6 + spot * 0.8 * it / 299.0 }
    val (pnl, info) = when (strategy) {
        BourseStrategyType.BULL_CALL -> {
            val kLow = spot * 0.95; val kHigh = spot * 1.05
            val debit = BlackScholes.price(spot, kLow, T, r, sigma, OptionType.CALL).price - BlackScholes.price(spot, kHigh, T, r, sigma, OptionType.CALL).price
            sRange.map { BourseStrategies.bullCallSpreadPnl(it, kLow, kHigh, debit, lot, n) } to listOf("هزینه خالص" to "%,.0f".format(debit * lot * n), "حداکثر سود" to "%,.0f".format((kHigh - kLow - debit) * lot * n))
        }
        BourseStrategyType.BEAR_PUT -> {
            val kHigh = spot * 1.05; val kLow = spot * 0.95
            val debit = BlackScholes.price(spot, kHigh, T, r, sigma, OptionType.PUT).price - BlackScholes.price(spot, kLow, T, r, sigma, OptionType.PUT).price
            sRange.map { BourseStrategies.bearPutSpreadPnl(it, kHigh, kLow, debit, lot, n) } to listOf("هزینه خالص" to "%,.0f".format(debit * lot * n), "حداکثر سود" to "%,.0f".format((kHigh - kLow - debit) * lot * n))
        }
        BourseStrategyType.STRADDLE -> {
            val prem = BlackScholes.price(spot, strike, T, r, sigma, OptionType.CALL).price + BlackScholes.price(spot, strike, T, r, sigma, OptionType.PUT).price
            sRange.map { BourseStrategies.straddlePnl(it, strike, prem, lot, n) } to listOf("هزینه کل" to "%,.0f".format(prem * lot * n), "Breakeven بالا" to "%,.0f".format(strike + prem), "Breakeven پایین" to "%,.0f".format(strike - prem))
        }
        BourseStrategyType.STRANGLE -> {
            val kPut = spot * 0.92; val kCall = spot * 1.08
            val prem = BlackScholes.price(spot, kCall, T, r, sigma, OptionType.CALL).price + BlackScholes.price(spot, kPut, T, r, sigma, OptionType.PUT).price
            sRange.map { BourseStrategies.stranglePnl(it, kPut, kCall, prem, lot, n) } to listOf("هزینه کل" to "%,.0f".format(prem * lot * n))
        }
        BourseStrategyType.COVERED_CALL -> {
            val kCc = spot * 1.05
            val prem = BlackScholes.price(spot, kCc, T, r, sigma, OptionType.CALL).price
            val units = lot * n
            sRange.map { BourseStrategies.coveredCallPnl(it, spot, kCc, prem, units) } to listOf("پرمیوم دریافتی" to "%,.0f".format(prem), "حداکثر سود" to "%,.0f".format((kCc - spot + prem) * units))
        }
        BourseStrategyType.PROTECTIVE_PUT -> {
            val kPp = spot * 0.95
            val prem = BlackScholes.price(spot, kPp, T, r, sigma, OptionType.PUT).price
            val units = lot * n
            sRange.map { BourseStrategies.protectivePutPnl(it, spot, kPp, prem, units) } to listOf("هزینه بیمه" to "%,.0f".format(prem), "حداکثر زیان" to "%,.0f".format((spot - kPp + prem) * units))
        }
    }
    Card(modifier = Modifier.fillMaxWidth()) {
        info.forEach { (l, v) -> Text("$l: $v", style = MaterialTheme.typography.labelSmall, color = colors.textPrimary) }
        LineChart(
            series = listOf(LineSeries(strategy.label, colors.blueAccent, pnl, fillToZero = true)),
            xValues = sRange, verticalRefLines = listOf(RefLine(spot, colors.muted, "Spot")),
            horizontalRefLines = listOf(RefLine(0.0, colors.plotTick)),
            title = "${strategy.label} — P&L در سررسید", showLegend = false, height = 300.dp,
        )
    }
}

@Composable
private fun ExpiryComparisonSection(spot: Double, strike: Double, r: Double, sigma: Double, isCall: Boolean, currentDte: Int) {
    val colors = LocalChartColors.current
    SectionHeader("⑧ مقایسه سررسید")
    val durations = listOf(7, 14, 21, 30, 45, 60, 90)
    val prices = durations.map { d -> BlackScholes.price(spot, strike, d / 365.0, r, sigma, if (isCall) OptionType.CALL else OptionType.PUT) }
    Card(modifier = Modifier.fillMaxWidth()) {
        durations.forEachIndexed { i, d -> Text("DTE $d: قیمت %,.0f · Delta %.3f".format(prices[i].price, prices[i].delta), style = MaterialTheme.typography.labelSmall, color = colors.textPrimary) }
        BarChart(
            categories = durations.map { it.toString() },
            series = listOf(BarSeries("قیمت BS", colors.blueAccent, prices.map { it.price })),
            title = "قیمت اختیار به ازای DTE", showLegend = false,
        )
    }
}

@Composable
private fun RoiCalculatorSection(spot: Double, strike: Double, T: Double, r: Double, sigma: Double, lot: Int, n: Int, isCall: Boolean, currentPrice: Double, dte: Int) {
    val colors = LocalChartColors.current
    SectionHeader("⑨ محاسبه بازده (ROI)")
    var buyPrice by remember { mutableStateOf(currentPrice) }
    var targetSpot by remember { mutableStateOf(spot * 1.10) }
    var targetDays by remember { mutableStateOf((dte * 0.7).coerceAtLeast(1.0)) }
    var stopSpot by remember { mutableStateOf(spot * 0.95) }

    Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
        NumField("قیمت خرید اختیار", buyPrice, { buyPrice = it }, modifier = Modifier.weight(1f))
        NumField("قیمت تارگت دارایی پایه", targetSpot, { targetSpot = it }, modifier = Modifier.weight(1f))
        NumField("قیمت استاپ دارایی پایه", stopSpot, { stopSpot = it }, modifier = Modifier.weight(1f))
    }
    val tTarget = max(dte - targetDays, 1.0) / 365.0
    val pTarget = BlackScholes.price(targetSpot, strike, tTarget, r, sigma, if (isCall) OptionType.CALL else OptionType.PUT).price
    val pStop = BlackScholes.price(stopSpot, strike, T, r, sigma, if (isCall) OptionType.CALL else OptionType.PUT).price
    val totalInvest = buyPrice * lot * n
    val pnlTarget = (pTarget - buyPrice) * lot * n
    val pnlStop = (pStop - buyPrice) * lot * n
    val rr = if (pnlStop != 0.0) abs(pnlTarget / pnlStop) else Double.POSITIVE_INFINITY

    Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.padding(top = 10.dp)) {
        MetricTile("سرمایه‌گذاری کل", "%,.0f".format(totalInvest))
        MetricTile("سود در تارگت", "%,.0f (%.1f%%)".format(pnlTarget, pnlTarget / totalInvest * 100), valueColor = colors.green)
        MetricTile("زیان در استاپ", "%,.0f (%.1f%%)".format(pnlStop, pnlStop / totalInvest * 100), valueColor = colors.red)
        MetricTile("R/R", if (rr.isFinite()) "%.2f".format(rr) else "∞", valueColor = if (rr >= 2) colors.green else if (rr >= 1) colors.gold else colors.red)
    }
}

@Composable
private fun ChecklistSection(dte: Int, iv: Double, delta: Double) {
    val colors = LocalChartColors.current
    SectionHeader("⑩ چک‌لیست پیش از معامله")
    val checks = listOf(
        (dte >= 21) to "زمان کافی تا سررسید (حداقل ۲۱ روز)",
        (iv <= 80) to "IV در محدوده منطقی (زیر ۸۰٪)",
        (abs(delta) <= 0.6) to "Delta مناسب",
    )
    val passed = checks.count { it.first }
    Card(modifier = Modifier.fillMaxWidth()) {
        Text("آمادگی معامله: $passed/${checks.size} ✓", style = MaterialTheme.typography.titleSmall, color = if (passed >= checks.size - 1) colors.green else colors.gold)
        checks.forEach { (ok, label) -> Text("${if (ok) "✓" else "✗"} $label", color = if (ok) colors.green else colors.red, style = MaterialTheme.typography.bodySmall, modifier = Modifier.padding(top = 4.dp)) }
        Text("💡 اختیار ایده‌آل: OTM · IV ۲۰–۶۰٪ · Delta ۰.۲–۰.۴ · DTE ۲۱–۴۵ روز · R/R بالای ۲", style = MaterialTheme.typography.labelSmall, color = colors.muted, modifier = Modifier.padding(top = 8.dp))
    }
}
