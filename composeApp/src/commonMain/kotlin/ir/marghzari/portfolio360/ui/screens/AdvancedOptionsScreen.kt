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
import ir.marghzari.portfolio360.core.math.CoveredCallResult
import ir.marghzari.portfolio360.core.math.IronCondorResult
import ir.marghzari.portfolio360.core.math.OptionType
import ir.marghzari.portfolio360.core.math.OptionsStrategies
import ir.marghzari.portfolio360.core.math.ProtectivePutResult
import ir.marghzari.portfolio360.core.math.RollingCcCycle
import ir.marghzari.portfolio360.core.model.RiskInputs
import ir.marghzari.portfolio360.state.AppState
import ir.marghzari.portfolio360.state.HedgedAsset
import ir.marghzari.portfolio360.theme.LocalChartColors
import ir.marghzari.portfolio360.ui.components.ScreenHeader
import ir.marghzari.portfolio360.ui.components.EmptyState
import ir.marghzari.portfolio360.ui.components.Card
import ir.marghzari.portfolio360.ui.components.MetricTile
import ir.marghzari.portfolio360.ui.components.SectionHeader
import ir.marghzari.portfolio360.ui.components.SimpleDropdown
import ir.marghzari.portfolio360.ui.components.VerdictCard
import ir.marghzari.portfolio360.ui.components.VerdictTone
import kotlin.math.max
import kotlin.math.min

private enum class OptionsStrategy(val label: String) {
    COVERED_CALL("Covered Call"), PROTECTIVE_PUT("Protective Put"), IRON_CONDOR("Iron Condor"), ROLLING_CC("Rolling Covered Call"),
}

@Composable
fun AdvancedOptionsScreen(appState: AppState) {
    var strategy by remember { mutableStateOf(OptionsStrategy.PROTECTIVE_PUT) }

    LazyColumn(verticalArrangement = Arrangement.spacedBy(16.dp), modifier = Modifier.fillMaxWidth()) {
        item {
            ScreenHeader("🎯 اختیار پیشرفته")
            SimpleDropdown("استراتژی", strategy, OptionsStrategy.entries, { it.label }, { strategy = it })
        }
        item {
            when (strategy) {
                OptionsStrategy.COVERED_CALL -> CoveredCallTool(appState)
                OptionsStrategy.PROTECTIVE_PUT -> ProtectivePutTool(appState)
                OptionsStrategy.IRON_CONDOR -> IronCondorTool(appState)
                OptionsStrategy.ROLLING_CC -> RollingCcTool(appState)
            }
        }
    }
}

@Composable
private fun NumField(label: String, value: Double, onChange: (Double) -> Unit) {
    var text by remember(value) { mutableStateOf(if (value == value.toLong().toDouble()) value.toLong().toString() else value.toString()) }
    Card {
        Text(label, style = MaterialTheme.typography.labelSmall, color = LocalChartColors.current.muted)
        androidx.compose.material3.OutlinedTextField(
            value = text, onValueChange = { text = it; it.toDoubleOrNull()?.let(onChange) },
            modifier = Modifier.fillMaxWidth().padding(top = 4.dp), singleLine = true,
        )
    }
}

@Composable
private fun CoveredCallTool(appState: AppState) {
    val colors = LocalChartColors.current
    var spot by remember { mutableStateOf(100.0) }
    var strike by remember { mutableStateOf(105.0) }
    var days by remember { mutableStateOf(30.0) }
    var iv by remember { mutableStateOf(30.0) }
    var premium by remember { mutableStateOf(0.0) }
    var contracts by remember { mutableStateOf(1.0) }
    var result by remember { mutableStateOf<CoveredCallResult?>(null) }

    Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
        NumField("قیمت فعلی ($)", spot) { spot = it }
        NumField("Strike ($)", strike) { strike = it }
    }
    Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.padding(top = 10.dp)) {
        NumField("روز تا انقضا", days) { days = it }
        NumField("IV (%)", iv) { iv = it }
    }
    Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.padding(top = 10.dp)) {
        NumField("پرمیوم دریافتی ($)", premium) { premium = it }
        NumField("تعداد قرارداد", contracts) { contracts = it }
    }
    Button(
        onClick = {
            result = OptionsStrategies.analyzeCoveredCall(spot, strike, days.toInt().coerceAtLeast(1), appState.rf, iv / 100, premium, contracts.toInt().coerceAtLeast(1), appState.riskInputs)
        },
        modifier = Modifier.fillMaxWidth().padding(top = 12.dp), colors = ButtonDefaults.buttonColors(containerColor = colors.blueAccent),
    ) { Text("📊 تحلیل Covered Call") }

    result?.let { r ->
        val tone = if (r.worthwhileScore >= 0.01) VerdictTone.POSITIVE else if (r.worthwhileScore >= -0.02) VerdictTone.NEUTRAL else VerdictTone.NEGATIVE
        VerdictCard(
            title = if (tone == VerdictTone.POSITIVE) "✅ فروش این قرارداد به‌صرفه است" else if (tone == VerdictTone.NEUTRAL) "⚠ مرزی" else "❌ فروش این قرارداد به‌صرفه نیست",
            body = "بازده تعدیل‌شده CC: %.2f%% در مقابل بازده مورد انتظار: %.2f%%".format(r.ccAdjRet * 100, r.expectedAnnAdj * 100),
            tone = tone, modifier = Modifier.padding(top = 12.dp),
        )
        LazyRow(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.padding(top = 10.dp)) {
            items(
                listOf(
                    "قیمت BS" to "$%.2f".format(r.bsPrice), "Delta" to "%.3f".format(r.delta),
                    "پرمیوم کل" to "$%.2f".format(r.totalPremium), "بازده سالانه پرمیوم" to "%.2f%%".format(r.annPremiumYield * 100),
                    "سربه‌سر" to "$%.2f".format(r.breakeven), "Moneyness" to "%.1f%%".format(r.moneyness),
                ),
            ) { (l, v) -> MetricTile(l, v) }
        }
        Card(modifier = Modifier.fillMaxWidth().padding(top = 10.dp)) {
            val sRange = (0 until 300).map { spot * 0.5 + (spot * 1.0) * it / 299.0 }
            val pnl = sRange.map { s -> (s - spot) * r.shares + (if (s <= strike) r.premium * r.shares else (r.premium - (s - strike)) * r.shares) }
            LineChart(
                series = listOf(LineSeries("P&L", colors.blueAccent, pnl, fillToZero = true)),
                xValues = sRange, verticalRefLines = listOf(RefLine(strike, colors.gold, "Strike")),
                horizontalRefLines = listOf(RefLine(0.0, colors.plotTick)),
                title = "P&L در انقضا", showLegend = false,
            )
        }
    }
}

@Composable
private fun ProtectivePutTool(appState: AppState) {
    val colors = LocalChartColors.current
    val prices = appState.prices
    var selectedTicker by remember { mutableStateOf<String?>(null) }
    val autoSpot = selectedTicker?.let { prices?.column(it)?.lastOrNull() }
    var spot by remember { mutableStateOf(100.0) }
    var strike by remember { mutableStateOf(95.0) }
    var days by remember { mutableStateOf(30.0) }
    var iv by remember { mutableStateOf(30.0) }
    var shares by remember { mutableStateOf(100.0) }
    var premium by remember { mutableStateOf(0.0) }
    var result by remember { mutableStateOf<ProtectivePutResult?>(null) }

    if (prices != null) {
        SimpleDropdown(
            "دارایی برای بیمه‌کردن (اختیاری)", selectedTicker ?: "— وارد دستی —",
            listOf("— وارد دستی —") + prices.tickers, { it }, {
                selectedTicker = if (it == "— وارد دستی —") null else it
                autoSpot?.let { s -> spot = s; strike = s * 0.95 }
            },
        )
    }

    Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.padding(top = 10.dp)) {
        NumField("قیمت فعلی ($)", spot) { spot = it }
        NumField("Strike Put ($)", strike) { strike = it }
    }
    Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.padding(top = 10.dp)) {
        NumField("روز تا انقضا", days) { days = it }
        NumField("IV (%)", iv) { iv = it }
    }
    Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.padding(top = 10.dp)) {
        NumField("تعداد سهام", shares) { shares = it }
        NumField("پرمیوم واقعی ($)", premium) { premium = it }
    }
    Button(
        onClick = {
            result = OptionsStrategies.analyzeProtectivePut(spot, strike, days.toInt().coerceAtLeast(1), appState.rf, iv / 100, premium, shares.toInt().coerceAtLeast(1), appState.riskInputs)
        },
        modifier = Modifier.fillMaxWidth().padding(top = 12.dp), colors = ButtonDefaults.buttonColors(containerColor = colors.blueAccent),
    ) { Text("محاسبه Protective Put") }

    result?.let { r ->
        VerdictCard(
            title = if (r.worthwhile) "✅ بیمه منطقی است" else "❌ هزینه بیمه زیاد است",
            body = "هزینه سالانه‌شده بیمه: %.2f%% — بازده انتظاری تعدیل‌شده: %.2f%%".format(r.costAnnPct, r.expectedAdjPct),
            tone = if (r.worthwhile) VerdictTone.POSITIVE else VerdictTone.NEGATIVE, modifier = Modifier.padding(top = 12.dp),
        )
        LazyRow(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.padding(top = 10.dp)) {
            items(
                listOf(
                    "قیمت Put (BS)" to "$%.2f".format(r.bsPrice), "کل هزینه بیمه" to "$%.2f".format(r.totalCost),
                    "حداکثر زیان بیمه‌شده" to "$%.2f".format(r.maxLossInsured), "سربه‌سر" to "$%.2f".format(r.breakeven),
                    "Delta" to "%.3f".format(r.delta), "هزینه سالانه‌شده" to "%.2f%%".format(r.costAnnPct),
                ),
            ) { (l, v) -> MetricTile(l, v) }
        }
        Card(modifier = Modifier.fillMaxWidth().padding(top = 10.dp)) {
            val sRange = (0 until 300).map { spot * 0.5 + spot * it / 299.0 }
            val naked = sRange.map { s -> (s - spot) * shares }
            val withPp = sRange.mapIndexed { i, s -> naked[i] + max(strike - s, 0.0) * shares - r.totalCost }
            LineChart(
                series = listOf(
                    LineSeries("با Protective Put", colors.blueAccent, withPp),
                    LineSeries("بدون بیمه", colors.muted, naked, dashed = true),
                ),
                xValues = sRange, verticalRefLines = listOf(RefLine(strike, colors.red, "Strike")),
                horizontalRefLines = listOf(RefLine(0.0, colors.plotTick)),
                title = "PROTECTIVE PUT — P&L AT EXPIRATION",
            )
        }
        if (selectedTicker != null) {
            Button(
                onClick = {
                    val t = selectedTicker!!
                    appState.hedgedAssets = appState.hedgedAssets + (t to HedgedAsset(t, strike, spot, r.premium, days.toInt()))
                },
                modifier = Modifier.fillMaxWidth().padding(top = 10.dp), colors = ButtonDefaults.buttonColors(containerColor = colors.green),
            ) { Text("✅ اعمال بیمه $selectedTicker به پرتفو") }
        }
    }
}

@Composable
private fun IronCondorTool(appState: AppState) {
    val colors = LocalChartColors.current
    var spot by remember { mutableStateOf(100.0) }
    var kpb by remember { mutableStateOf(88.0) }
    var kps by remember { mutableStateOf(92.0) }
    var kcs by remember { mutableStateOf(108.0) }
    var kcb by remember { mutableStateOf(112.0) }
    var days by remember { mutableStateOf(30.0) }
    var iv by remember { mutableStateOf(30.0) }
    var contracts by remember { mutableStateOf(1.0) }
    var result by remember { mutableStateOf<IronCondorResult?>(null) }

    Row(horizontalArrangement = Arrangement.spacedBy(10.dp)) {
        NumField("قیمت فعلی ($)", spot) { spot = it }
        NumField("تعداد قرارداد", contracts) { contracts = it }
    }
    Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.padding(top = 10.dp)) {
        NumField("Strike Put Buy", kpb) { kpb = it }
        NumField("Strike Put Sell", kps) { kps = it }
    }
    Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.padding(top = 10.dp)) {
        NumField("Strike Call Sell", kcs) { kcs = it }
        NumField("Strike Call Buy", kcb) { kcb = it }
    }
    Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.padding(top = 10.dp)) {
        NumField("روز تا انقضا", days) { days = it }
        NumField("IV (%)", iv) { iv = it }
    }
    Button(
        onClick = {
            result = OptionsStrategies.analyzeIronCondor(spot, kpb, kps, kcs, kcb, days.toInt().coerceAtLeast(1), appState.rf, iv / 100, contracts.toInt().coerceAtLeast(1), appState.riskInputs)
        },
        modifier = Modifier.fillMaxWidth().padding(top = 12.dp), colors = ButtonDefaults.buttonColors(containerColor = colors.blueAccent),
    ) { Text("محاسبه Iron Condor") }

    result?.let { r ->
        val tone = if (r.worthwhileScorePct >= 1) VerdictTone.POSITIVE else if (r.worthwhileScorePct >= -2) VerdictTone.NEUTRAL else VerdictTone.NEGATIVE
        VerdictCard(
            title = if (tone == VerdictTone.POSITIVE) "✅ Iron Condor به‌صرفه است" else if (tone == VerdictTone.NEUTRAL) "⚠ مرزی" else "❌ به‌صرفه نیست",
            body = "محدوده سود: $%.2f تا $%.2f (%.1f%% از قیمت فعلی)".format(r.beLower, r.beUpper, r.profitZonePct),
            tone = tone, modifier = Modifier.padding(top = 12.dp),
        )
        LazyRow(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.padding(top = 10.dp)) {
            items(
                listOf(
                    "اعتبار خالص" to "$%.2f".format(r.netCredit), "کل اعتبار" to "$%.2f".format(r.totalCredit),
                    "حداکثر زیان" to "$%.2f".format(r.maxLoss), "بازده روی ریسک" to "%.1f%%".format(r.retOnRiskPct),
                ),
            ) { (l, v) -> MetricTile(l, v) }
        }
        Card(modifier = Modifier.fillMaxWidth().padding(top = 10.dp)) {
            val sRange = (0 until 300).map { kpb * 0.9 + (kcb * 1.1 - kpb * 0.9) * it / 299.0 }
            val pnl = sRange.map { s -> OptionsStrategies.ironCondorPnlAt(s, kpb, kps, kcs, kcb, r.netCredit, contracts.toInt().coerceAtLeast(1)) }
            LineChart(
                series = listOf(LineSeries("Iron Condor P&L", colors.blueAccent, pnl, fillToZero = true)),
                xValues = sRange,
                verticalRefLines = listOf(RefLine(kpb, colors.red, "PB"), RefLine(kps, colors.gold, "PS"), RefLine(kcs, colors.gold, "CS"), RefLine(kcb, colors.red, "CB")),
                horizontalRefLines = listOf(RefLine(0.0, colors.plotTick)),
                title = "IRON CONDOR — P&L AT EXPIRATION",
            )
        }
    }
}

@Composable
private fun RollingCcTool(appState: AppState) {
    val colors = LocalChartColors.current
    val prices = appState.prices
    if (prices == null) {
        EmptyState(title = "هنوز داده‌ای دریافت نشده", hint = "از تب «تخصیص دارایی» داده پرتفوی را دانلود کنید.")
        return
    }
    var ticker by remember { mutableStateOf(prices.tickers.first()) }
    var offsetPct by remember { mutableStateOf(5.0) }
    var dte by remember { mutableStateOf(30.0) }
    var iv by remember { mutableStateOf(30.0) }
    var contracts by remember { mutableStateOf(1.0) }
    var cycles by remember { mutableStateOf<List<RollingCcCycle>>(emptyList()) }

    SimpleDropdown("نماد پایه", ticker, prices.tickers, { it }, { ticker = it })
    Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.padding(top = 10.dp)) {
        NumField("Strike OTM (%)", offsetPct) { offsetPct = it }
        NumField("DTE هر دوره", dte) { dte = it }
    }
    Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.padding(top = 10.dp)) {
        NumField("IV فرضی (%)", iv) { iv = it }
        NumField("تعداد قرارداد", contracts) { contracts = it }
    }
    Button(
        onClick = {
            cycles = OptionsStrategies.simulateRollingCc(prices.column(ticker), offsetPct, dte.toInt().coerceAtLeast(5), appState.rf, iv / 100, contracts.toInt().coerceAtLeast(1))
        },
        modifier = Modifier.fillMaxWidth().padding(top = 12.dp), colors = ButtonDefaults.buttonColors(containerColor = colors.blueAccent),
    ) { Text("اجرای شبیه‌سازی Rolling CC") }

    if (cycles.isNotEmpty()) {
        val totalPremium = cycles.sumOf { it.premiumEarned }
        val optionPnl = cycles.sumOf { it.optionPnl }
        val exercisedCount = cycles.count { it.exercised }
        LazyRow(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.padding(top = 10.dp)) {
            items(
                listOf(
                    "کل پرمیوم دریافتی" to "$%.2f".format(totalPremium), "سود/زیان اختیارات" to "$%.2f".format(optionPnl),
                    "تعداد سیکل‌ها" to cycles.size.toString(), "دفعات اعمال" to "$exercisedCount (%.0f%%)".format(exercisedCount * 100.0 / cycles.size),
                ),
            ) { (l, v) -> MetricTile(l, v) }
        }
        Card(modifier = Modifier.fillMaxWidth().padding(top = 10.dp)) {
            var cum = 0.0
            val cumPremium = cycles.map { cum += it.optionPnl; cum }
            BarChart(
                categories = cycles.indices.map { (it + 1).toString() },
                series = listOf(BarSeries("سود اختیار", colors.green, cycles.map { it.optionPnl })),
                title = "ROLLING CC — سود دوره‌ای", showLegend = false,
            )
            LineChart(series = listOf(LineSeries("تجمعی", colors.gold, cumPremium)), title = "تجمعی پرمیوم", showLegend = false)
        }
    }
}
