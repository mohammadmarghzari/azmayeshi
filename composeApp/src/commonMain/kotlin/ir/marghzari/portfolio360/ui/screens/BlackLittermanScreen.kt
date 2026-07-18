package ir.marghzari.portfolio360.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import ir.marghzari.portfolio360.charts.BarChart
import ir.marghzari.portfolio360.charts.BarSeries
import ir.marghzari.portfolio360.core.math.BlackLitterman
import ir.marghzari.portfolio360.core.math.FactorExposure
import ir.marghzari.portfolio360.core.math.FactorRow
import ir.marghzari.portfolio360.core.math.Stats
import ir.marghzari.portfolio360.state.AppState
import ir.marghzari.portfolio360.theme.LocalChartColors
import ir.marghzari.portfolio360.ui.components.Card
import ir.marghzari.portfolio360.ui.components.InfoBanner
import ir.marghzari.portfolio360.ui.components.SectionHeader
import ir.marghzari.portfolio360.ui.components.SimpleDropdown

@Composable
fun BlackLittermanScreen(appState: AppState) {
    val colors = LocalChartColors.current
    val prices = appState.prices
    if (prices == null) {
        InfoBanner("ابتدا داده پرتفوی را دانلود کنید.")
        return
    }
    val returns = remember(prices) { prices.dailyReturns() }
    val cov = remember(returns) { Stats.annualizedCovariance(returns, prices.nAssets) }
    val meanDaily = remember(returns) { DoubleArray(prices.nAssets) { j -> Stats.mean(DoubleArray(returns.size) { returns[it][j] }) } }
    val equalWeights = remember(prices) { DoubleArray(prices.nAssets) { 1.0 / prices.nAssets } }

    var nViews by remember { mutableStateOf(0f) }
    var viewTickers by remember { mutableStateOf(List(prices.nAssets) { prices.tickers[it % prices.tickers.size] }) }
    var viewReturns by remember { mutableStateOf(List(prices.nAssets) { 10.0 }) }
    var tau by remember { mutableStateOf(0.05f) }
    var blWeights by remember { mutableStateOf<DoubleArray?>(null) }
    var factorRows by remember { mutableStateOf<List<FactorRow>>(emptyList()) }

    LazyColumn(verticalArrangement = Arrangement.spacedBy(16.dp), modifier = Modifier.fillMaxWidth()) {
        item {
            SectionHeader("🧠 Black-Litterman — دیدگاه‌های شخصی روی بازار")
            Text("برای هر نمادی که دیدگاه دارید، بازده سالانه مورد انتظار وارد کنید.", style = MaterialTheme.typography.bodySmall, color = colors.muted)
        }
        item {
            Text("تعداد دیدگاه: ${nViews.toInt()}", style = MaterialTheme.typography.labelMedium)
            Slider(value = nViews, onValueChange = { nViews = it }, valueRange = 0f..minOf(prices.nAssets, 10).toFloat(), steps = minOf(prices.nAssets, 10) - 1)
        }
        items(nViews.toInt()) { i ->
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.fillMaxWidth()) {
                SimpleDropdown(
                    "نماد ${i + 1}", viewTickers.getOrElse(i) { prices.tickers[0] }, prices.tickers, { it },
                    { t -> viewTickers = viewTickers.toMutableList().also { if (it.size <= i) repeat(i - it.size + 1) { _ -> it.add(prices.tickers[0]) }; it[i] = t } },
                    modifier = Modifier.weight(1f),
                )
                Card(modifier = Modifier.weight(1f)) {
                    Text("بازده انتظاری (%)", style = MaterialTheme.typography.labelSmall, color = colors.muted)
                    var text by remember { mutableStateOf(viewReturns.getOrElse(i) { 10.0 }.toString()) }
                    OutlinedTextField(
                        value = text, onValueChange = { s ->
                            text = s
                            s.toDoubleOrNull()?.let { v -> viewReturns = viewReturns.toMutableList().also { if (it.size <= i) repeat(i - it.size + 1) { _ -> it.add(10.0) }; it[i] = v } }
                        },
                        modifier = Modifier.fillMaxWidth().padding(top = 4.dp), singleLine = true,
                    )
                }
            }
        }
        item {
            Text("Tau (اطمینان به پرایور): %.2f".format(tau), style = MaterialTheme.typography.labelMedium)
            Slider(value = tau, onValueChange = { tau = it }, valueRange = 0.01f..0.20f)
        }
        item {
            Button(
                onClick = {
                    val views = (0 until nViews.toInt()).associate { i -> viewTickers[i] to viewReturns[i] / 100.0 }
                    val result = BlackLitterman.compute(equalWeights, cov, meanDaily, prices.tickers, views, tau.toDouble())
                    blWeights = result.weights
                },
                modifier = Modifier.fillMaxWidth(), colors = ButtonDefaults.buttonColors(containerColor = colors.blueAccent),
            ) { Text("محاسبه وزن‌های Black-Litterman") }
        }
        blWeights?.let { w ->
            item {
                Card(modifier = Modifier.fillMaxWidth()) {
                    val sortedIdx = prices.tickers.indices.sortedByDescending { w[it] }
                    sortedIdx.forEach { i ->
                        Row(modifier = Modifier.fillMaxWidth().padding(vertical = 3.dp), horizontalArrangement = Arrangement.SpaceBetween) {
                            Text(prices.tickers[i], color = colors.textPrimary)
                            Text("%.2f%%".format(w[i] * 100), color = colors.blueAccent)
                            Text("برابر: %.2f%%".format(100.0 / prices.nAssets), color = colors.muted)
                        }
                    }
                    BarChart(
                        categories = sortedIdx.map { prices.tickers[it] },
                        series = listOf(
                            BarSeries("BL", colors.blueAccent, sortedIdx.map { w[it] * 100 }),
                            BarSeries("برابر", colors.muted, sortedIdx.map { 100.0 / prices.nAssets }),
                        ),
                        title = "BL vs EQUAL WEIGHT",
                    )
                }
            }
        }
        item {
            SectionHeader("Factor Exposure — عوامل موثر بر دارایی‌ها")
            Button(
                onClick = { factorRows = FactorExposure.compute(returns, prices.tickers) },
                modifier = Modifier.fillMaxWidth(), colors = ButtonDefaults.buttonColors(containerColor = colors.gold),
            ) { Text("محاسبه Factor Exposure") }
        }
        if (factorRows.isNotEmpty()) {
            item {
                Card(modifier = Modifier.fillMaxWidth()) {
                    factorRows.forEach { r ->
                        Row(modifier = Modifier.fillMaxWidth().padding(vertical = 3.dp), horizontalArrangement = Arrangement.SpaceBetween) {
                            Text(r.ticker, color = colors.textPrimary)
                            Text("مومنتوم %.1f%%".format(r.momentum6m), color = colors.textPrimary, style = MaterialTheme.typography.labelSmall)
                            Text("نوسان %.1f%%".format(r.annualVol), color = colors.textPrimary, style = MaterialTheme.typography.labelSmall)
                            Text("بتا %.2f".format(r.beta), color = colors.textPrimary, style = MaterialTheme.typography.labelSmall)
                            Text("شارپ %.2f".format(r.sharpe), color = colors.textPrimary, style = MaterialTheme.typography.labelSmall)
                        }
                    }
                    BarChart(
                        categories = factorRows.map { it.ticker },
                        series = listOf(
                            BarSeries("مومنتوم", colors.blueAccent, factorRows.map { it.momentum6m }),
                            BarSeries("نوسان", colors.red, factorRows.map { it.annualVol }),
                            BarSeries("بتا", colors.gold, factorRows.map { it.beta }),
                            BarSeries("شارپ", colors.green, factorRows.map { it.sharpe }),
                        ),
                        title = "FACTOR EXPOSURE",
                    )
                }
            }
        }
    }
}
