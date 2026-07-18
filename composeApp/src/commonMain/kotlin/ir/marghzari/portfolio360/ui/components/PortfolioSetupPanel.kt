package ir.marghzari.portfolio360.ui.components

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ExperimentalLayoutApi
import androidx.compose.foundation.layout.FlowRow
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.text.KeyboardOptions
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.CircularProgressIndicator
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.FilterChip
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.input.KeyboardType
import androidx.compose.ui.unit.dp
import ir.marghzari.portfolio360.core.math.PortfolioEngine
import ir.marghzari.portfolio360.core.model.HistoryPeriod
import ir.marghzari.portfolio360.core.model.PortfolioStyle
import ir.marghzari.portfolio360.core.model.RiskInputs
import ir.marghzari.portfolio360.core.model.SymbolCatalog
import ir.marghzari.portfolio360.core.network.PriceAligner
import ir.marghzari.portfolio360.state.AppState
import ir.marghzari.portfolio360.theme.LocalChartColors
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

/**
 * The persistent "sidebar" of the original Streamlit app — asset selection, period, risk-free
 * rate, expected return + risk sliders, and the fetch/calculate actions — surfaced here as the
 * setup panel at the top of the Allocation screen (the natural landing tab).
 */
@OptIn(ExperimentalMaterial3Api::class, ExperimentalLayoutApi::class)
@Composable
fun PortfolioSetupPanel(appState: AppState, modifier: Modifier = Modifier) {
    val colors = LocalChartColors.current
    val scope = rememberCoroutineScope()

    Card(modifier = modifier.fillMaxWidth()) {
        Text("انتخاب دارایی و تنظیمات", style = MaterialTheme.typography.titleMedium, color = colors.textPrimary)

        // Period selector
        val selectedPeriod = HistoryPeriod.entries.firstOrNull { it.apiCode == appState.periodCode } ?: HistoryPeriod.Y2
        SimpleDropdown(
            label = "بازه زمانی", selected = selectedPeriod, options = HistoryPeriod.entries,
            optionLabel = { it.faLabel }, onSelected = { appState.periodCode = it.apiCode },
            modifier = Modifier.padding(top = 14.dp),
        )

        // Risk-free rate
        LabeledNumberField(
            label = "نرخ بدون ریسک (%)", value = appState.riskFreeRatePct,
            onValueChange = { appState.riskFreeRatePct = it }, modifier = Modifier.padding(top = 12.dp),
        )

        SectionHeader("انتظارات و ریسک")
        LabeledNumberField(
            label = "بازده مورد انتظار (%)", value = appState.riskInputs.expectedReturnPct,
            onValueChange = { appState.riskInputs = appState.riskInputs.copy(expectedReturnPct = it) },
        )
        RiskSlider("🌐 ریسک ژئوپولیتیک (%)", appState.riskInputs.riskGeoPct, colors.riskGeo) {
            appState.riskInputs = appState.riskInputs.copy(riskGeoPct = it)
        }
        RiskSlider("🏦 ریسک سیاست پولی (%)", appState.riskInputs.riskMonPct, colors.riskMon) {
            appState.riskInputs = appState.riskInputs.copy(riskMonPct = it)
        }
        RiskSlider("📉 ریسک سیستماتیک (%)", appState.riskInputs.riskSysPct, colors.riskSys) {
            appState.riskInputs = appState.riskInputs.copy(riskSysPct = it)
        }
        val penalty = PortfolioEngine.riskPenalty(appState.riskInputs) * 100
        Text(
            "تنزل بازده: −%.1f%% · نرخ مؤثر: %.2f%%".format(penalty, appState.riskFreeRatePct + penalty),
            style = MaterialTheme.typography.labelSmall, color = colors.gold, modifier = Modifier.padding(top = 4.dp),
        )

        SectionHeader("انتخاب دارایی — ${appState.selectedTickers.size} نماد انتخاب‌شده")
        SymbolCatalog.CATEGORIES.forEach { category ->
            var expanded by remember { mutableStateOf(false) }
            Column(modifier = Modifier.padding(top = 6.dp)) {
                Row(
                    modifier = Modifier.fillMaxWidth().padding(vertical = 4.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                ) {
                    Text(category.labelFa, style = MaterialTheme.typography.labelLarge, color = colors.textPrimary)
                    TextButton(onClick = { expanded = !expanded }) {
                        Text(if (expanded) "بستن" else "نمایش")
                    }
                }
                if (expanded) {
                    FlowRow(horizontalArrangement = Arrangement.spacedBy(6.dp), verticalArrangement = Arrangement.spacedBy(6.dp)) {
                        category.symbols.forEach { (ticker, name) ->
                            val isSelected = ticker in appState.selectedTickers
                            FilterChip(
                                selected = isSelected,
                                onClick = {
                                    appState.selectedTickers = if (isSelected) appState.selectedTickers - ticker else appState.selectedTickers + ticker
                                },
                                label = { Text("$ticker · $name", style = MaterialTheme.typography.labelSmall) },
                            )
                        }
                    }
                }
            }
        }

        Button(
            onClick = {
                scope.launch {
                    appState.isFetching = true
                    appState.lastError = null
                    try {
                        val histories = withContext(Dispatchers.Default) {
                            appState.selectedTickers.map { t -> appState.yahoo.fetchHistory(t, appState.periodCode) }
                        }
                        val result = PriceAligner.align(histories, appState.selectedTickers)
                        appState.prices = result.series
                        appState.fetchFailedTickers = result.failedTickers
                        appState.resetComputedPortfolio()
                        if (result.series == null) appState.lastError = "دریافت داده ناموفق بود."
                    } catch (e: Exception) {
                        appState.lastError = e.message ?: "خطای ناشناخته"
                    } finally {
                        appState.isFetching = false
                    }
                }
            },
            enabled = appState.selectedTickers.size >= 2 && !appState.isFetching,
            modifier = Modifier.fillMaxWidth().padding(top = 14.dp),
            colors = ButtonDefaults.buttonColors(containerColor = colors.blueAccent),
        ) {
            if (appState.isFetching) {
                CircularProgressIndicator(modifier = Modifier.width(16.dp), color = MaterialTheme.colorScheme.onPrimary, strokeWidth = 2.dp)
            } else {
                Text("↓  دریافت داده از Yahoo Finance")
            }
        }
        if (appState.selectedTickers.size < 2) {
            Text("⚠ حداقل ۲ نماد برای محاسبه پرتفوی لازم است", style = MaterialTheme.typography.labelSmall, color = colors.gold, modifier = Modifier.padding(top = 6.dp))
        }
        appState.fetchFailedTickers.takeIf { it.isNotEmpty() }?.let {
            Text("⚠ دریافت ناموفق: ${it.joinToString(", ")}", style = MaterialTheme.typography.labelSmall, color = colors.red, modifier = Modifier.padding(top = 4.dp))
        }
        appState.lastError?.let {
            Text(it, style = MaterialTheme.typography.labelSmall, color = colors.red, modifier = Modifier.padding(top = 4.dp))
        }

        SectionHeader("استراتژی پرتفوی")
        SimpleDropdown(
            label = "روش بهینه‌سازی", selected = appState.portfolioStyle, options = PortfolioStyle.entries,
            optionLabel = { it.faLabel }, onSelected = { appState.portfolioStyle = it },
        )

        Button(
            onClick = {
                val prices = appState.prices ?: return@Button
                scope.launch {
                    appState.isCalculating = true
                    try {
                        withContext(Dispatchers.Default) {
                            val returns = prices.dailyReturns()
                            val result = PortfolioEngine.calcWeights(appState.portfolioStyle, returns, appState.rf, prices.tickers, appState.riskInputs)
                            val metrics = PortfolioEngine.portfolioMetrics(result.weights, returns, appState.rf, appState.riskInputs)
                            appState.weights = result.weights
                            appState.covariance = result.covariance
                            appState.metrics = metrics
                            appState.styleLabelUsed = appState.portfolioStyle.faLabel
                            appState.lastUsedRisk = appState.riskInputs
                        }
                    } finally {
                        appState.isCalculating = false
                    }
                }
            },
            enabled = appState.prices != null && appState.prices!!.nAssets >= 2 && !appState.isCalculating,
            modifier = Modifier.fillMaxWidth().padding(top = 12.dp),
            colors = ButtonDefaults.buttonColors(containerColor = colors.gold),
        ) {
            if (appState.isCalculating) {
                CircularProgressIndicator(modifier = Modifier.width(16.dp), color = MaterialTheme.colorScheme.onPrimary, strokeWidth = 2.dp)
            } else {
                Text("▶  محاسبه پرتفوی")
            }
        }
    }
}

@Composable
private fun LabeledNumberField(label: String, value: Double, onValueChange: (Double) -> Unit, modifier: Modifier = Modifier) {
    val colors = LocalChartColors.current
    var text by remember(value) { mutableStateOf(if (value == value.toInt().toDouble()) value.toInt().toString() else value.toString()) }
    Column(modifier = modifier.fillMaxWidth()) {
        Text(label, style = MaterialTheme.typography.labelMedium, color = colors.muted)
        OutlinedTextField(
            value = text,
            onValueChange = { text = it; it.toDoubleOrNull()?.let(onValueChange) },
            keyboardOptions = KeyboardOptions(keyboardType = KeyboardType.Number),
            modifier = Modifier.fillMaxWidth().padding(top = 4.dp),
            singleLine = true,
        )
    }
}

@Composable
private fun RiskSlider(label: String, value: Double, color: androidx.compose.ui.graphics.Color, onValueChange: (Double) -> Unit) {
    val colors = LocalChartColors.current
    Column(modifier = Modifier.padding(top = 8.dp)) {
        Row(horizontalArrangement = Arrangement.SpaceBetween, modifier = Modifier.fillMaxWidth()) {
            Text(label, style = MaterialTheme.typography.labelMedium, color = colors.textPrimary)
            Text("%.0f%%".format(value), style = MaterialTheme.typography.labelMedium, color = color)
        }
        Slider(
            value = value.toFloat(), onValueChange = { onValueChange((it / 5).toInt() * 5.0) },
            valueRange = 0f..100f, colors = androidx.compose.material3.SliderDefaults.colors(thumbColor = color, activeTrackColor = color),
        )
    }
}
