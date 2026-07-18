package ir.marghzari.portfolio360.ui.screens

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.Button
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.Slider
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import ir.marghzari.portfolio360.core.model.SymbolCatalog
import ir.marghzari.portfolio360.core.network.YahooFinanceClient
import ir.marghzari.portfolio360.state.AppState
import ir.marghzari.portfolio360.state.PriceAlert
import ir.marghzari.portfolio360.theme.LocalChartColors
import ir.marghzari.portfolio360.ui.components.Card
import ir.marghzari.portfolio360.ui.components.InfoBanner
import ir.marghzari.portfolio360.ui.components.SectionHeader
import ir.marghzari.portfolio360.ui.components.SimpleDropdown
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext

@Composable
fun AlertsScreen(appState: AppState) {
    val colors = LocalChartColors.current
    val scope = rememberCoroutineScope()

    var symbol by remember { mutableStateOf(SymbolCatalog.ALL_TICKERS.first()) }
    var isAbove by remember { mutableStateOf(true) }
    var thresholdText by remember { mutableStateOf("100") }
    var checking by remember { mutableStateOf(false) }

    LazyColumn(verticalArrangement = Arrangement.spacedBy(16.dp), modifier = Modifier.fillMaxWidth()) {
        item {
            SectionHeader("🔔 هشدار قیمت")
            Text("قیمت فعلی نمادها را با آستانه‌های تعریف‌شده مقایسه می‌کند.", style = MaterialTheme.typography.bodySmall, color = colors.muted)
            Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.padding(top = 10.dp)) {
                SimpleDropdown("نماد", symbol, SymbolCatalog.ALL_TICKERS, { it }, { symbol = it }, modifier = Modifier.weight(1f))
                SimpleDropdown("نوع", isAbove, listOf(true, false), { if (it) "بالاتر از" else "پایین‌تر از" }, { isAbove = it }, modifier = Modifier.weight(1f))
            }
            OutlinedTextField(
                value = thresholdText, onValueChange = { thresholdText = it }, modifier = Modifier.fillMaxWidth().padding(top = 10.dp),
                label = { Text("قیمت ($)") }, singleLine = true,
            )
            Button(
                onClick = {
                    val price = thresholdText.toDoubleOrNull() ?: return@Button
                    appState.priceAlerts = appState.priceAlerts + PriceAlert(symbol, isAbove, price)
                },
                modifier = Modifier.fillMaxWidth().padding(top = 10.dp), colors = ButtonDefaults.buttonColors(containerColor = colors.blueAccent),
            ) { Text("➕ افزودن هشدار") }
        }

        if (appState.priceAlerts.isNotEmpty()) {
            item {
                SectionHeader("بررسی هشدارها")
                Button(
                    onClick = {
                        scope.launch {
                            checking = true
                            val yahoo = appState.yahoo
                            val updated = appState.priceAlerts.map { al ->
                                val price = withContext(Dispatchers.Default) { yahoo.fetchLastClose(al.symbol, ttlMs = 0) }
                                if (price != null) {
                                    al.copy(currentPrice = price, triggered = if (al.isAbove) price > al.threshold else price < al.threshold)
                                } else al
                            }
                            appState.priceAlerts = updated
                            checking = false
                        }
                    },
                    modifier = Modifier.fillMaxWidth(), colors = ButtonDefaults.buttonColors(containerColor = colors.gold),
                ) { Text(if (checking) "در حال دریافت قیمت‌ها..." else "🔄 بررسی قیمت‌های فعلی") }
            }
            items(appState.priceAlerts) { al ->
                val borderColor = if (al.triggered) colors.red else if (al.currentPrice != null) colors.green else colors.muted
                Row(
                    modifier = Modifier.fillMaxWidth()
                        .background(borderColor.copy(alpha = 0.08f), RoundedCornerShape(10.dp))
                        .padding(12.dp),
                    horizontalArrangement = Arrangement.SpaceBetween,
                ) {
                    Text("${al.symbol}  ${if (al.isAbove) "بالاتر از" else "پایین‌تر از"} $${"%.2f".format(al.threshold)}", color = colors.textPrimary)
                    val pctDiff = al.currentPrice?.let { (it - al.threshold) / al.threshold * 100 }
                    Text(
                        if (al.triggered) "🔔 فعال" else if (al.currentPrice != null) "✓ ${"%+.2f".format(pctDiff)}%" else "—",
                        color = borderColor,
                    )
                    Button(onClick = { appState.priceAlerts = appState.priceAlerts - al }, colors = ButtonDefaults.buttonColors(containerColor = colors.red)) {
                        Text("🗑")
                    }
                }
            }
        } else {
            item { InfoBanner("هنوز هشداری تعریف نشده.") }
        }

        item {
            SectionHeader("🔔 هشدار Fear & Greed")
            Text("اگر شاخص ترس و طمع از آستانه‌های تعریف‌شده رد شود، هشدار نمایش می‌دهد.", style = MaterialTheme.typography.bodySmall, color = colors.muted)
            var lower by remember { mutableStateOf(appState.fearGreedAlertConfig.lowerBound.toFloat()) }
            var upper by remember { mutableStateOf(appState.fearGreedAlertConfig.upperBound.toFloat()) }
            Text("هشدار ترس شدید (زیر این مقدار): ${lower.toInt()}", style = MaterialTheme.typography.labelMedium)
            Slider(value = lower, onValueChange = { lower = it; appState.fearGreedAlertConfig = appState.fearGreedAlertConfig.copy(lowerBound = it.toInt()) }, valueRange = 0f..50f)
            Text("هشدار طمع شدید (بالای این مقدار): ${upper.toInt()}", style = MaterialTheme.typography.labelMedium)
            Slider(value = upper, onValueChange = { upper = it; appState.fearGreedAlertConfig = appState.fearGreedAlertConfig.copy(upperBound = it.toInt()) }, valueRange = 50f..100f)

            var status by remember { mutableStateOf<String?>(null) }
            var statusColor by remember { mutableStateOf(colors.muted) }
            Button(
                onClick = {
                    scope.launch {
                        val fg = appState.fearGreed.fetch()
                        if (fg != null) {
                            appState.fearGreedAlertConfig = appState.fearGreedAlertConfig.copy(lastScore = fg.score)
                            when {
                                fg.score <= lower -> { status = "😱 هشدار ترس شدید! شاخص = ${fg.score.toInt()}"; statusColor = colors.red }
                                fg.score >= upper -> { status = "🤑 هشدار طمع شدید! شاخص = ${fg.score.toInt()}"; statusColor = colors.gold }
                                else -> { status = "✓ شاخص در محدوده عادی: ${fg.score.toInt()}"; statusColor = colors.green }
                            }
                        } else {
                            status = "دریافت شاخص ناموفق بود."
                            statusColor = colors.red
                        }
                    }
                },
                modifier = Modifier.fillMaxWidth().padding(top = 10.dp), colors = ButtonDefaults.buttonColors(containerColor = colors.blueAccent),
            ) { Text("🔄 بررسی Fear & Greed اکنون") }
            status?.let { Text(it, color = statusColor, modifier = Modifier.padding(top = 8.dp)) }
        }
    }
}
