package ir.marghzari.portfolio360.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Switch
import androidx.compose.material3.SwitchDefaults
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import ir.marghzari.portfolio360.core.model.HistoryPeriod
import ir.marghzari.portfolio360.core.model.PortfolioStyle
import ir.marghzari.portfolio360.state.AppState
import ir.marghzari.portfolio360.theme.LocalChartColors
import ir.marghzari.portfolio360.ui.components.AppButton
import ir.marghzari.portfolio360.ui.components.AppDialog
import ir.marghzari.portfolio360.ui.components.AppTextField
import ir.marghzari.portfolio360.ui.components.ButtonStyle
import ir.marghzari.portfolio360.ui.components.Card
import ir.marghzari.portfolio360.ui.components.ScreenHeader
import ir.marghzari.portfolio360.ui.components.SectionHeader
import ir.marghzari.portfolio360.ui.components.SimpleDropdown

/** One labeled toggle row of the settings list. */
@Composable
private fun SettingToggleRow(title: String, subtitle: String, checked: Boolean, onToggle: (Boolean) -> Unit) {
    val colors = LocalChartColors.current
    Row(
        modifier = Modifier.fillMaxWidth().padding(vertical = 6.dp),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Column(modifier = Modifier.weight(1f)) {
            Text(title, style = MaterialTheme.typography.bodyMedium, color = colors.textPrimary)
            Text(subtitle, style = MaterialTheme.typography.labelSmall, color = colors.muted)
        }
        Switch(
            checked = checked,
            onCheckedChange = onToggle,
            colors = SwitchDefaults.colors(
                checkedTrackColor = colors.blueAccent,
                checkedThumbColor = androidx.compose.ui.graphics.Color.White,
            ),
        )
    }
}

/**
 * Settings tab: display preferences, calculation defaults and data management in grouped
 * design-system cards. Every preference here is already covered by AppPersistence, so a change
 * survives app restarts without extra wiring.
 */
@Composable
fun SettingsScreen(appState: AppState) {
    val colors = LocalChartColors.current
    var confirmClear by remember { mutableStateOf<String?>(null) }
    var rfText by remember { mutableStateOf(appState.riskFreeRatePct.toString().removeSuffix(".0")) }

    LazyColumn(verticalArrangement = Arrangement.spacedBy(16.dp), modifier = Modifier.fillMaxWidth()) {
        item {
            ScreenHeader("تنظیمات", "نمایش، پیش‌فرض‌های محاسبه و مدیریت داده‌ها — همه تغییرات به‌صورت خودکار ذخیره می‌شوند.")
        }

        item {
            SectionHeader("نمایش")
            Card(modifier = Modifier.fillMaxWidth()) {
                SettingToggleRow(
                    title = "تم تیره",
                    subtitle = "پیش‌فرض حرفه‌ای اپ؛ تم روشن برای محیط‌های پرنور",
                    checked = appState.isDarkTheme,
                    onToggle = { appState.isDarkTheme = it },
                )
                SettingToggleRow(
                    title = "کاهش انیمیشن",
                    subtitle = "خاموش‌کردن حرکت‌های تزئینی برای تمرکز یا صرفه‌جویی باتری",
                    checked = appState.reducedMotion,
                    onToggle = { appState.reducedMotion = it },
                )
            }
        }

        item {
            SectionHeader("پیش‌فرض‌های محاسبه")
            Card(modifier = Modifier.fillMaxWidth()) {
                AppTextField(
                    value = rfText,
                    onValueChange = { rfText = it; it.toDoubleOrNull()?.let { v -> appState.riskFreeRatePct = v } },
                    label = "نرخ بدون ریسک (%)",
                    isError = rfText.toDoubleOrNull() == null,
                    modifier = Modifier.fillMaxWidth(),
                )
                SimpleDropdown(
                    label = "بازه زمانی داده",
                    selected = HistoryPeriod.entries.firstOrNull { it.apiCode == appState.periodCode } ?: HistoryPeriod.Y2,
                    options = HistoryPeriod.entries,
                    optionLabel = { it.faLabel },
                    onSelected = { appState.periodCode = it.apiCode },
                    modifier = Modifier.padding(top = 10.dp),
                )
                SimpleDropdown(
                    label = "روش بهینه‌سازی پیش‌فرض",
                    selected = appState.portfolioStyle,
                    options = PortfolioStyle.entries,
                    optionLabel = { it.faLabel },
                    onSelected = { appState.portfolioStyle = it },
                    modifier = Modifier.padding(top = 10.dp),
                )
            }
        }

        item {
            SectionHeader("داده‌ها")
            Card(modifier = Modifier.fillMaxWidth()) {
                Column(verticalArrangement = Arrangement.spacedBy(8.dp)) {
                    AppButton(
                        "پاک‌کردن دفتر تراکنش‌ها", style = ButtonStyle.Destructive,
                        onClick = { confirmClear = "journal" }, modifier = Modifier.fillMaxWidth(),
                    )
                    AppButton(
                        "پاک‌کردن واچ‌لیست", style = ButtonStyle.Destructive,
                        onClick = { confirmClear = "watchlist" }, modifier = Modifier.fillMaxWidth(),
                    )
                    AppButton(
                        "پاک‌کردن همه داده‌ها", style = ButtonStyle.Destructive,
                        onClick = { confirmClear = "all" }, modifier = Modifier.fillMaxWidth(),
                    )
                }
            }
        }

        item {
            SectionHeader("درباره")
            Card(modifier = Modifier.fillMaxWidth()) {
                Text("Portfolio360", style = MaterialTheme.typography.titleMedium, color = colors.textPrimary)
                Text("نسخه 1.0.0", style = MaterialTheme.typography.labelSmall, color = colors.muted, modifier = Modifier.padding(top = 4.dp))
                Text(
                    "Kotlin Multiplatform · Compose Multiplatform · اندروید و ویندوز از یک کد مشترک",
                    style = MaterialTheme.typography.labelSmall, color = colors.muted,
                    modifier = Modifier.padding(top = 2.dp),
                )
            }
        }
    }

    confirmClear?.let { target ->
        AppDialog(
            title = when (target) {
                "journal" -> "پاک‌کردن دفتر تراکنش‌ها؟"
                "watchlist" -> "پاک‌کردن واچ‌لیست؟"
                else -> "پاک‌کردن همه داده‌ها؟"
            },
            onDismiss = { confirmClear = null },
            confirmText = "بله، پاک کن",
            onConfirm = {
                when (target) {
                    "journal" -> appState.transactions = emptyList()
                    "watchlist" -> appState.favoriteTickers = emptySet()
                    else -> {
                        appState.transactions = emptyList()
                        appState.favoriteTickers = emptySet()
                        appState.selectedTickers = emptyList()
                        appState.priceAlerts = emptyList()
                        appState.savedPortfolios = emptyMap()
                        appState.resetComputedPortfolio()
                        appState.prices = null
                    }
                }
                confirmClear = null
            },
            dismissText = "انصراف",
        ) {
            Text("این عمل برگشت‌پذیر نیست و داده حذف‌شده قابل بازیابی نخواهد بود.")
        }
    }
}
