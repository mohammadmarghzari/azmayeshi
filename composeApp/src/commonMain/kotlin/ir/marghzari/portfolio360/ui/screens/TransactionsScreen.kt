package ir.marghzari.portfolio360.ui.screens

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.DeleteOutline
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import ir.marghzari.portfolio360.state.AppState
import ir.marghzari.portfolio360.state.Transaction
import ir.marghzari.portfolio360.theme.LocalChartColors
import ir.marghzari.portfolio360.ui.components.AppButton
import ir.marghzari.portfolio360.ui.components.AppChip
import ir.marghzari.portfolio360.ui.components.AppTextField
import ir.marghzari.portfolio360.ui.components.AssetRow
import ir.marghzari.portfolio360.ui.components.Card
import ir.marghzari.portfolio360.ui.components.EmptyState
import ir.marghzari.portfolio360.ui.components.HeroMetric
import ir.marghzari.portfolio360.ui.components.ScreenHeader
import ir.marghzari.portfolio360.ui.components.SectionHeader
import ir.marghzari.portfolio360.ui.motion.StaggerIn
import ir.marghzari.portfolio360.util.money
import ir.marghzari.portfolio360.util.pct

/** Net quantity + weighted average buy price for one symbol of the journal. */
private data class Position(val symbol: String, val netQty: Double, val avgBuyPrice: Double?)

private fun positionsOf(txs: List<Transaction>): List<Position> =
    txs.groupBy { it.symbol.trim().uppercase() }.map { (symbol, rows) ->
        val bought = rows.filter { it.isBuy }
        val netQty = rows.sumOf { if (it.isBuy) it.quantity else -it.quantity }
        val boughtQty = bought.sumOf { it.quantity }
        Position(
            symbol = symbol,
            netQty = netQty,
            avgBuyPrice = if (boughtQty > 0) bought.sumOf { it.quantity * it.price } / boughtQty else null,
        )
    }.sortedByDescending { (it.avgBuyPrice ?: 0.0) * it.netQty }

/**
 * Manual trade journal: record real buys/sells, see net positions with average cost, and — when
 * the symbol also exists in the downloaded portfolio data — unrealized P&L against the latest
 * close. Session-scoped (in-memory) like saved portfolios; persistence is a later milestone.
 */
@Composable
fun TransactionsScreen(appState: AppState) {
    val colors = LocalChartColors.current

    var symbol by remember { mutableStateOf("") }
    var isBuy by remember { mutableStateOf(true) }
    var qtyText by remember { mutableStateOf("") }
    var priceText by remember { mutableStateOf("") }
    var dateText by remember { mutableStateOf("") }

    val txs = appState.transactions
    val qty = qtyText.toDoubleOrNull()
    val price = priceText.toDoubleOrNull()
    val formValid = symbol.isNotBlank() && qty != null && qty > 0 && price != null && price > 0

    fun lastClose(sym: String): Double? = appState.prices?.let { p ->
        p.tickers.firstOrNull { it.equals(sym, ignoreCase = true) }?.let { t -> p.column(t).lastOrNull() }
    }

    LazyColumn(verticalArrangement = Arrangement.spacedBy(16.dp), modifier = Modifier.fillMaxWidth()) {
        item {
            ScreenHeader(
                "دفتر تراکنش‌ها",
                "خرید و فروش‌های واقعی خود را ثبت کنید تا موقعیت خالص، میانگین خرید و سود/زیان هر نماد را ببینید.",
            )
        }

        if (txs.isNotEmpty()) {
            item {
                val invested = txs.sumOf { if (it.isBuy) it.quantity * it.price else -it.quantity * it.price }
                Card(modifier = Modifier.fillMaxWidth(), highlighted = true) {
                    HeroMetric(
                        label = "سرمایه خالص واردشده",
                        value = invested.money(),
                        delta = "${txs.size} تراکنش",
                    )
                }
            }
        }

        item {
            Card(modifier = Modifier.fillMaxWidth()) {
                Text("ثبت تراکنش جدید", style = MaterialTheme.typography.titleSmall, color = colors.textPrimary)
                Row(
                    horizontalArrangement = Arrangement.spacedBy(8.dp),
                    modifier = Modifier.padding(top = 10.dp),
                ) {
                    AppChip("خرید", selected = isBuy, onClick = { isBuy = true })
                    AppChip("فروش", selected = !isBuy, onClick = { isBuy = false })
                }
                AppTextField(
                    value = symbol, onValueChange = { symbol = it },
                    label = "نماد (مثل BTC-USD)",
                    modifier = Modifier.fillMaxWidth().padding(top = 8.dp),
                )
                Row(horizontalArrangement = Arrangement.spacedBy(10.dp), modifier = Modifier.padding(top = 8.dp)) {
                    AppTextField(
                        value = qtyText, onValueChange = { qtyText = it },
                        label = "تعداد", isError = qtyText.isNotBlank() && (qty == null || qty <= 0),
                        modifier = Modifier.weight(1f),
                    )
                    AppTextField(
                        value = priceText, onValueChange = { priceText = it },
                        label = "قیمت واحد ($)", isError = priceText.isNotBlank() && (price == null || price <= 0),
                        modifier = Modifier.weight(1f),
                    )
                }
                AppTextField(
                    value = dateText, onValueChange = { dateText = it },
                    label = "تاریخ (اختیاری)", placeholder = "مثلاً 2026-07-20",
                    modifier = Modifier.fillMaxWidth().padding(top = 8.dp),
                )
                AppButton(
                    text = if (isBuy) "ثبت خرید" else "ثبت فروش",
                    enabled = formValid,
                    onClick = {
                        val nextId = (txs.maxOfOrNull { it.id } ?: 0L) + 1
                        appState.transactions = txs + Transaction(
                            id = nextId,
                            symbol = symbol.trim().uppercase(),
                            isBuy = isBuy,
                            quantity = qty!!,
                            price = price!!,
                            dateLabel = dateText.trim(),
                        )
                        qtyText = ""; priceText = ""; dateText = ""
                    },
                    modifier = Modifier.fillMaxWidth().padding(top = 12.dp),
                )
            }
        }

        if (txs.isEmpty()) {
            item {
                EmptyState(
                    title = "هنوز تراکنشی ثبت نشده",
                    hint = "اولین خرید یا فروش خود را با فرم بالا ثبت کنید؛ موقعیت خالص و سود/زیان همین‌جا ساخته می‌شود.",
                )
            }
        } else {
            item {
                SectionHeader("موقعیت‌های خالص")
                Card(modifier = Modifier.fillMaxWidth()) {
                    positionsOf(txs).forEachIndexed { idx, pos ->
                        val last = lastClose(pos.symbol)
                        val unrealizedPct = if (last != null && pos.avgBuyPrice != null && pos.avgBuyPrice > 0 && pos.netQty > 0) {
                            (last / pos.avgBuyPrice - 1) * 100
                        } else null
                        StaggerIn(index = idx) {
                            AssetRow(
                                symbol = pos.symbol,
                                title = pos.symbol,
                                caption = pos.avgBuyPrice?.let { "میانگین خرید ${it.money()}" },
                                value = "%,.4f واحد".format(pos.netQty),
                                delta = unrealizedPct?.pct(2, signed = true),
                                deltaPositive = unrealizedPct?.let { it >= 0 },
                            )
                        }
                    }
                    Text(
                        "سود/زیان بر اساس آخرین قیمت دانلودشده در تب «تخصیص پرتفوی» محاسبه می‌شود؛ نمادهایی که آنجا نیستند فقط موقعیت خالص دارند.",
                        style = MaterialTheme.typography.labelSmall, color = colors.muted,
                        modifier = Modifier.padding(top = 8.dp),
                    )
                }
            }

            item {
                SectionHeader("تاریخچه تراکنش‌ها")
                Card(modifier = Modifier.fillMaxWidth()) {
                    txs.sortedByDescending { it.id }.forEachIndexed { idx, t ->
                        StaggerIn(index = idx) {
                            Row(modifier = Modifier.fillMaxWidth()) {
                                Box(modifier = Modifier.weight(1f)) {
                                    AssetRow(
                                        symbol = t.symbol,
                                        title = t.symbol,
                                        caption = listOfNotNull(
                                            if (t.isBuy) "خرید" else "فروش",
                                            "%,.4f × %s".format(t.quantity, t.price.money()),
                                            t.dateLabel.ifBlank { null },
                                        ).joinToString(" · "),
                                        value = (t.quantity * t.price).money(),
                                        delta = if (t.isBuy) "خرید" else "فروش",
                                        deltaPositive = t.isBuy,
                                    )
                                }
                                IconButton(onClick = { appState.transactions = txs - t }) {
                                    Icon(Icons.Rounded.DeleteOutline, contentDescription = "حذف تراکنش", tint = colors.red)
                                }
                            }
                        }
                    }
                }
            }
        }
    }
}
