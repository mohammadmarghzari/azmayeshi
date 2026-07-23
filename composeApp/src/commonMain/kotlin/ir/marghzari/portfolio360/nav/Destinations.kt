package ir.marghzari.portfolio360.nav

import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.Balance
import androidx.compose.material.icons.rounded.Dashboard
import androidx.compose.material.icons.rounded.Home
import androidx.compose.material.icons.rounded.Insights
import androidx.compose.material.icons.rounded.LocalFireDepartment
import androidx.compose.material.icons.rounded.QueryStats
import androidx.compose.material.icons.rounded.ReceiptLong
import androidx.compose.material.icons.rounded.Save
import androidx.compose.material.icons.rounded.Settings
import androidx.compose.material.icons.rounded.Star
import androidx.compose.material.icons.rounded.ShowChart
import androidx.compose.material.icons.rounded.TrendingUp
import androidx.compose.ui.graphics.vector.ImageVector

/** Icons use the "Rounded" Material family (soft terminals, thinner strokes) for a look closer to iOS's SF Symbols than the default sharp "Filled" set. */
enum class Destination(val route: String, val labelFa: String, val icon: ImageVector) {
    DASHBOARD("dashboard", "داشبورد", Icons.Rounded.Home),
    ALLOCATION("allocation", "تخصیص پرتفوی", Icons.Rounded.Dashboard),
    RISK_RETURN("risk_return", "ریسک و بازده", Icons.Rounded.Insights),
    PRICE_CHART("price_chart", "نمودار قیمت", Icons.Rounded.ShowChart),
    TRANSACTIONS("transactions", "دفتر تراکنش‌ها", Icons.Rounded.ReceiptLong),
    MARKETS("markets", "بازارها", Icons.Rounded.QueryStats),
    WATCHLIST("watchlist", "واچ‌لیست", Icons.Rounded.Star),
    LIQUIDATION_HEATMAP("liquidation_heatmap", "نقشه نقدشوندگی", Icons.Rounded.LocalFireDepartment),
    STYLE_COMPARE("style_compare", "مقایسه سبک‌ها", Icons.Rounded.Balance),
    EFFICIENT_FRONTIER("efficient_frontier", "Efficient Frontier", Icons.Rounded.TrendingUp),
    SAVE_PORTFOLIO("save_portfolio", "ذخیره پرتفوی", Icons.Rounded.Save),
    SETTINGS("settings", "تنظیمات", Icons.Rounded.Settings),
}
