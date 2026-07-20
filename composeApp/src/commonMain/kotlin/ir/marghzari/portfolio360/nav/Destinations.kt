package ir.marghzari.portfolio360.nav

import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.AccountBalance
import androidx.compose.material.icons.rounded.Balance
import androidx.compose.material.icons.rounded.Dashboard
import androidx.compose.material.icons.rounded.Inventory2
import androidx.compose.material.icons.rounded.Insights
import androidx.compose.material.icons.rounded.Language
import androidx.compose.material.icons.rounded.Notifications
import androidx.compose.material.icons.rounded.Psychology
import androidx.compose.material.icons.rounded.QueryStats
import androidx.compose.material.icons.rounded.ReceiptLong
import androidx.compose.material.icons.rounded.Public
import androidx.compose.material.icons.rounded.Save
import androidx.compose.material.icons.rounded.Star
import androidx.compose.material.icons.rounded.SatelliteAlt
import androidx.compose.material.icons.rounded.ShowChart
import androidx.compose.material.icons.rounded.Timeline
import androidx.compose.material.icons.rounded.TrendingUp
import androidx.compose.material.icons.rounded.Whatshot
import androidx.compose.ui.graphics.vector.ImageVector

/** Icons use the "Rounded" Material family (soft terminals, thinner strokes) for a look closer to iOS's SF Symbols than the default sharp "Filled" set. */
enum class Destination(val route: String, val labelFa: String, val icon: ImageVector) {
    ALLOCATION("allocation", "تخصیص پرتفوی", Icons.Rounded.Dashboard),
    RISK_RETURN("risk_return", "ریسک و بازده", Icons.Rounded.Insights),
    PRICE_CHART("price_chart", "نمودار قیمت", Icons.Rounded.ShowChart),
    TRANSACTIONS("transactions", "دفتر تراکنش‌ها", Icons.Rounded.ReceiptLong),
    MARKETS("markets", "بازارها", Icons.Rounded.QueryStats),
    WATCHLIST("watchlist", "واچ‌لیست", Icons.Rounded.Star),
    STYLE_COMPARE("style_compare", "مقایسه سبک‌ها", Icons.Rounded.Balance),
    EFFICIENT_FRONTIER("efficient_frontier", "Efficient Frontier", Icons.Rounded.TrendingUp),
    ADVANCED_OPTIONS("advanced_options", "اختیار پیشرفته", Icons.Rounded.Timeline),
    BLACK_LITTERMAN("black_litterman", "Black-Litterman", Icons.Rounded.Psychology),
    STRESS_MC("stress_mc", "Stress Test & MC", Icons.Rounded.Whatshot),
    REBALANCE("rebalance", "ری‌بالانس", Icons.Rounded.Balance),
    BENCHMARK("benchmark", "Benchmark", Icons.Rounded.Insights),
    LIVE_DATA("live_data", "داده زنده", Icons.Rounded.Language),
    SAVE_PORTFOLIO("save_portfolio", "ذخیره پرتفوی", Icons.Rounded.Save),
    ALERTS("alerts", "هشدار", Icons.Rounded.Notifications),
    IRAN_TOOLS("iran_tools", "ابزار ایران", Icons.Rounded.Public),
    CERTIFICATES("certificates", "گواهی سپرده کالایی", Icons.Rounded.Inventory2),
    BOURSE_OPTIONS("bourse_options", "اختیار بورس کالا", Icons.Rounded.AccountBalance),
    IME_LIVE("ime_live", "IME Live", Icons.Rounded.SatelliteAlt),
}
