package ir.marghzari.portfolio360.nav

import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.AccountBalance
import androidx.compose.material.icons.filled.Balance
import androidx.compose.material.icons.filled.Bolt
import androidx.compose.material.icons.filled.Dashboard
import androidx.compose.material.icons.filled.Inventory2
import androidx.compose.material.icons.filled.Insights
import androidx.compose.material.icons.filled.Language
import androidx.compose.material.icons.filled.Notifications
import androidx.compose.material.icons.filled.Psychology
import androidx.compose.material.icons.filled.Public
import androidx.compose.material.icons.filled.Save
import androidx.compose.material.icons.filled.SatelliteAlt
import androidx.compose.material.icons.filled.ShowChart
import androidx.compose.material.icons.filled.Timeline
import androidx.compose.material.icons.filled.TrendingUp
import androidx.compose.material.icons.filled.Whatshot
import androidx.compose.ui.graphics.vector.ImageVector

enum class Destination(val route: String, val labelFa: String, val icon: ImageVector) {
    ALLOCATION("allocation", "تخصیص پرتفوی", Icons.Filled.Dashboard),
    RISK_RETURN("risk_return", "ریسک و بازده", Icons.Filled.Insights),
    PRICE_CHART("price_chart", "نمودار قیمت", Icons.Filled.ShowChart),
    STYLE_COMPARE("style_compare", "مقایسه سبک‌ها", Icons.Filled.Balance),
    EFFICIENT_FRONTIER("efficient_frontier", "Efficient Frontier", Icons.Filled.TrendingUp),
    ADVANCED_OPTIONS("advanced_options", "اختیار پیشرفته", Icons.Filled.Timeline),
    BLACK_LITTERMAN("black_litterman", "Black-Litterman", Icons.Filled.Psychology),
    STRESS_MC("stress_mc", "Stress Test & MC", Icons.Filled.Whatshot),
    REBALANCE("rebalance", "ری‌بالانس", Icons.Filled.Balance),
    BENCHMARK("benchmark", "Benchmark", Icons.Filled.Insights),
    LIVE_DATA("live_data", "داده زنده", Icons.Filled.Language),
    SAVE_PORTFOLIO("save_portfolio", "ذخیره پرتفوی", Icons.Filled.Save),
    ALERTS("alerts", "هشدار", Icons.Filled.Notifications),
    IRAN_TOOLS("iran_tools", "ابزار ایران", Icons.Filled.Public),
    CERTIFICATES("certificates", "گواهی سپرده کالایی", Icons.Filled.Inventory2),
    BOURSE_OPTIONS("bourse_options", "اختیار بورس کالا", Icons.Filled.AccountBalance),
    IME_LIVE("ime_live", "IME Live", Icons.Filled.SatelliteAlt),
}
