package ir.marghzari.portfolio360.core.math

import kotlinx.datetime.Clock
import kotlinx.datetime.LocalDate
import kotlinx.datetime.TimeZone
import kotlinx.datetime.daysUntil
import kotlinx.datetime.todayIn
import kotlin.math.max

/** Iran Mercantile Exchange (بورس کالا) commodity-options helpers: Shamsi dates, Rial formatting, combined strategies. */
object BourseStrategies {

    data class Underlying(val key: String, val labelFa: String, val symbol: String, val unitFa: String, val lotSize: Int, val descriptionFa: String, val icon: String)

    val UNDERLYINGS = listOf(
        Underlying("TLDY", "صندوق طلای لوتوس", "TLDY", "ریال", 5000, "صندوق سرمایه‌گذاری طلای لوتوس", "🪙"),
        Underlying("GOLD", "شمش طلا", "GOLD", "ریال/گرم", 100, "گواهی سپرده شمش طلا", "🥇"),
        Underlying("KHRB", "صندوق کهربا ۱۰", "KHRB", "ریال", 5000, "صندوق سرمایه‌گذاری کهربا", "🟡"),
        Underlying("ZNAY", "صندوق طلای دنای زاگرس", "ZNAY", "ریال", 5000, "صندوق سرمایه‌گذاری طلای دنای زاگرس", "🟨"),
        Underlying("SILV", "شمش نقره", "SILV", "ریال/گرم", 1000, "گواهی سپرده شمش نقره", "🥈"),
        Underlying("DRSH", "صندوق طلای درخشان", "DRSH", "ریال", 5000, "صندوق سرمایه‌گذاری طلای درخشان", "✨"),
    )

    fun formatRial(v: Double): String = when {
        v >= 1e9 -> "%,.2f میلیارد".format(v / 1e9)
        v >= 1e6 -> "%,.1f میلیون".format(v / 1e6)
        else -> "%,.0f".format(v)
    }

    /** Approximate Jalali(Shamsi) "YYYY/MM/DD" (or "-") -> days-to-expiry from today. Falls back to 30 on any parse error. */
    fun shamsiToDaysFromToday(shamsiDate: String, today: LocalDate = Clock.System.todayIn(TimeZone.currentSystemDefault())): Int {
        return try {
            val parts = shamsiDate.replace('-', '/').split('/').map { it.trim().toInt() }
            val (y, m, d) = Triple(parts[0], parts[1], parts[2])
            var gy = y + 621
            val monthMap = mapOf(1 to 3, 2 to 4, 3 to 5, 4 to 6, 5 to 7, 6 to 8, 7 to 9, 8 to 10, 9 to 11, 10 to 12, 11 to 1, 12 to 2)
            val gm = monthMap.getValue(m)
            if (m >= 11) gy += 1
            val expiry = LocalDate(gy, gm, d.coerceIn(1, 28))
            val days = today.daysUntil(expiry)
            max(days, 0)
        } catch (e: Exception) {
            30
        }
    }

    // ---- Combined strategy payoffs (all "at expiry", per the ⑦ section of the bourse options tab) ----

    fun bullCallSpreadPnl(S: Double, kLow: Double, kHigh: Double, netDebit: Double, lot: Int, n: Int): Double =
        (max(S - kLow, 0.0) - max(S - kHigh, 0.0) - netDebit) * lot * n

    fun bearPutSpreadPnl(S: Double, kHigh: Double, kLow: Double, netDebit: Double, lot: Int, n: Int): Double =
        (max(kHigh - S, 0.0) - max(kLow - S, 0.0) - netDebit) * lot * n

    fun straddlePnl(S: Double, K: Double, totalPremium: Double, lot: Int, n: Int): Double =
        (max(S - K, 0.0) + max(K - S, 0.0) - totalPremium) * lot * n

    fun stranglePnl(S: Double, kPut: Double, kCall: Double, totalPremium: Double, lot: Int, n: Int): Double =
        (max(kPut - S, 0.0) + max(S - kCall, 0.0) - totalPremium) * lot * n

    fun coveredCallPnl(S: Double, spot: Double, kCall: Double, premiumReceived: Double, units: Int): Double =
        (S - spot) * units + (premiumReceived - max(S - kCall, 0.0)) * units

    fun protectivePutPnl(S: Double, spot: Double, kPut: Double, premiumPaid: Double, units: Int): Double =
        (S - spot) * units + max(kPut - S, 0.0) * units - premiumPaid * units
}
