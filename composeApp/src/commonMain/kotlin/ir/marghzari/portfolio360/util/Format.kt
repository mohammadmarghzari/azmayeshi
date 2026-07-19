package ir.marghzari.portfolio360.util

import kotlin.math.absoluteValue

/**
 * One home for number formatting so every screen renders values the same way (Latin digits,
 * thousands separators, fixed decimal counts) instead of scattering `"%.2f".format(...)` variants.
 */

/** "12.34%" — signed variant adds an explicit +. */
fun Double.pct(digits: Int = 2, signed: Boolean = false): String =
    (if (signed) "%+.${digits}f%%" else "%.${digits}f%%").format(this)

/** "$1,234.56" (or another currency marker, prefix position). */
fun Double.money(symbol: String = "$", digits: Int = 2): String =
    "$symbol%,.${digits}f".format(this)

/** "1,234,567" with no decimals — for toman/rial style amounts. */
fun Double.thousands(): String = "%,.0f".format(this)

/** "1.2K", "3.4M", "5.6B" — compact display for large values in tight tiles. */
fun Double.compact(digits: Int = 1): String {
    val a = absoluteValue
    return when {
        a >= 1_000_000_000 -> "%.${digits}fB".format(this / 1_000_000_000)
        a >= 1_000_000 -> "%.${digits}fM".format(this / 1_000_000)
        a >= 1_000 -> "%.${digits}fK".format(this / 1_000)
        else -> "%.${digits}f".format(this)
    }
}
