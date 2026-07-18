package ir.marghzari.portfolio360.ui.motion

import androidx.compose.ui.graphics.Color
import ir.marghzari.portfolio360.core.math.IranTools

private val CRYPTO_GLOW = listOf(Color(0xFF5DD62C), Color(0xFF337418))
private val GOLD_GLOW = listOf(Color(0xFFE8C87A), Color(0xFFC9A227))
private val SILVER_GLOW = listOf(Color(0xFFD8DCE6), Color(0xFFA9B0BD))
private val BRONZE_GLOW = listOf(Color(0xFFD98A4A), Color(0xFF8A5A2A))

/**
 * Maps a commodity to the particle/glow palette it should use — warm gold, cool silver, bronze
 * copper, or the default green crypto/brand glow for anything else (stocks, crypto tickers,
 * industrial metals with no precious-metal analog).
 */
fun motionColorsFor(key: IranTools.CommodityKey): List<Color> = when (key) {
    IranTools.CommodityKey.GoldBar, IranTools.CommodityKey.GoldCoin -> GOLD_GLOW
    IranTools.CommodityKey.SilverBar -> SILVER_GLOW
    IranTools.CommodityKey.CopperCthd -> BRONZE_GLOW
    else -> CRYPTO_GLOW
}
