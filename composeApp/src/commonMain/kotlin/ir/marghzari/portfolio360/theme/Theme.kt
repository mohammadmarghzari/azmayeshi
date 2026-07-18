package ir.marghzari.portfolio360.theme

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.staticCompositionLocalOf
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.sp

/**
 * Direct port of the light/dark CSS custom-property palettes from app.py's injected `<style>` block,
 * kept as a single source of truth for both the Material3 [ColorScheme] and the custom chart/canvas code.
 */
data class BlueprintColors(
    val bg: Color, val bg2: Color, val panel: Color, val card: Color,
    val accent: Color, val accent2: Color, val gold: Color, val green: Color, val red: Color,
    val textPrimary: Color, val silver: Color, val muted: Color, val sidebarBg: Color,
    val plotBg: Color, val plotGrid: Color, val plotTick: Color, val plotText: Color,
    val riskGeo: Color, val riskMon: Color, val riskSys: Color,
    val blueAccent: Color,
    val isDark: Boolean,
)

/** Brand palette: near-black + neon green, per the user's reference (Oscillate Marketing card). */
private val NEON_GREEN = Color(0xFF5DD62C)
private val DEEP_GREEN = Color(0xFF337418)
private val NEAR_BLACK = Color(0xFF0F0F0F)
private val CARD_DARK = Color(0xFF202020)
private val OFF_WHITE = Color(0xFFF8F8F8)

val LightBlueprint = BlueprintColors(
    bg = OFF_WHITE, bg2 = Color(0xFFECECEC), panel = Color(0xFFE0E0E0), card = Color(0xFFEFEFEF),
    accent = Color(0xFF1A1A18), accent2 = Color(0xFF555550), gold = Color(0xFF8A6A1A),
    green = DEEP_GREEN, red = Color(0xFF8A2020),
    textPrimary = Color(0xFF111110), silver = Color(0xFF444440), muted = Color(0xFF888882),
    sidebarBg = Color(0xFFE6E6E6),
    plotBg = Color(0xFFECECEC), plotGrid = Color(0x1A3C3C37), plotTick = Color(0xFF444440), plotText = Color(0xFF222220),
    riskGeo = Color(0xFF7A3A00), riskMon = Color(0xFF1A4A7A), riskSys = Color(0xFF4A1A6A),
    blueAccent = NEON_GREEN,
    isDark = false,
)

val DarkBlueprint = BlueprintColors(
    bg = NEAR_BLACK, bg2 = Color(0xFF161616), panel = Color(0xFF1A1A1A), card = CARD_DARK,
    accent = Color(0xFFB0B0B0), accent2 = Color(0xFF888888), gold = Color(0xFFC8A84B),
    green = NEON_GREEN, red = Color(0xFFCC5555),
    textPrimary = OFF_WHITE, silver = Color(0xFF909090), muted = Color(0xFF666666),
    sidebarBg = NEAR_BLACK,
    plotBg = Color(0xFF161616), plotGrid = Color(0x0DFFFFFF), plotTick = Color(0xFF888888), plotText = Color(0xFFC0C0C0),
    riskGeo = Color(0xFFE8945A), riskMon = Color(0xFF5A9BE8), riskSys = Color(0xFFB07AD4),
    blueAccent = NEON_GREEN,
    isDark = true,
)

val LocalBlueprintColors = staticCompositionLocalOf { DarkBlueprint }

private val monoFontFamily = FontFamily.Monospace
private val bodyFontFamily = FontFamily.SansSerif

@Composable
fun Portfolio360Theme(darkTheme: Boolean = isSystemInDarkTheme(), content: @Composable () -> Unit) {
    val colors = if (darkTheme) DarkBlueprint else LightBlueprint
    val scheme = if (darkTheme) {
        androidx.compose.material3.darkColorScheme(
            primary = colors.blueAccent, secondary = colors.gold, tertiary = colors.green,
            background = colors.bg, surface = colors.card, error = colors.red,
            onBackground = colors.textPrimary, onSurface = colors.textPrimary,
        )
    } else {
        androidx.compose.material3.lightColorScheme(
            primary = colors.blueAccent, secondary = colors.gold, tertiary = colors.green,
            background = colors.bg, surface = colors.card, error = colors.red,
            onBackground = colors.textPrimary, onSurface = colors.textPrimary,
        )
    }
    val typography = Typography(
        titleLarge = TextStyle(fontFamily = monoFontFamily, fontWeight = FontWeight.SemiBold, fontSize = 22.sp),
        titleMedium = TextStyle(fontFamily = monoFontFamily, fontWeight = FontWeight.SemiBold, fontSize = 17.sp),
        titleSmall = TextStyle(fontFamily = monoFontFamily, fontWeight = FontWeight.Medium, fontSize = 14.sp),
        bodyLarge = TextStyle(fontFamily = bodyFontFamily, fontSize = 16.sp),
        bodyMedium = TextStyle(fontFamily = bodyFontFamily, fontSize = 14.sp),
        bodySmall = TextStyle(fontFamily = bodyFontFamily, fontSize = 12.sp),
        labelLarge = TextStyle(fontFamily = monoFontFamily, fontWeight = FontWeight.Medium, fontSize = 14.sp),
        labelMedium = TextStyle(fontFamily = monoFontFamily, fontSize = 12.sp),
        labelSmall = TextStyle(fontFamily = monoFontFamily, fontSize = 11.sp),
    )
    CompositionLocalProvider(LocalBlueprintColors provides colors) {
        MaterialTheme(colorScheme = scheme, typography = typography, content = content)
    }
}

val LocalChartColors get() = LocalBlueprintColors
