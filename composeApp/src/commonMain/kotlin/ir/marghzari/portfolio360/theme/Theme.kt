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

/** Brand palette: clean light fintech + violet accent, per the user's "finvest" dashboard reference. */
private val VIOLET = Color(0xFF7C3AED)
private val VIOLET_SOFT = Color(0xFFA78BFA)
private val PAGE_GRAY = Color(0xFFF4F4F6)
private val CARD_WHITE = Color(0xFFFFFFFF)
private val INK = Color(0xFF17171C)

val LightBlueprint = BlueprintColors(
    bg = PAGE_GRAY, bg2 = Color(0xFFECECF1), panel = Color(0xFFE9E9EF), card = CARD_WHITE,
    accent = INK, accent2 = Color(0xFF55555F), gold = Color(0xFFB7791F),
    green = Color(0xFF16A34A), red = Color(0xFFDC2626),
    textPrimary = INK, silver = Color(0xFF4B4B55), muted = Color(0xFF8A8A94),
    sidebarBg = CARD_WHITE,
    plotBg = CARD_WHITE, plotGrid = Color(0x14202030), plotTick = Color(0xFF6B6B75), plotText = Color(0xFF2A2A32),
    riskGeo = Color(0xFFB45309), riskMon = Color(0xFF1D4ED8), riskSys = Color(0xFF9333EA),
    blueAccent = VIOLET,
    isDark = false,
)

val DarkBlueprint = BlueprintColors(
    bg = Color(0xFF0E0E12), bg2 = Color(0xFF16161C), panel = Color(0xFF181820), card = Color(0xFF1C1C24),
    accent = Color(0xFFB4B4C4), accent2 = Color(0xFF8A8AA2), gold = Color(0xFFD9B44A),
    green = Color(0xFF4ADE80), red = Color(0xFFF87171),
    textPrimary = Color(0xFFF7F7FC), silver = Color(0xFF9494A6), muted = Color(0xFF70707E),
    sidebarBg = Color(0xFF0E0E12),
    plotBg = Color(0xFF16161C), plotGrid = Color(0x14FFFFFF), plotTick = Color(0xFF8A8AA2), plotText = Color(0xFFC6C6D2),
    riskGeo = Color(0xFFE8945A), riskMon = Color(0xFF5A9BE8), riskSys = Color(0xFFB07AD4),
    blueAccent = VIOLET_SOFT,
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
