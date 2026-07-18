package ir.marghzari.portfolio360.ui.motion

import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.geometry.CornerRadius
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.rotate
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import kotlin.math.sin

/**
 * A slowly rotating, gently pulsing gradient ring around an element's border — used for the
 * active tab, a selected asset, a focused/hero card, or a selected dropdown value. Reserved for
 * "this one thing matters right now" rather than every card, so it stays a meaningful signal
 * instead of visual noise.
 */
@Composable
fun Modifier.energyRing(
    active: Boolean = true,
    colors: List<Color> = listOf(Color(0xFF5DD62C), Color(0xFF337418)),
    strokeWidth: Dp = 1.6.dp,
    cornerRadius: Dp = 16.dp,
): Modifier {
    if (!active) return this
    val clock = LocalMotionClock.current ?: return this
    val reduced = LocalReducedMotion.current

    return this.drawWithContent {
        drawContent()
        if (reduced) return@drawWithContent
        val strokePx = strokeWidth.toPx()
        val insetPx = strokePx / 2f + 1.dp.toPx()
        val pulse = 0.40f + 0.25f * (sin(clock.time.value * 6.2831855f * 3f) * 0.5f + 0.5f)
        rotate(degrees = clock.ringRotation.value) {
            drawRoundRect(
                brush = Brush.sweepGradient(colors + colors.first()),
                topLeft = Offset(insetPx, insetPx),
                size = Size(size.width - insetPx * 2, size.height - insetPx * 2),
                cornerRadius = CornerRadius(cornerRadius.toPx()),
                style = Stroke(width = strokePx),
                alpha = pulse,
            )
        }
    }
}
