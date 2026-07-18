package ir.marghzari.portfolio360.ui.motion

import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.drawBehind
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import kotlin.math.cos
import kotlin.math.sin

/**
 * Ultra-subtle orbiting dots traced along an ellipse just inside the element's edge, so they never
 * cross the center where text/values sit. Each particle gets its own phase and speed derived from
 * [seed] so identical widgets don't move in lockstep. No-ops when [LocalReducedMotion] is on or
 * when no [MotionClock] has been installed (e.g. an isolated preview).
 */
@Composable
fun Modifier.orbitParticles(
    colors: List<Color> = listOf(Color(0xFF7C5CF6), Color(0xFF4E6BF2)),
    count: Int = 4,
    seed: Any? = null,
): Modifier {
    val clock = LocalMotionClock.current ?: return this
    if (LocalReducedMotion.current) return this
    val baseSeed = seedFrom(seed)

    return this.drawBehind {
        val cx = size.width / 2f
        val cy = size.height / 2f
        val rx = size.width / 2f * 0.95f
        val ry = size.height / 2f * 0.95f
        val dotRadius = 1.6.dp.toPx()

        for (i in 0 until count) {
            val phase = (baseSeed + i / count.toFloat()) % 1f
            val speed = 0.6f + 0.18f * i
            val t = (clock.time.value * speed + phase) % 1f
            val angle = t * 6.2831855f
            val x = cx + cos(angle) * rx
            val y = cy + sin(angle) * ry
            val twinkle = sin(angle * 2f + phase * 6.2831855f) * 0.5f + 0.5f
            val alpha = (0.05f + 0.05f * twinkle).coerceIn(0.05f, 0.10f)
            drawCircle(color = colors[i % colors.size], radius = dotRadius, center = Offset(x, y), alpha = alpha)
        }
    }
}
