package ir.marghzari.portfolio360.ui.branding

import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.size
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.rotate
import androidx.compose.ui.unit.dp

private val PORTFOLIO_PURPLE = Color(0xFF7C5CF6)
private val PORTFOLIO_BLUE = Color(0xFF4E6BF2)
private val PORTFOLIO_DARK = Color(0xFF20244A)

/**
 * A Compose-native (lightweight, GPU-composited) animated take on the launcher glyph — a slowly
 * rotating orbit ring with a node, a breathing glow, and ascending bars. The home-screen launcher
 * icon itself can't animate (no Android launcher renders continuous motion for a regular app's
 * icon), so this is where the "premium motion" from the icon's design language actually plays,
 * e.g. on the splash screen.
 */
@Composable
fun AnimatedAppIcon(modifier: Modifier = Modifier, size: androidx.compose.ui.unit.Dp = 96.dp) {
    val transition = rememberInfiniteTransition(label = "app-icon")
    val rotation by transition.animateFloat(
        initialValue = 0f, targetValue = 360f,
        animationSpec = infiniteRepeatable(tween(9000, easing = LinearEasing)),
        label = "icon-rotation",
    )
    val glow by transition.animateFloat(
        initialValue = 0.55f, targetValue = 1f,
        animationSpec = infiniteRepeatable(tween(1800, easing = FastOutSlowInEasing), RepeatMode.Reverse),
        label = "icon-glow",
    )
    val barBreath by transition.animateFloat(
        initialValue = 0.96f, targetValue = 1f,
        animationSpec = infiniteRepeatable(tween(2200, easing = FastOutSlowInEasing), RepeatMode.Reverse),
        label = "icon-bar-breath",
    )

    Canvas(modifier = modifier.size(size)) {
        val cx = size.toPx() / 2f
        val cy = size.toPx() / 2f
        val ringRadius = size.toPx() * 0.30f

        // Background glow.
        drawCircle(
            brush = Brush.radialGradient(
                listOf(PORTFOLIO_PURPLE.copy(alpha = 0.35f * glow), Color.Transparent),
                center = Offset(cx, cy * 0.9f),
                radius = ringRadius * 1.8f,
            ),
            radius = ringRadius * 1.8f,
            center = Offset(cx, cy * 0.9f),
        )

        rotate(degrees = rotation, pivot = Offset(cx, cy)) {
            drawCircle(
                brush = Brush.sweepGradient(
                    listOf(PORTFOLIO_PURPLE, PORTFOLIO_BLUE, PORTFOLIO_PURPLE),
                    center = Offset(cx, cy),
                ),
                radius = ringRadius,
                center = Offset(cx, cy),
                style = Stroke(width = size.toPx() * 0.032f, cap = StrokeCap.Round),
            )
            val nodeAngleRad = Math.toRadians(-45.0)
            val nodeCenter = Offset(
                cx + ringRadius * kotlin.math.cos(nodeAngleRad).toFloat(),
                cy + ringRadius * kotlin.math.sin(nodeAngleRad).toFloat(),
            )
            drawCircle(color = Color(0xFFE8E4FF), radius = size.toPx() * 0.045f, center = nodeCenter)
        }

        // Ascending bars, gently breathing in height.
        val barWidth = size.toPx() * 0.11f
        val gap = size.toPx() * 0.045f
        val baseline = cy + size.toPx() * 0.22f
        val heights = listOf(size.toPx() * 0.20f, size.toPx() * 0.32f, size.toPx() * 0.44f)
        val startX = cx - (barWidth * 3 + gap * 2) / 2f
        heights.forEachIndexed { i, h ->
            val animatedH = h * barBreath
            val left = startX + i * (barWidth + gap)
            drawRect(
                brush = Brush.verticalGradient(
                    listOf(PORTFOLIO_PURPLE.copy(alpha = 0.95f), PORTFOLIO_BLUE),
                    startY = baseline - animatedH, endY = baseline,
                ),
                topLeft = Offset(left, baseline - animatedH),
                size = androidx.compose.ui.geometry.Size(barWidth, animatedH),
            )
        }
    }
}
