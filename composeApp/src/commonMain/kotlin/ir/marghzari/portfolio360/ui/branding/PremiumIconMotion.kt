package ir.marghzari.portfolio360.ui.branding

import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.gestures.detectDragGestures
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.size
import androidx.compose.material3.Icon
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.rotate
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.input.pointer.pointerInput
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import kotlin.math.cos
import kotlin.math.sin
import kotlin.random.Random

private val ORBIT_VIOLET = Color(0xFF7C3AED)
private val ORBIT_VIOLET_SOFT = Color(0xFFA78BFA)

private class OrbitParticle(val radiusFactor: Float, val speed: Float, val phase: Float, val color: Color)

private val orbitParticles = List(4) { i ->
    OrbitParticle(
        radiusFactor = 0.78f + Random.nextFloat() * 0.22f,
        speed = 0.55f + Random.nextFloat() * 0.5f,
        phase = Random.nextFloat() * 6.28f,
        color = if (i % 2 == 0) ORBIT_VIOLET else ORBIT_VIOLET_SOFT,
    )
}

/**
 * Wraps a nav/feature icon with a rotating energy-ring + orbiting-particle glow and a small 3D
 * tilt, reusing one animation engine for every icon that opts in. Deliberately NOT applied to the
 * main launcher icon (a separate native Android resource, not a Compose icon at all).
 *
 * Continuous ring/particle motion only runs while [active] is true (meant for "this is the
 * selected/focused icon") — animating all icons in a 16-item nav rail at once would burn battery
 * for effect nobody's looking at, so inactive icons render as a plain static [Icon] with the same
 * touch-tilt behavior, and only pick up the full motion once selected.
 */
@Composable
fun PremiumIconMotion(
    icon: ImageVector,
    contentDescription: String?,
    tint: Color,
    active: Boolean,
    modifier: Modifier = Modifier,
    iconSize: Dp = 24.dp,
) {
    val deviceTilt by DeviceTilt.degrees
    var touchTilt by remember { mutableStateOf(Offset.Zero) }

    val tiltX = (deviceTilt.y + touchTilt.y).coerceIn(-6f, 6f)
    val tiltY = (deviceTilt.x + touchTilt.x).coerceIn(-6f, 6f)

    val boxSize = iconSize + 10.dp

    Box(
        modifier = modifier
            .size(boxSize)
            .graphicsLayer {
                rotationX = tiltX
                rotationY = tiltY
                cameraDistance = 18f * density
            }
            .pointerInput(Unit) {
                detectDragGestures(
                    onDrag = { change, _ ->
                        val nx = ((change.position.x / size.width) - 0.5f) * 2f
                        val ny = ((change.position.y / size.height) - 0.5f) * 2f
                        touchTilt = Offset((nx * 6f).coerceIn(-6f, 6f), (ny * 6f).coerceIn(-6f, 6f))
                    },
                    onDragEnd = { touchTilt = Offset.Zero },
                    onDragCancel = { touchTilt = Offset.Zero },
                )
            },
        contentAlignment = Alignment.Center,
    ) {
        if (active) {
            val transition = rememberInfiniteTransition(label = "icon-motion")
            val ringRotation by transition.animateFloat(
                initialValue = 0f, targetValue = 360f,
                animationSpec = infiniteRepeatable(tween(15000, easing = LinearEasing)),
                label = "ring-rotation",
            )
            val ringPulse by transition.animateFloat(
                initialValue = 0.35f, targetValue = 0.65f,
                animationSpec = infiniteRepeatable(tween(2200, easing = FastOutSlowInEasing), RepeatMode.Reverse),
                label = "ring-pulse",
            )
            val particleTime by transition.animateFloat(
                initialValue = 0f, targetValue = 1f,
                animationSpec = infiniteRepeatable(tween(10000, easing = LinearEasing)),
                label = "particle-time",
            )

            Canvas(modifier = Modifier.size(boxSize)) {
                rotate(degrees = ringRotation) {
                    drawCircle(
                        brush = Brush.sweepGradient(listOf(ORBIT_VIOLET, ORBIT_VIOLET_SOFT, ORBIT_VIOLET)),
                        radius = size.minDimension / 2f * 0.94f,
                        style = Stroke(width = 1.4.dp.toPx(), cap = StrokeCap.Round),
                        alpha = ringPulse,
                    )
                }
                orbitParticles.forEach { p ->
                    val t = (particleTime * p.speed + p.phase / 6.28f) % 1f
                    val angle = t * 6.2832f
                    val r = size.minDimension / 2f * p.radiusFactor
                    val cx = size.width / 2f + cos(angle) * r
                    val cy = size.height / 2f + sin(angle) * r
                    val twinkle = (sin(t * 6.2832f * 2f + p.phase) * 0.5f + 0.5f)
                    val alpha = (0.30f + 0.40f * twinkle).coerceIn(0.30f, 0.70f)
                    drawCircle(color = p.color, radius = 1.4.dp.toPx(), center = Offset(cx, cy), alpha = alpha)
                }
            }
        }
        Icon(icon, contentDescription = contentDescription, tint = tint, modifier = Modifier.size(iconSize))
    }
}
