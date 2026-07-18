package ir.marghzari.portfolio360.ui.background

import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.Canvas
import androidx.compose.foundation.Image
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxScope
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.layout.ContentScale
import kotlin.math.sin
import kotlin.random.Random
import org.jetbrains.compose.resources.DrawableResource
import org.jetbrains.compose.resources.painterResource

private class Ember(val seedX: Float, val speed: Float, val size: Float, val phase: Float, val sway: Float)

private val embers = List(24) {
    Ember(
        seedX = Random.nextFloat(),
        speed = 0.4f + Random.nextFloat() * 0.6f,
        size = 1.3f + Random.nextFloat() * 2.4f,
        phase = Random.nextFloat() * 6.28f,
        sway = 8f + Random.nextFloat() * 18f,
    )
}

/**
 * Reusable cinematic backdrop: a slow breathing/parallax image, a dark scrim + vignette so
 * foreground text always stays legible over busy artwork, and a soft drifting-ember particle
 * field. Every screen and the splash draw through this one composable so the whole app shares a
 * single motion language instead of each place reinventing it.
 */
@Composable
fun AnimatedBackground(
    image: DrawableResource,
    modifier: Modifier = Modifier,
    intensity: Float = 1f,
    scrimAlpha: Float = 0.74f,
    particles: Boolean = true,
    content: @Composable BoxScope.() -> Unit,
) {
    val transition = rememberInfiniteTransition(label = "bg-transition")
    val scale by transition.animateFloat(
        initialValue = 1.0f,
        targetValue = 1f + 0.06f * intensity,
        animationSpec = infiniteRepeatable(tween(17000, easing = FastOutSlowInEasing), RepeatMode.Reverse),
        label = "bg-scale",
    )
    val driftX by transition.animateFloat(
        initialValue = -16f * intensity,
        targetValue = 16f * intensity,
        animationSpec = infiniteRepeatable(tween(21000, easing = FastOutSlowInEasing), RepeatMode.Reverse),
        label = "bg-driftX",
    )
    val driftY by transition.animateFloat(
        initialValue = -10f * intensity,
        targetValue = 10f * intensity,
        animationSpec = infiniteRepeatable(tween(25000, easing = FastOutSlowInEasing), RepeatMode.Reverse),
        label = "bg-driftY",
    )
    val time by transition.animateFloat(
        initialValue = 0f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(tween(38000, easing = LinearEasing)),
        label = "bg-time",
    )

    Box(modifier = modifier.fillMaxSize().background(Color.Black)) {
        Image(
            painter = painterResource(image),
            contentDescription = null,
            contentScale = ContentScale.Crop,
            modifier = Modifier.fillMaxSize().graphicsLayer {
                scaleX = scale; scaleY = scale
                translationX = driftX; translationY = driftY
            },
        )
        Box(
            modifier = Modifier.fillMaxSize().background(
                Brush.verticalGradient(
                    listOf(
                        Color.Black.copy(alpha = scrimAlpha * 0.72f),
                        Color.Black.copy(alpha = scrimAlpha * 0.90f),
                        Color.Black.copy(alpha = scrimAlpha * 1.05f),
                    ),
                ),
            ),
        )
        if (particles) {
            Canvas(modifier = Modifier.fillMaxSize()) {
                val vignette = Brush.radialGradient(
                    colors = listOf(Color.Transparent, Color.Black.copy(alpha = 0.5f)),
                    center = Offset(size.width / 2f, size.height * 0.4f),
                    radius = size.maxDimension * 0.8f,
                )
                drawRect(vignette)
                embers.forEach { e ->
                    val t = (time * e.speed + e.phase / 6.28f) % 1f
                    val y = size.height * (1f - t)
                    val x = size.width * e.seedX + sin((t * 6.28f) + e.phase).toFloat() * e.sway
                    val twinkle = (sin(t * 6.28f * 2f + e.phase).toFloat() * 0.5f + 0.5f)
                    val alpha = (0.10f + 0.28f * twinkle) * (1f - kotlin.math.abs(t - 0.5f) * 0.6f)
                    if (alpha > 0.01f) {
                        drawCircle(color = Color(0xFFE8C87A), radius = e.size * 2.4f, center = Offset(x, y), alpha = (alpha * 0.3f).coerceIn(0f, 1f))
                        drawCircle(color = Color(0xFFF4E4B0), radius = e.size, center = Offset(x, y), alpha = alpha.coerceIn(0f, 1f))
                    }
                }
            }
        } else {
            Canvas(modifier = Modifier.fillMaxSize()) {
                val vignette = Brush.radialGradient(
                    colors = listOf(Color.Transparent, Color.Black.copy(alpha = 0.5f)),
                    center = Offset(size.width / 2f, size.height * 0.4f),
                    radius = size.maxDimension * 0.8f,
                )
                drawRect(vignette)
            }
        }
        content()
    }
}
