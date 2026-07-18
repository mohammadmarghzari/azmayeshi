package ir.marghzari.portfolio360.ui.motion

import androidx.compose.animation.core.Animatable
import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.tween
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.State
import androidx.compose.runtime.derivedStateOf
import androidx.compose.runtime.remember

/**
 * Shared "draw the chart in" progress (0..1) reused by every chart type instead of each
 * reimplementing its own reveal animation. Re-plays whenever [key] changes (e.g. new data), so a
 * chart that swaps datasets re-draws itself in rather than snapping. Respects reduced-motion.
 */
@Composable
fun rememberChartReveal(key: Any?): State<Float> {
    val reduced = LocalReducedMotion.current
    val progress = remember { Animatable(if (reduced) 1f else 0f) }
    LaunchedEffect(key, reduced) {
        if (reduced) {
            progress.snapTo(1f)
        } else {
            progress.snapTo(0f)
            progress.animateTo(1f, animationSpec = tween(800, easing = FastOutSlowInEasing))
        }
    }
    return remember { derivedStateOf { progress.value } }
}
