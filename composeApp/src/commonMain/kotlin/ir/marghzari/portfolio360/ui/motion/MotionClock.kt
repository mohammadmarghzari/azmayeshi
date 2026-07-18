package ir.marghzari.portfolio360.ui.motion

import androidx.compose.animation.core.LinearEasing
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.runtime.Composable
import androidx.compose.runtime.State
import androidx.compose.runtime.compositionLocalOf
import androidx.compose.runtime.staticCompositionLocalOf

/**
 * One shared pair of animation drivers for the whole app's decorative motion (orbit particles,
 * energy rings, floating cards, chart pulses). Every effect below reads these same two continuous
 * values and derives its own oscillation via sin()/cos() with a per-instance phase offset — so
 * applying motion to dozens of cards/metrics/charts across all 16 screens costs two shared
 * animations plus cheap per-instance math, not one animation loop per element, and every instance
 * still moves independently instead of in lockstep.
 */
class MotionClock internal constructor(
    /** Linear ramp 0..1 every 20s — the base "time" for orbit particles, pulses, and floating. */
    val time: State<Float>,
    /** Linear ramp 0..360 every 15s — used directly as an energy-ring rotation angle. */
    val ringRotation: State<Float>,
)

@Composable
fun rememberMotionClock(): MotionClock {
    val transition = rememberInfiniteTransition(label = "app-motion-clock")
    val time = transition.animateFloat(
        initialValue = 0f, targetValue = 1f,
        animationSpec = infiniteRepeatable(tween(20000, easing = LinearEasing)),
        label = "motion-time",
    )
    val ringRotation = transition.animateFloat(
        initialValue = 0f, targetValue = 360f,
        animationSpec = infiniteRepeatable(tween(15000, easing = LinearEasing)),
        label = "ring-rotation",
    )
    return MotionClock(time, ringRotation)
}

val LocalMotionClock = compositionLocalOf<MotionClock?> { null }

/** Global "reduce motion" switch — when on, every decorative motion effect below turns itself off. */
val LocalReducedMotion = staticCompositionLocalOf { false }

/** Stable per-call-site jitter (0..1) so identical widgets (e.g. many MetricTiles) don't move in lockstep. */
internal fun seedFrom(vararg keys: Any?): Float {
    var h = 0
    keys.forEach { h = h * 31 + (it?.hashCode() ?: 0) }
    return (h.mod(1000)) / 1000f
}
