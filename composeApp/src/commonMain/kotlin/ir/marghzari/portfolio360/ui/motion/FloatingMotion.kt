package ir.marghzari.portfolio360.ui.motion

import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.graphicsLayer
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import kotlin.math.sin

/** A slow, tiny vertical bob (2-4px) so cards feel gently alive rather than static plates. */
@Composable
fun Modifier.floatingMotion(seed: Any? = null, amplitude: Dp = 3.dp): Modifier {
    val clock = LocalMotionClock.current ?: return this
    if (LocalReducedMotion.current) return this
    val phase = seedFrom(seed) * 6.2831855f
    return this.graphicsLayer {
        translationY = sin(clock.time.value * 6.2831855f * 1.4f + phase) * amplitude.toPx()
    }
}
