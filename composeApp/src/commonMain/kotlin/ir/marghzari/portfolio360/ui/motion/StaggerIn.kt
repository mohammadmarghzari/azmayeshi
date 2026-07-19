package ir.marghzari.portfolio360.ui.motion

import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.layout.Box
import androidx.compose.runtime.Composable
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.graphicsLayer
import kotlinx.coroutines.delay

/**
 * Staggered entrance for list content: each item fades in and rises a little, delayed by its
 * [index], so lists "cascade" into place instead of popping in all at once (the Coinbase/Revolut
 * list-load feel). The delay is capped so long lists don't keep the tail invisible; under
 * [LocalReducedMotion] items appear immediately.
 */
@Composable
fun StaggerIn(index: Int, modifier: Modifier = Modifier, content: @Composable () -> Unit) {
    val reduced = LocalReducedMotion.current
    var shown by remember { mutableStateOf(reduced) }
    LaunchedEffect(Unit) {
        if (!shown) {
            delay(index.coerceAtMost(12) * 45L)
            shown = true
        }
    }
    val alpha by animateFloatAsState(
        targetValue = if (shown) 1f else 0f,
        animationSpec = tween(320, easing = FastOutSlowInEasing),
        label = "stagger-alpha",
    )
    val rise by animateFloatAsState(
        targetValue = if (shown) 0f else 1f,
        animationSpec = tween(320, easing = FastOutSlowInEasing),
        label = "stagger-rise",
    )
    Box(
        modifier = modifier.graphicsLayer {
            this.alpha = alpha
            translationY = rise * 22f
        },
    ) {
        content()
    }
}
