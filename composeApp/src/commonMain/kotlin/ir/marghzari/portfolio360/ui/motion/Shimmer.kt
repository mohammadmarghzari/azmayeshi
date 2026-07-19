package ir.marghzari.portfolio360.ui.motion

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.draw.drawWithContent
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import ir.marghzari.portfolio360.theme.LocalChartColors
import ir.marghzari.portfolio360.theme.Radii
import ir.marghzari.portfolio360.theme.Spacing

/**
 * Loading-skeleton shimmer: a soft diagonal light band sweeping across the element. Driven by the
 * shared [MotionClock] (read only inside the draw phase, so each sweep frame re-draws this layer
 * without recomposing anything), and disabled entirely under [LocalReducedMotion].
 */
@Composable
fun Modifier.shimmer(): Modifier {
    val clock = LocalMotionClock.current
    val reduced = LocalReducedMotion.current
    if (clock == null || reduced) return this
    return drawWithContent {
        drawContent()
        // 8 sweeps per 20s clock cycle => one pass every 2.5s.
        val t = (clock.time.value * 8f) % 1f
        val band = size.width * 0.35f
        val start = -band + (size.width + 2f * band) * t
        drawRect(
            brush = Brush.linearGradient(
                colors = listOf(Color.Transparent, Color.White.copy(alpha = 0.10f), Color.Transparent),
                start = Offset(start, 0f),
                end = Offset(start + band, size.height),
            ),
        )
    }
}

/** One gray placeholder bar; building block for skeleton layouts. */
@Composable
fun SkeletonBox(modifier: Modifier = Modifier, height: Dp = 14.dp, cornerRadius: Dp = Radii.sm) {
    val colors = LocalChartColors.current
    Box(
        modifier = modifier
            .height(height)
            .clip(RoundedCornerShape(cornerRadius))
            .background(colors.bg2)
            .shimmer(),
    )
}

/** A stack of shimmering placeholder lines, the last one shorter — reads as "text is coming". */
@Composable
fun SkeletonLines(lines: Int = 3, modifier: Modifier = Modifier) {
    Column(modifier = modifier.fillMaxWidth(), verticalArrangement = Arrangement.spacedBy(Spacing.md)) {
        repeat(lines) { i ->
            SkeletonBox(modifier = Modifier.fillMaxWidth(if (i == lines - 1) 0.6f else 1f))
        }
    }
}

/** Card-shaped loading placeholder used as the default [StateHost] skeleton. */
@Composable
fun SkeletonCard(lines: Int = 4, modifier: Modifier = Modifier) {
    val colors = LocalChartColors.current
    Column(
        modifier = modifier.fillMaxWidth()
            .clip(RoundedCornerShape(Radii.lg))
            .background(colors.card.copy(alpha = 0.85f))
            .padding(Spacing.lg),
        verticalArrangement = Arrangement.spacedBy(Spacing.md),
    ) {
        SkeletonBox(modifier = Modifier.fillMaxWidth(0.4f), height = 18.dp)
        SkeletonLines(lines = lines)
    }
}
