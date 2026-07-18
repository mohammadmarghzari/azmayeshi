package ir.marghzari.portfolio360.ui.components

import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.tween
import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.hoverable
import androidx.compose.foundation.interaction.MutableInteractionSource
import androidx.compose.foundation.interaction.collectIsHoveredAsState
import androidx.compose.foundation.interaction.collectIsPressedAsState
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.ColumnScope
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.heightIn
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.width
import androidx.compose.foundation.clickable
import androidx.compose.foundation.verticalScroll
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.rounded.ExpandMore
import androidx.compose.material3.ButtonDefaults
import androidx.compose.material3.DropdownMenu
import androidx.compose.material3.DropdownMenuItem
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.remember
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.scale
import androidx.compose.ui.draw.shadow
import androidx.compose.ui.graphics.Brush
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.unit.dp
import ir.marghzari.portfolio360.theme.LocalChartColors
import ir.marghzari.portfolio360.ui.motion.energyRing
import ir.marghzari.portfolio360.ui.motion.floatingMotion
import ir.marghzari.portfolio360.ui.motion.orbitParticles
import ir.marghzari.portfolio360.ui.motion.tilt3D
import kotlin.random.Random

/** Small-caps section divider with a trailing rule, matching the `.bp-section` CSS class. */
@Composable
fun SectionHeader(title: String, modifier: Modifier = Modifier) {
    val colors = LocalChartColors.current
    Row(
        modifier = modifier.fillMaxWidth().padding(top = 18.dp, bottom = 10.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Text(
            title,
            style = MaterialTheme.typography.titleSmall,
            color = colors.textPrimary,
            modifier = Modifier
                .background(colors.bg2.copy(alpha = 0.92f), RoundedCornerShape(8.dp))
                .padding(horizontal = 10.dp, vertical = 5.dp),
        )
        Box(
            modifier = Modifier.weight(1f).height(1.dp)
                .background(colors.plotGrid),
        )
    }
}

private val DEFAULT_MOTION_COLORS = listOf(Color(0xFF7C5CF6), Color(0xFF4E6BF2))

/**
 * Glassmorphism card container: translucent gradient fill, soft shadow, gradient edge highlight,
 * plus the app-wide motion system (subtle float, orbit particles, passive device-tilt) — every
 * screen picks this up automatically since they all already build on this one composable.
 * Set [highlighted] on the one "hero" card per screen (a selected asset, a focused summary) to add
 * the rotating energy ring; leave it off elsewhere so the ring stays a meaningful signal.
 * [motionColors] lets a caller theme the particles/ring (e.g. gold/silver/copper for a commodity
 * card via `motionColorsFor`); defaults to the app's purple/blue crypto glow.
 */
@Composable
fun Card(
    modifier: Modifier = Modifier,
    highlighted: Boolean = false,
    motionColors: List<Color>? = null,
    content: @Composable ColumnScope.() -> Unit,
) {
    val colors = LocalChartColors.current
    val shape = RoundedCornerShape(18.dp)
    val motionSeed = remember { Random.nextInt() }
    val particleColors = motionColors ?: DEFAULT_MOTION_COLORS
    Column(
        modifier = modifier
            .floatingMotion(seed = motionSeed)
            .tilt3D()
            .shadow(elevation = 8.dp, shape = shape, ambientColor = Color.Black.copy(alpha = 0.25f), spotColor = Color.Black.copy(alpha = 0.35f))
            .background(
                Brush.verticalGradient(listOf(colors.card.copy(alpha = 0.97f), colors.card.copy(alpha = 0.92f))),
                shape,
            )
            .border(
                width = 1.dp,
                brush = Brush.linearGradient(
                    listOf(Color.White.copy(alpha = 0.16f), colors.plotGrid.copy(alpha = 0.5f), Color.White.copy(alpha = 0.04f)),
                ),
                shape = shape,
            )
            .orbitParticles(colors = particleColors, seed = motionSeed)
            .energyRing(active = highlighted, colors = particleColors, cornerRadius = 18.dp)
            .padding(16.dp),
        content = content,
    )
}

@Composable
fun MetricTile(
    label: String,
    value: String,
    modifier: Modifier = Modifier,
    caption: String? = null,
    valueColor: Color? = null,
    delta: String? = null,
    deltaPositive: Boolean? = null,
) {
    val colors = LocalChartColors.current
    Card(modifier = modifier.width(170.dp)) {
        Text(label, style = MaterialTheme.typography.labelSmall, color = colors.muted)
        Text(
            value, style = MaterialTheme.typography.titleMedium, color = valueColor ?: colors.textPrimary,
            modifier = Modifier.padding(top = 4.dp),
        )
        if (delta != null) {
            val dc = when (deltaPositive) { true -> colors.green; false -> colors.red; null -> colors.muted }
            Text(delta, style = MaterialTheme.typography.labelSmall, color = dc, modifier = Modifier.padding(top = 2.dp))
        }
        if (caption != null) {
            Text(caption, style = MaterialTheme.typography.labelSmall, color = colors.muted, modifier = Modifier.padding(top = 2.dp))
        }
    }
}

enum class VerdictTone { POSITIVE, NEGATIVE, NEUTRAL }

/** Colored verdict banner, matching `.cc-verdict-card` (+ `.positive/.negative/.neutral` modifiers). */
@Composable
fun VerdictCard(title: String, body: String, tone: VerdictTone, modifier: Modifier = Modifier) {
    val colors = LocalChartColors.current
    val (borderColor, bg) = when (tone) {
        VerdictTone.POSITIVE -> colors.green to colors.green.copy(alpha = 0.08f)
        VerdictTone.NEGATIVE -> colors.red to colors.red.copy(alpha = 0.08f)
        VerdictTone.NEUTRAL -> colors.gold to colors.gold.copy(alpha = 0.08f)
    }
    Column(
        modifier = modifier.fillMaxWidth()
            .background(bg, RoundedCornerShape(14.dp))
            .border(1.dp, borderColor.copy(alpha = 0.5f), RoundedCornerShape(14.dp))
            .padding(16.dp),
    ) {
        Text(title, style = MaterialTheme.typography.titleSmall, color = borderColor)
        Text(body, style = MaterialTheme.typography.bodyMedium, color = colors.textPrimary, modifier = Modifier.padding(top = 6.dp))
    }
}

@Composable
fun InfoBanner(text: String, modifier: Modifier = Modifier, icon: ImageVector? = null) {
    val colors = LocalChartColors.current
    Row(
        modifier = modifier.fillMaxWidth()
            .background(colors.blueAccent.copy(alpha = 0.08f), RoundedCornerShape(12.dp))
            .padding(14.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        icon?.let { Icon(it, contentDescription = null, tint = colors.blueAccent) }
        Text(text, style = MaterialTheme.typography.bodyMedium, color = colors.textPrimary)
    }
}

@Composable
fun WarningBanner(text: String, modifier: Modifier = Modifier) {
    val colors = LocalChartColors.current
    Row(
        modifier = modifier.fillMaxWidth()
            .background(colors.gold.copy(alpha = 0.10f), RoundedCornerShape(12.dp))
            .padding(14.dp),
    ) {
        Text(text, style = MaterialTheme.typography.bodyMedium, color = colors.gold)
    }
}

/** A simple, robust dropdown (avoids ExposedDropdownMenuBox/Menu API instability across Compose versions). */
@Composable
fun <T> SimpleDropdown(
    label: String,
    selected: T,
    options: List<T>,
    optionLabel: (T) -> String,
    onSelected: (T) -> Unit,
    modifier: Modifier = Modifier,
) {
    val colors = LocalChartColors.current
    var expanded by remember { mutableStateOf(false) }
    val pillShape = RoundedCornerShape(50)
    Column(modifier = modifier.fillMaxWidth()) {
        Text(label, style = MaterialTheme.typography.labelMedium, color = colors.muted)
        Row(
            modifier = Modifier.fillMaxWidth()
                .padding(top = 4.dp)
                .background(colors.bg2, pillShape)
                .border(1.dp, colors.plotGrid, pillShape)
                .energyRing(active = true, cornerRadius = 999.dp, strokeWidth = 1.2.dp)
                .clickable { expanded = true }
                .padding(horizontal = 16.dp, vertical = 12.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(optionLabel(selected), style = MaterialTheme.typography.bodyMedium, color = colors.textPrimary)
            Icon(Icons.Rounded.ExpandMore, contentDescription = null, tint = colors.muted)
        }
        DropdownMenu(expanded = expanded, onDismissRequest = { expanded = false }) {
            options.forEach { opt ->
                DropdownMenuItem(text = { Text(optionLabel(opt)) }, onClick = { onSelected(opt); expanded = false })
            }
        }
    }
}

/**
 * Shown instead of letting an uncaught exception crash the app silently. The full stack trace
 * text is selectable so it can be copied/screenshotted and reported.
 */
@Composable
fun CrashScreen(screenName: String, traceText: String, onRetry: () -> Unit) {
    val colors = LocalChartColors.current
    Column(
        modifier = Modifier.fillMaxWidth()
            .background(colors.red.copy(alpha = 0.06f), RoundedCornerShape(16.dp))
            .padding(18.dp),
    ) {
        Text("⚠ خطا در $screenName", style = MaterialTheme.typography.titleMedium, color = colors.red)
        Text(
            "برنامه با خطا مواجه شد. لطفاً از متن زیر اسکرین‌شات بگیرید و بفرستید تا برطرف شود.",
            style = MaterialTheme.typography.bodySmall, color = colors.muted, modifier = Modifier.padding(top = 6.dp, bottom = 12.dp),
        )
        androidx.compose.foundation.text.selection.SelectionContainer {
            Column(
                modifier = Modifier.fillMaxWidth()
                    .heightIn(max = 420.dp)
                    .verticalScroll(androidx.compose.foundation.rememberScrollState())
                    .background(colors.bg2, RoundedCornerShape(10.dp))
                    .padding(12.dp),
            ) {
                Text(
                    traceText,
                    style = MaterialTheme.typography.labelSmall.copy(fontFamily = androidx.compose.ui.text.font.FontFamily.Monospace),
                    color = colors.textPrimary,
                )
            }
        }
        GlowButton(text = "تلاش دوباره", onClick = onRetry, modifier = Modifier.fillMaxWidth().padding(top = 12.dp))
    }
}

@Composable
fun CrashScreen(screenName: String, error: Throwable, onRetry: () -> Unit) {
    CrashScreen(screenName, error.stackTraceToString(), onRetry)
}

/**
 * Premium call-to-action button: gradient fill, press-down scale, and a hover lift on desktop
 * (hover state is simply never triggered by touch, so it's a no-op on Android).
 */
@Composable
fun GlowButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    enabled: Boolean = true,
) {
    val colors = LocalChartColors.current
    val interactionSource = remember { MutableInteractionSource() }
    val pressed by interactionSource.collectIsPressedAsState()
    val hovered by interactionSource.collectIsHoveredAsState()
    val scale by animateFloatAsState(
        targetValue = if (pressed) 0.96f else if (hovered) 1.02f else 1f,
        animationSpec = tween(140, easing = FastOutSlowInEasing),
        label = "glow-button-scale",
    )
    val shape = RoundedCornerShape(50)
    androidx.compose.material3.Button(
        onClick = onClick,
        enabled = enabled,
        interactionSource = interactionSource,
        shape = shape,
        colors = ButtonDefaults.buttonColors(containerColor = Color.Transparent, contentColor = Color.White),
        contentPadding = androidx.compose.foundation.layout.PaddingValues(horizontal = 22.dp, vertical = 12.dp),
        modifier = modifier
            .scale(scale)
            .hoverable(interactionSource)
            .shadow(if (hovered) 12.dp else 6.dp, shape, ambientColor = colors.blueAccent.copy(alpha = 0.4f), spotColor = colors.blueAccent.copy(alpha = 0.5f))
            .background(Brush.horizontalGradient(listOf(colors.blueAccent, colors.gold.copy(alpha = 0.9f))), shape),
    ) {
        Text(text, style = MaterialTheme.typography.labelLarge)
    }
}
