package ir.marghzari.portfolio360.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
import androidx.compose.foundation.clickable
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.shape.CircleShape
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.AlertDialog
import androidx.compose.material3.Icon
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.OutlinedTextField
import androidx.compose.material3.OutlinedTextFieldDefaults
import androidx.compose.material3.Snackbar
import androidx.compose.material3.SnackbarHost
import androidx.compose.material3.SnackbarHostState
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import ir.marghzari.portfolio360.theme.LocalChartColors
import ir.marghzari.portfolio360.theme.Radii
import ir.marghzari.portfolio360.theme.Spacing
import kotlin.math.absoluteValue

/**
 * Design-system button styles. [Primary] is the loud gradient call-to-action (delegates to
 * [GlowButton]); [Secondary] is a quiet tonal fill for supporting actions; [Ghost] is text-only
 * for tertiary/inline actions; [Destructive] is reserved for irreversible operations.
 */
enum class ButtonStyle { Primary, Secondary, Ghost, Destructive }

@Composable
fun AppButton(
    text: String,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
    style: ButtonStyle = ButtonStyle.Primary,
    enabled: Boolean = true,
    loading: Boolean = false,
) {
    val colors = LocalChartColors.current
    when (style) {
        ButtonStyle.Primary -> GlowButton(text, onClick, modifier, enabled, loading)
        ButtonStyle.Destructive -> GlowButton(
            text, onClick, modifier, enabled, loading,
            gradient = listOf(colors.red, colors.red.copy(alpha = 0.75f)),
        )
        ButtonStyle.Secondary -> {
            val shape = RoundedCornerShape(Radii.pill)
            val alpha = if (enabled) 1f else 0.5f
            Row(
                modifier = modifier
                    .background(colors.blueAccent.copy(alpha = 0.14f * alpha), shape)
                    .border(1.dp, colors.blueAccent.copy(alpha = 0.35f * alpha), shape)
                    .clickable(enabled = enabled && !loading) { onClick() }
                    .padding(horizontal = 22.dp, vertical = 12.dp),
                horizontalArrangement = Arrangement.Center,
                verticalAlignment = Alignment.CenterVertically,
            ) {
                Text(text, style = MaterialTheme.typography.labelLarge, color = colors.blueAccent.copy(alpha = alpha))
            }
        }
        ButtonStyle.Ghost -> TextButton(onClick = onClick, enabled = enabled && !loading, modifier = modifier) {
            Text(text, style = MaterialTheme.typography.labelLarge, color = colors.blueAccent)
        }
    }
}

/**
 * Themed text input: one place defines the border/label/cursor colors and corner radius so every
 * screen's fields look identical instead of each call site repeating `OutlinedTextField` boilerplate.
 */
@Composable
fun AppTextField(
    value: String,
    onValueChange: (String) -> Unit,
    modifier: Modifier = Modifier,
    label: String? = null,
    placeholder: String? = null,
    singleLine: Boolean = true,
    enabled: Boolean = true,
    isError: Boolean = false,
    supportingText: String? = null,
) {
    val colors = LocalChartColors.current
    OutlinedTextField(
        value = value,
        onValueChange = onValueChange,
        label = label?.let { { Text(it) } },
        placeholder = placeholder?.let { { Text(it, color = colors.muted) } },
        singleLine = singleLine,
        enabled = enabled,
        isError = isError,
        supportingText = supportingText?.let { { Text(it) } },
        shape = RoundedCornerShape(Radii.md),
        colors = OutlinedTextFieldDefaults.colors(
            focusedBorderColor = colors.blueAccent,
            unfocusedBorderColor = colors.plotGrid,
            errorBorderColor = colors.red,
            focusedLabelColor = colors.blueAccent,
            unfocusedLabelColor = colors.muted,
            cursorColor = colors.blueAccent,
            focusedTextColor = colors.textPrimary,
            unfocusedTextColor = colors.textPrimary,
        ),
        modifier = modifier,
    )
}

/** Selectable filter chip: filled violet pill when selected, quiet outline otherwise. */
@Composable
fun AppChip(
    text: String,
    selected: Boolean,
    onClick: () -> Unit,
    modifier: Modifier = Modifier,
) {
    val colors = LocalChartColors.current
    val shape = RoundedCornerShape(Radii.pill)
    Row(
        modifier = modifier
            .background(if (selected) colors.blueAccent else colors.bg2, shape)
            .border(1.dp, if (selected) colors.blueAccent else colors.plotGrid, shape)
            .clickable { onClick() }
            .padding(horizontal = Spacing.lg, vertical = Spacing.sm),
        verticalAlignment = Alignment.CenterVertically,
    ) {
        Text(
            text,
            style = MaterialTheme.typography.labelMedium,
            color = if (selected) Color.White else colors.textPrimary,
        )
    }
}

enum class BadgeTone { SUCCESS, WARNING, ERROR, NEUTRAL }

/** Small tinted status pill ("فعال", "منقضی", "در انتظار"...) for rows and cards. */
@Composable
fun StatusBadge(text: String, tone: BadgeTone, modifier: Modifier = Modifier) {
    val colors = LocalChartColors.current
    val tint = when (tone) {
        BadgeTone.SUCCESS -> colors.green
        BadgeTone.WARNING -> colors.gold
        BadgeTone.ERROR -> colors.red
        BadgeTone.NEUTRAL -> colors.muted
    }
    Text(
        text,
        style = MaterialTheme.typography.labelSmall,
        color = tint,
        modifier = modifier
            .background(tint.copy(alpha = 0.12f), RoundedCornerShape(Radii.pill))
            .padding(horizontal = Spacing.md, vertical = Spacing.xs),
    )
}

/** Themed dialog wrapper so every dialog shares the same surface color, radius and actions. */
@Composable
fun AppDialog(
    title: String,
    onDismiss: () -> Unit,
    modifier: Modifier = Modifier,
    confirmText: String? = null,
    onConfirm: (() -> Unit)? = null,
    dismissText: String = "بستن",
    content: @Composable () -> Unit,
) {
    val colors = LocalChartColors.current
    AlertDialog(
        onDismissRequest = onDismiss,
        containerColor = colors.panel,
        titleContentColor = colors.textPrimary,
        textContentColor = colors.textPrimary,
        shape = RoundedCornerShape(Radii.lg),
        title = { Text(title, style = MaterialTheme.typography.titleMedium) },
        text = content,
        confirmButton = {
            if (confirmText != null && onConfirm != null) {
                AppButton(confirmText, onClick = onConfirm, style = ButtonStyle.Primary)
            }
        },
        dismissButton = {
            AppButton(dismissText, onClick = onDismiss, style = ButtonStyle.Ghost)
        },
        modifier = modifier,
    )
}

/** Snackbar host with the app's surface/accent colors; pair with a remembered [SnackbarHostState]. */
@Composable
fun AppSnackbarHost(hostState: SnackbarHostState, modifier: Modifier = Modifier) {
    val colors = LocalChartColors.current
    SnackbarHost(hostState = hostState, modifier = modifier) { data ->
        Snackbar(
            snackbarData = data,
            containerColor = colors.panel,
            contentColor = colors.textPrimary,
            actionColor = colors.blueAccent,
            shape = RoundedCornerShape(Radii.md),
        )
    }
}

private val COIN_AVATAR_PALETTE = listOf(
    Color(0xFFF7931A), // bitcoin orange
    Color(0xFF627EEA), // ether indigo
    Color(0xFF7C3AED), // brand violet
    Color(0xFF16A34A), // green
    Color(0xFF0EA5E9), // sky
    Color(0xFFE11D48), // rose
    Color(0xFF14B8A6), // teal
    Color(0xFFF59E0B), // amber
)

/**
 * Delta/finvest-style asset icon: a colored circle with the ticker's initials. The color is
 * derived deterministically from the symbol so the same asset always gets the same hue.
 */
@Composable
fun CoinAvatar(symbol: String, modifier: Modifier = Modifier, size: Dp = 36.dp) {
    val clean = symbol.trim().removeSuffix("-USD").ifBlank { "?" }
    val color = COIN_AVATAR_PALETTE[clean.hashCode().absoluteValue % COIN_AVATAR_PALETTE.size]
    Box(
        modifier = modifier.size(size).background(color, CircleShape),
        contentAlignment = Alignment.Center,
    ) {
        Text(
            clean.take(2).uppercase(),
            style = MaterialTheme.typography.labelMedium.copy(fontWeight = FontWeight.Bold),
            color = Color.White,
        )
    }
}

/** Friendly full-bleed placeholder for screens/sections that have nothing to show yet. */
@Composable
fun EmptyState(
    title: String,
    modifier: Modifier = Modifier,
    hint: String? = null,
    icon: ImageVector? = null,
    actionText: String? = null,
    onAction: (() -> Unit)? = null,
) {
    val colors = LocalChartColors.current
    Column(
        modifier = modifier.fillMaxWidth().padding(vertical = 40.dp, horizontal = Spacing.xl),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(Spacing.md),
    ) {
        if (icon != null) {
            Box(
                modifier = Modifier.size(64.dp).background(colors.blueAccent.copy(alpha = 0.10f), CircleShape),
                contentAlignment = Alignment.Center,
            ) {
                Icon(icon, contentDescription = null, tint = colors.blueAccent, modifier = Modifier.size(30.dp))
            }
        }
        Text(title, style = MaterialTheme.typography.titleSmall, color = colors.textPrimary)
        if (hint != null) {
            Text(hint, style = MaterialTheme.typography.bodySmall, color = colors.muted)
        }
        if (actionText != null && onAction != null) {
            AppButton(actionText, onClick = onAction, style = ButtonStyle.Secondary)
        }
    }
}

/** Error placeholder with an optional retry action; used by StateHost and directly by screens. */
@Composable
fun ErrorState(
    message: String,
    modifier: Modifier = Modifier,
    onRetry: (() -> Unit)? = null,
) {
    val colors = LocalChartColors.current
    Column(
        modifier = modifier.fillMaxWidth()
            .background(colors.red.copy(alpha = 0.06f), RoundedCornerShape(Radii.lg))
            .border(1.dp, colors.red.copy(alpha = 0.35f), RoundedCornerShape(Radii.lg))
            .padding(Spacing.xl),
        horizontalAlignment = Alignment.CenterHorizontally,
        verticalArrangement = Arrangement.spacedBy(Spacing.md),
    ) {
        Text("خطا در دریافت اطلاعات", style = MaterialTheme.typography.titleSmall, color = colors.red)
        Text(message, style = MaterialTheme.typography.bodySmall, color = colors.textPrimary)
        if (onRetry != null) {
            AppButton("تلاش دوباره", onClick = onRetry, style = ButtonStyle.Secondary)
        }
    }
}

/**
 * Standard screen intro: large title + muted one-line subtitle with consistent rhythm, replacing
 * each screen's ad-hoc first SectionHeader + caption pair.
 */
@Composable
fun ScreenHeader(title: String, subtitle: String? = null, modifier: Modifier = Modifier) {
    val colors = LocalChartColors.current
    Column(modifier = modifier.fillMaxWidth().padding(top = Spacing.sm, bottom = Spacing.sm)) {
        Text(title, style = MaterialTheme.typography.titleLarge, color = colors.textPrimary)
        if (subtitle != null) {
            Text(
                subtitle, style = MaterialTheme.typography.bodySmall, color = colors.muted,
                modifier = Modifier.padding(top = Spacing.xs),
            )
        }
    }
}

/**
 * Hero number for summary cards: oversized mono-bold value with a label above and an optional
 * tinted delta pill beside it — the Coinbase/Delta "big balance" pattern.
 */
@Composable
fun HeroMetric(
    label: String,
    value: String,
    modifier: Modifier = Modifier,
    delta: String? = null,
    deltaPositive: Boolean? = null,
) {
    val colors = LocalChartColors.current
    Column(modifier = modifier) {
        Text(label, style = MaterialTheme.typography.labelMedium, color = colors.muted)
        Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(Spacing.md)) {
            Text(
                value,
                style = MaterialTheme.typography.titleLarge.copy(fontSize = 28.sp, fontWeight = FontWeight.Bold),
                color = colors.textPrimary,
                modifier = Modifier.padding(top = Spacing.xs),
            )
            if (delta != null) {
                StatusBadge(
                    text = delta,
                    tone = when (deltaPositive) {
                        true -> BadgeTone.SUCCESS
                        false -> BadgeTone.ERROR
                        null -> BadgeTone.NEUTRAL
                    },
                )
            }
        }
    }
}

/**
 * Asset list row: colored [CoinAvatar] + name/caption on one side, right-aligned mono value and
 * colored delta on the other — the standard holdings/quotes row of Delta/CoinStats.
 */
@Composable
fun AssetRow(
    symbol: String,
    title: String,
    value: String,
    modifier: Modifier = Modifier,
    caption: String? = null,
    delta: String? = null,
    deltaPositive: Boolean? = null,
) {
    val colors = LocalChartColors.current
    Row(
        modifier = modifier.fillMaxWidth().padding(vertical = Spacing.sm),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(Spacing.md),
    ) {
        CoinAvatar(symbol)
        Column(modifier = Modifier.weight(1f)) {
            Text(title, style = MaterialTheme.typography.bodyMedium, color = colors.textPrimary)
            if (caption != null) {
                Text(caption, style = MaterialTheme.typography.labelSmall, color = colors.muted)
            }
        }
        Column(horizontalAlignment = Alignment.End) {
            Text(value, style = MaterialTheme.typography.labelLarge, color = colors.textPrimary)
            if (delta != null) {
                val dc = when (deltaPositive) { true -> colors.green; false -> colors.red; null -> colors.muted }
                Text(delta, style = MaterialTheme.typography.labelSmall, color = dc)
            }
        }
    }
}
