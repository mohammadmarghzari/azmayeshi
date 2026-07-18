package ir.marghzari.portfolio360.ui.components

import androidx.compose.foundation.background
import androidx.compose.foundation.border
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
import androidx.compose.material.icons.filled.ArrowDropDown
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
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.vector.ImageVector
import androidx.compose.ui.unit.dp
import ir.marghzari.portfolio360.theme.LocalChartColors

/** Small-caps section divider with a trailing rule, matching the `.bp-section` CSS class. */
@Composable
fun SectionHeader(title: String, modifier: Modifier = Modifier) {
    val colors = LocalChartColors.current
    Row(
        modifier = modifier.fillMaxWidth().padding(top = 18.dp, bottom = 10.dp),
        verticalAlignment = Alignment.CenterVertically,
        horizontalArrangement = Arrangement.spacedBy(10.dp),
    ) {
        Text(title, style = MaterialTheme.typography.titleSmall, color = colors.accent)
        Box(
            modifier = Modifier.weight(1f).height(1.dp)
                .background(colors.plotGrid),
        )
    }
}

/** Generic card container, matching `.metric-card` / `.glass`. */
@Composable
fun Card(modifier: Modifier = Modifier, content: @Composable ColumnScope.() -> Unit) {
    val colors = LocalChartColors.current
    Column(
        modifier = modifier
            .background(colors.card, RoundedCornerShape(16.dp))
            .border(1.dp, colors.plotGrid, RoundedCornerShape(16.dp))
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
    Column(modifier = modifier.fillMaxWidth()) {
        Text(label, style = MaterialTheme.typography.labelMedium, color = colors.muted)
        Row(
            modifier = Modifier.fillMaxWidth()
                .padding(top = 4.dp)
                .background(colors.bg2, RoundedCornerShape(10.dp))
                .border(1.dp, colors.plotGrid, RoundedCornerShape(10.dp))
                .clickable { expanded = true }
                .padding(horizontal = 14.dp, vertical = 12.dp),
            horizontalArrangement = Arrangement.SpaceBetween,
            verticalAlignment = Alignment.CenterVertically,
        ) {
            Text(optionLabel(selected), style = MaterialTheme.typography.bodyMedium, color = colors.textPrimary)
            Icon(Icons.Filled.ArrowDropDown, contentDescription = null, tint = colors.muted)
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
        androidx.compose.material3.Button(
            onClick = onRetry, modifier = Modifier.fillMaxWidth().padding(top = 12.dp),
            colors = androidx.compose.material3.ButtonDefaults.buttonColors(containerColor = colors.blueAccent),
        ) { Text("تلاش دوباره") }
    }
}

@Composable
fun CrashScreen(screenName: String, error: Throwable, onRetry: () -> Unit) {
    CrashScreen(screenName, error.stackTraceToString(), onRetry)
}
