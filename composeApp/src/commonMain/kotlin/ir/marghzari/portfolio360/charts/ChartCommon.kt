package ir.marghzari.portfolio360.charts

import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.size
import androidx.compose.foundation.layout.wrapContentWidth
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.unit.dp
import ir.marghzari.portfolio360.theme.LocalChartColors
import kotlin.math.abs
import kotlin.math.ceil
import kotlin.math.floor
import kotlin.math.ln
import kotlin.math.pow

/** "Nice" axis tick computation, matching the clean gridlines Plotly auto-generates. */
data class AxisTicks(val min: Double, val max: Double, val step: Double) {
    val values: List<Double> get() {
        val out = mutableListOf<Double>()
        var v = ceil(min / step) * step
        while (v <= max + step * 1e-6) {
            out.add(v)
            v += step
        }
        return out
    }
}

fun niceAxisTicks(dataMin: Double, dataMax: Double, targetCount: Int = 5): AxisTicks {
    if (dataMax <= dataMin) {
        val pad = if (dataMax == 0.0) 1.0 else abs(dataMax) * 0.1
        return niceAxisTicks(dataMin - pad, dataMax + pad, targetCount)
    }
    val range = dataMax - dataMin
    val roughStep = range / targetCount
    val magnitude = 10.0.pow(floor(ln(roughStep) / ln(10.0)))
    val normalized = roughStep / magnitude
    val niceNormalized = when {
        normalized < 1.5 -> 1.0
        normalized < 3 -> 2.0
        normalized < 7 -> 5.0
        else -> 10.0
    }
    val step = niceNormalized * magnitude
    val min = floor(dataMin / step) * step
    val max = ceil(dataMax / step) * step
    return AxisTicks(min, max, step)
}

fun formatCompactNumber(v: Double): String {
    val abs = abs(v)
    return when {
        abs >= 1_000_000_000 -> "%.1fB".format(v / 1_000_000_000)
        abs >= 1_000_000 -> "%.1fM".format(v / 1_000_000)
        abs >= 1_000 -> "%.1fK".format(v / 1_000)
        abs >= 10 -> "%.0f".format(v)
        else -> "%.2f".format(v)
    }
}

fun formatPct(v: Double, decimals: Int = 1): String = "%.${decimals}f%%".format(v)

@Composable
fun ChartLegend(entries: List<Pair<String, Color>>, modifier: Modifier = Modifier) {
    Row(modifier = modifier.wrapContentWidth(), horizontalArrangement = Arrangement.spacedBy(14.dp)) {
        entries.forEach { (label, color) ->
            Row(verticalAlignment = Alignment.CenterVertically, horizontalArrangement = Arrangement.spacedBy(6.dp)) {
                Box(modifier = Modifier.size(9.dp).clip(RoundedCornerShape(2.dp)).background(color))
                Text(label, style = MaterialTheme.typography.labelSmall, color = LocalChartColors.current.plotText)
            }
        }
    }
}
