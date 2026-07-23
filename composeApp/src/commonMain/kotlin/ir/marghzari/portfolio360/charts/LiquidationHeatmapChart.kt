package ir.marghzari.portfolio360.charts

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.Path
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.StrokeJoin
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.lerp
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import ir.marghzari.portfolio360.core.network.HeatmapCell
import ir.marghzari.portfolio360.theme.LocalChartColors
import ir.marghzari.portfolio360.ui.motion.rememberChartReveal

/** Color stops approximating CoinGlass's own palette: near-black -> violet -> teal -> yellow-hot. */
private val LIQUIDATION_STOPS = listOf(
    0.00f to Color(0xFF0F0A1E),
    0.35f to Color(0xFF3A1E6E),
    0.65f to Color(0xFF16A6A6),
    1.00f to Color(0xFFF5E663),
)

private fun liquidationColor(t: Float): Color {
    val clamped = t.coerceIn(0f, 1f)
    for (i in 0 until LIQUIDATION_STOPS.size - 1) {
        val (p0, c0) = LIQUIDATION_STOPS[i]
        val (p1, c1) = LIQUIDATION_STOPS[i + 1]
        if (clamped in p0..p1) {
            val span = (p1 - p0).takeIf { it > 0f } ?: 1f
            return lerp(c0, c1, (clamped - p0) / span)
        }
    }
    return LIQUIDATION_STOPS.last().second
}

/**
 * Continuous-gradient liquidation heatmap: unlike [HeatmapChart] (a small labeled grid for
 * correlation/seasonality matrices), this renders potentially thousands of sparse [cells] as
 * intensity-colored blobs across a continuous price/time plane, with the actual price path drawn
 * on top — the CoinGlass "liquidation heatmap" visual.
 */
@Composable
fun LiquidationHeatmapChart(
    cells: List<HeatmapCell>,
    priceLine: List<Pair<Long, Double>>,
    modifier: Modifier = Modifier,
    title: String? = null,
    height: Dp = 340.dp,
) {
    val colors = LocalChartColors.current
    val textMeasurer = rememberTextMeasurer()
    val reveal by rememberChartReveal(cells.size to priceLine.size)

    Column(modifier = modifier.fillMaxWidth()) {
        if (title != null) {
            Text(title, style = MaterialTheme.typography.titleSmall, color = colors.plotText, modifier = Modifier.padding(bottom = 6.dp))
        }
        Canvas(modifier = Modifier.fillMaxWidth().height(height)) {
            if (cells.isEmpty() && priceLine.isEmpty()) return@Canvas

            val allTimes = (cells.map { it.timeMs } + priceLine.map { it.first })
            val allPrices = (cells.map { it.price } + priceLine.map { it.second })
            val minT = allTimes.minOrNull() ?: return@Canvas
            val maxT = allTimes.maxOrNull() ?: return@Canvas
            val minP = allPrices.minOrNull() ?: return@Canvas
            val maxP = allPrices.maxOrNull() ?: return@Canvas
            val tSpan = (maxT - minT).takeIf { it > 0 } ?: 1L
            val pSpan = (maxP - minP).takeIf { it > 0.0 } ?: 1.0

            val leftPad = 58.dp.toPx()
            val bottomPad = 4.dp.toPx()
            val plotW = size.width - leftPad
            val plotH = size.height - bottomPad
            if (plotW <= 0 || plotH <= 0) return@Canvas

            fun xOf(t: Long): Float = leftPad + ((t - minT).toFloat() / tSpan) * plotW
            fun yOf(p: Double): Float = (plotH - ((p - minP) / pSpan * plotH)).toFloat()

            val maxIntensity = cells.maxOfOrNull { it.intensity } ?: 1.0
            val cellSize = 6.dp.toPx()
            cells.forEach { cell ->
                val alphaFrac = (cell.intensity / maxIntensity).toFloat().coerceIn(0f, 1f)
                drawCircle(
                    color = liquidationColor(alphaFrac).copy(alpha = (0.15f + 0.7f * alphaFrac) * reveal),
                    radius = cellSize / 2f + alphaFrac * cellSize,
                    center = Offset(xOf(cell.timeMs), yOf(cell.price)),
                )
            }

            if (priceLine.size >= 2) {
                val path = Path()
                priceLine.forEachIndexed { i, (t, p) ->
                    val x = xOf(t)
                    val y = yOf(p)
                    if (i == 0) path.moveTo(x, y) else path.lineTo(x, y)
                }
                drawPath(
                    path = path,
                    color = colors.textPrimary.copy(alpha = 0.9f * reveal),
                    style = Stroke(width = 1.6.dp.toPx(), cap = StrokeCap.Round, join = StrokeJoin.Round),
                )
            }

            // Price-axis labels (4 evenly spaced ticks).
            for (i in 0..3) {
                val price = minP + pSpan * i / 3.0
                val y = yOf(price)
                val label = "%,.0f".format(price)
                val measured = textMeasurer.measure(label, style = TextStyle(fontSize = 9.sp, color = colors.plotTick))
                drawText(measured, topLeft = Offset(leftPad - measured.size.width - 6f, y - measured.size.height / 2f))
            }
        }
    }
}
