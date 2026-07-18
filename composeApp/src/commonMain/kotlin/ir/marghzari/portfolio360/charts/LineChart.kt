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
import androidx.compose.runtime.remember
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.PathEffect
import androidx.compose.ui.graphics.StrokeCap
import androidx.compose.ui.graphics.drawscope.Fill
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.drawscope.clipRect
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import ir.marghzari.portfolio360.theme.LocalChartColors
import ir.marghzari.portfolio360.ui.motion.LocalMotionClock
import ir.marghzari.portfolio360.ui.motion.rememberChartReveal
import kotlin.math.sin

data class LineSeries(
    val name: String,
    val color: Color,
    val values: List<Double>,
    val fillToZero: Boolean = false,
    /** Index (within the same series list) of another series to fill the band against, e.g. a Monte Carlo percentile fan. */
    val fillToSeriesIndex: Int? = null,
    val fillAlpha: Float = 0.16f,
    val dashed: Boolean = false,
    val strokeWidth: Dp = 2.dp,
)

data class RefLine(val value: Double, val color: Color, val label: String? = null, val dashed: Boolean = true)

@Composable
fun LineChart(
    series: List<LineSeries>,
    modifier: Modifier = Modifier,
    title: String? = null,
    yFormatter: (Double) -> String = ::formatCompactNumber,
    xLabels: List<String>? = null,
    horizontalRefLines: List<RefLine> = emptyList(),
    /** When set, vertical reference lines are placed by interpolating this numeric x-domain (e.g. a price axis for payoff diagrams). */
    xValues: List<Double>? = null,
    verticalRefLines: List<RefLine> = emptyList(),
    height: Dp = 260.dp,
    showLegend: Boolean = true,
    yDomain: Pair<Double, Double>? = null,
) {
    val colors = LocalChartColors.current
    val textMeasurer = rememberTextMeasurer()
    val reveal by rememberChartReveal(series)
    // Read only in the Canvas draw phase below (not here in the composable body) so the shared
    // clock's per-frame ticks trigger a cheap re-draw of this chart, not a full recomposition.
    val motionClock = LocalMotionClock.current

    Column(modifier = modifier.fillMaxWidth()) {
        if (title != null) {
            Text(title, style = MaterialTheme.typography.titleSmall, color = colors.plotText, modifier = Modifier.padding(bottom = 6.dp))
        }
        Canvas(modifier = Modifier.fillMaxWidth().height(height)) {
            val pulse = motionClock?.let { sin(it.time.value * 6.2831855f * 3f) * 0.5f + 0.5f } ?: 0.5f
            val allValues = series.flatMap { it.values } + horizontalRefLines.map { it.value }
            if (allValues.isEmpty()) return@Canvas
            val dataMin = yDomain?.first ?: allValues.min()
            val dataMax = yDomain?.second ?: allValues.max()
            val ticks = niceAxisTicks(dataMin, dataMax)

            val leftPad = 54.dp.toPx()
            val rightPad = 8.dp.toPx()
            val topPad = 10.dp.toPx()
            val bottomPad = if (xLabels != null || xValues != null) 22.dp.toPx() else 8.dp.toPx()
            val plotW = size.width - leftPad - rightPad
            val plotH = size.height - topPad - bottomPad
            if (plotW <= 0 || plotH <= 0) return@Canvas

            fun yToPx(v: Double): Float = (topPad + plotH * (1f - ((v - ticks.min) / (ticks.max - ticks.min)).toFloat())).let {
                if (it.isFinite()) it else topPad + plotH
            }
            val maxN = (series.maxOfOrNull { it.values.size } ?: 1).coerceAtLeast(1)
            fun xToPx(i: Int): Float = leftPad + plotW * (if (maxN <= 1) 0f else i.toFloat() / (maxN - 1).toFloat())

            // Gridlines + y-axis labels
            ticks.values.forEach { t ->
                val y = yToPx(t)
                drawLine(colors.plotGrid, Offset(leftPad, y), Offset(size.width - rightPad, y), strokeWidth = 1f)
                val label = yFormatter(t)
                val measured = textMeasurer.measure(label, style = TextStyle(fontSize = 10.sp, color = colors.plotTick))
                drawText(measured, topLeft = Offset(4.dp.toPx(), y - measured.size.height / 2f))
            }

            // Everything data-derived (fills/lines/ref-lines/markers) draws inside the reveal clip,
            // so the whole plot appears to draw itself in left-to-right rather than popping in.
            clipRect(left = 0f, top = 0f, right = leftPad + plotW * reveal, bottom = size.height) {
                // Horizontal reference lines
                horizontalRefLines.forEach { ref ->
                    val y = yToPx(ref.value)
                    val effect = if (ref.dashed) PathEffect.dashPathEffect(floatArrayOf(8f, 6f)) else null
                    drawLine(ref.color.copy(alpha = 0.75f), Offset(leftPad, y), Offset(size.width - rightPad, y), strokeWidth = 1.4f, pathEffect = effect)
                    // Energy-ring-style pulsing marker at the target/ref line.
                    drawCircle(ref.color, radius = (3.5f + 1.5f * pulse), center = Offset(size.width - rightPad - 6f, y), alpha = 0.5f + 0.3f * pulse)
                    ref.label?.let { lbl ->
                        val measured = textMeasurer.measure(lbl, style = TextStyle(fontSize = 9.sp, color = ref.color))
                        drawText(measured, topLeft = Offset(size.width - rightPad - measured.size.width - 4f, y - measured.size.height - 2f))
                    }
                }

                // Series fills first (so lines draw on top)
                series.forEach { s ->
                    if (s.values.isEmpty()) return@forEach
                    if (s.fillToZero) {
                        val zeroY = yToPx(0.0).coerceIn(topPad, topPad + plotH)
                        val path = androidx.compose.ui.graphics.Path().apply {
                            moveTo(xToPx(0), zeroY)
                            s.values.forEachIndexed { i, v -> lineTo(xToPx(i), yToPx(v)) }
                            lineTo(xToPx(s.values.size - 1), zeroY)
                            close()
                        }
                        drawPath(path, s.color.copy(alpha = s.fillAlpha), style = Fill)
                    } else if (s.fillToSeriesIndex != null) {
                        val other = series.getOrNull(s.fillToSeriesIndex) ?: return@forEach
                        val n = minOf(s.values.size, other.values.size)
                        if (n < 2) return@forEach
                        val path = androidx.compose.ui.graphics.Path().apply {
                            moveTo(xToPx(0), yToPx(s.values[0]))
                            for (i in 1 until n) lineTo(xToPx(i), yToPx(s.values[i]))
                            for (i in n - 1 downTo 0) lineTo(xToPx(i), yToPx(other.values[i]))
                            close()
                        }
                        drawPath(path, s.color.copy(alpha = s.fillAlpha), style = Fill)
                    }
                }

                // Series lines + a pulsing "current value" marker at each series' last point
                series.forEach { s ->
                    if (s.values.size < 2) return@forEach
                    val path = androidx.compose.ui.graphics.Path().apply {
                        moveTo(xToPx(0), yToPx(s.values[0]))
                        for (i in 1 until s.values.size) lineTo(xToPx(i), yToPx(s.values[i]))
                    }
                    val effect = if (s.dashed) PathEffect.dashPathEffect(floatArrayOf(10f, 6f)) else null
                    drawPath(
                        path, s.color,
                        style = Stroke(width = s.strokeWidth.toPx(), cap = StrokeCap.Round, pathEffect = effect),
                    )
                    if (!s.dashed) {
                        val lastX = xToPx(s.values.size - 1)
                        val lastY = yToPx(s.values.last())
                        drawCircle(s.color.copy(alpha = 0.25f + 0.20f * pulse), radius = (5f + 3f * pulse), center = Offset(lastX, lastY))
                        drawCircle(s.color, radius = 3f, center = Offset(lastX, lastY))
                    }
                }

                // Vertical reference lines, positioned via [xValues] interpolation (e.g. Strike/Spot/Breakeven markers).
                if (xValues != null && xValues.size >= 2) {
                    val xMin = xValues.first()
                    val xMax = xValues.last()
                    verticalRefLines.forEach { ref ->
                        val frac = ((ref.value - xMin) / (xMax - xMin)).coerceIn(0.0, 1.0)
                        val x = leftPad + plotW * frac.toFloat()
                        val effect = if (ref.dashed) PathEffect.dashPathEffect(floatArrayOf(8f, 6f)) else null
                        drawLine(ref.color.copy(alpha = 0.8f), Offset(x, topPad), Offset(x, topPad + plotH), strokeWidth = 1.4f, pathEffect = effect)
                        ref.label?.let { lbl ->
                            val measured = textMeasurer.measure(lbl, style = TextStyle(fontSize = 9.sp, color = ref.color))
                            drawText(measured, topLeft = Offset((x - measured.size.width / 2f).coerceIn(0f, size.width - measured.size.width), topPad + 2f))
                        }
                    }
                }
            }

            // X labels (sparse: first, middle, last)
            xLabels?.let { labels ->
                if (labels.isEmpty()) return@let
                val idxs = if (labels.size <= 6) labels.indices.toList() else listOf(0, labels.size / 4, labels.size / 2, labels.size * 3 / 4, labels.size - 1)
                idxs.distinct().forEach { i ->
                    val label = labels.getOrNull((i.toDouble() / (labels.size - 1).coerceAtLeast(1) * (maxN - 1)).toInt().coerceIn(0, labels.size - 1)) ?: return@forEach
                    val measured = textMeasurer.measure(label, style = TextStyle(fontSize = 9.sp, color = colors.plotTick))
                    val x = xToPx((i.toDouble() / (labels.size - 1).coerceAtLeast(1) * (maxN - 1)).toInt())
                    drawText(measured, topLeft = Offset((x - measured.size.width / 2f).coerceIn(0f, size.width - measured.size.width), size.height - bottomPad + 4f))
                }
            } ?: run {
                if (xValues != null && xValues.size >= 2 && bottomPad > 12.dp.toPx()) {
                    val xMin = xValues.first(); val xMax = xValues.last()
                    listOf(0.0, 0.25, 0.5, 0.75, 1.0).forEach { frac ->
                        val v = xMin + (xMax - xMin) * frac
                        val x = leftPad + plotW * frac.toFloat()
                        val measured = textMeasurer.measure(formatCompactNumber(v), style = TextStyle(fontSize = 9.sp, color = colors.plotTick))
                        drawText(measured, topLeft = Offset((x - measured.size.width / 2f).coerceIn(0f, size.width - measured.size.width), size.height - bottomPad + 4f))
                    }
                }
            }
        }
        if (showLegend && series.size > 1) {
            ChartLegend(series.map { it.name to it.color }, modifier = Modifier.padding(top = 6.dp))
        }
    }
}
