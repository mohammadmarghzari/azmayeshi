package ir.marghzari.portfolio360.charts

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.graphics.lerp
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import ir.marghzari.portfolio360.theme.LocalChartColors

data class ScatterPoint(val x: Double, val y: Double, val colorValue: Double = 0.0)
data class ScatterMarker(val x: Double, val y: Double, val label: String, val color: Color)

/** Point cloud (e.g. simulated portfolios colored by Sharpe) + an optional connecting line (efficient frontier) + highlighted markers. */
@Composable
fun ScatterChart(
    points: List<ScatterPoint>,
    modifier: Modifier = Modifier,
    title: String? = null,
    colorLow: Color = Color(0xFFCC5555),
    colorHigh: Color = Color(0xFF5AAA78),
    colorRange: Pair<Double, Double>? = null,
    lineOverlay: List<Pair<Double, Double>>? = null,
    lineColor: Color = Color(0xFF5B9BD5),
    markers: List<ScatterMarker> = emptyList(),
    xFormatter: (Double) -> String = ::formatCompactNumber,
    yFormatter: (Double) -> String = ::formatCompactNumber,
    height: Dp = 420.dp,
) {
    val colors = LocalChartColors.current
    val textMeasurer = rememberTextMeasurer()

    Column(modifier = modifier.fillMaxWidth()) {
        if (title != null) {
            Text(title, style = MaterialTheme.typography.titleSmall, color = colors.plotText, modifier = Modifier.padding(bottom = 6.dp))
        }
        Canvas(modifier = Modifier.fillMaxWidth().height(height)) {
            if (points.isEmpty()) return@Canvas
            val allX = points.map { it.x } + (lineOverlay?.map { it.first } ?: emptyList()) + markers.map { it.x }
            val allY = points.map { it.y } + (lineOverlay?.map { it.second } ?: emptyList()) + markers.map { it.y }
            val xTicks = niceAxisTicks(allX.min(), allX.max())
            val yTicks = niceAxisTicks(allY.min(), allY.max())

            val leftPad = 54.dp.toPx(); val rightPad = 8.dp.toPx(); val topPad = 10.dp.toPx(); val bottomPad = 24.dp.toPx()
            val plotW = size.width - leftPad - rightPad
            val plotH = size.height - topPad - bottomPad
            if (plotW <= 0 || plotH <= 0) return@Canvas

            fun xToPx(v: Double): Float = leftPad + plotW * ((v - xTicks.min) / (xTicks.max - xTicks.min)).toFloat()
            fun yToPx(v: Double): Float = topPad + plotH * (1f - ((v - yTicks.min) / (yTicks.max - yTicks.min)).toFloat())

            yTicks.values.forEach { t ->
                val y = yToPx(t)
                drawLine(colors.plotGrid, Offset(leftPad, y), Offset(size.width - rightPad, y), strokeWidth = 1f)
                val measured = textMeasurer.measure(yFormatter(t), style = TextStyle(fontSize = 10.sp, color = colors.plotTick))
                drawText(measured, topLeft = Offset(4.dp.toPx(), y - measured.size.height / 2f))
            }
            xTicks.values.forEach { t ->
                val x = xToPx(t)
                val measured = textMeasurer.measure(xFormatter(t), style = TextStyle(fontSize = 9.sp, color = colors.plotTick))
                drawText(measured, topLeft = Offset(x - measured.size.width / 2f, size.height - bottomPad + 4f))
            }

            val (cLow, cHigh) = colorRange ?: (points.minOf { it.colorValue } to points.maxOf { it.colorValue })
            points.forEach { p ->
                val frac = if (cHigh > cLow) ((p.colorValue - cLow) / (cHigh - cLow)).coerceIn(0.0, 1.0) else 0.5
                val color = lerp(colorLow, colorHigh, frac.toFloat())
                drawCircle(color.copy(alpha = 0.55f), radius = 2.6f, center = Offset(xToPx(p.x), yToPx(p.y)))
            }

            lineOverlay?.let { line ->
                if (line.size >= 2) {
                    val path = androidx.compose.ui.graphics.Path().apply {
                        moveTo(xToPx(line[0].first), yToPx(line[0].second))
                        for (i in 1 until line.size) lineTo(xToPx(line[i].first), yToPx(line[i].second))
                    }
                    drawPath(path, lineColor, style = Stroke(width = 2.4f))
                }
            }

            markers.forEach { m ->
                val center = Offset(xToPx(m.x), yToPx(m.y))
                drawCircle(m.color, radius = 6f, center = center)
                drawCircle(colors.plotBg, radius = 6f, center = center, style = Stroke(width = 1.5f))
                val measured = textMeasurer.measure(m.label, style = TextStyle(fontSize = 10.sp, color = m.color))
                drawText(measured, topLeft = Offset(center.x + 8f, center.y - measured.size.height / 2f))
            }
        }
    }
}
