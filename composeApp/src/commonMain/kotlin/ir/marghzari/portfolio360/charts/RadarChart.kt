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
import androidx.compose.ui.graphics.drawscope.Fill
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import ir.marghzari.portfolio360.theme.LocalChartColors
import ir.marghzari.portfolio360.ui.motion.rememberChartReveal
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.sin

data class RadarSeries(val name: String, val color: Color, val values: List<Double>)

/** Polygon radar/spider chart. [axisLabels] and every series' values must be the same length; values in [0, maxValue]. */
@Composable
fun RadarChart(
    axisLabels: List<String>,
    series: List<RadarSeries>,
    modifier: Modifier = Modifier,
    title: String? = null,
    maxValue: Double = 100.0,
    height: Dp = 300.dp,
    showLegend: Boolean = true,
) {
    val colors = LocalChartColors.current
    val textMeasurer = rememberTextMeasurer()
    val n = axisLabels.size
    val reveal by rememberChartReveal(series)

    Column(modifier = modifier.fillMaxWidth()) {
        if (title != null) {
            Text(title, style = MaterialTheme.typography.titleSmall, color = colors.plotText, modifier = Modifier.padding(bottom = 6.dp))
        }
        Canvas(modifier = Modifier.fillMaxWidth().height(height)) {
            if (n < 3) return@Canvas
            val cx = size.width / 2f
            val cy = size.height / 2f - 6f
            val radius = min(size.width, size.height) / 2f * 0.62f

            fun pointFor(axisIndex: Int, fraction: Double): Offset {
                val angle = -Math.PI / 2 + 2 * Math.PI * axisIndex / n
                val r = (radius * fraction.coerceIn(0.0, 1.2)).toFloat()
                return Offset(cx + (r * cos(angle)).toFloat(), cy + (r * sin(angle)).toFloat())
            }

            // Gridlines (25/50/75/100%)
            listOf(0.25, 0.5, 0.75, 1.0).forEach { frac ->
                val path = Path()
                for (i in 0 until n) {
                    val p = pointFor(i, frac)
                    if (i == 0) path.moveTo(p.x, p.y) else path.lineTo(p.x, p.y)
                }
                path.close()
                drawPath(path, colors.plotGrid, style = Stroke(width = 1f))
            }
            // Spokes + axis labels
            for (i in 0 until n) {
                val p = pointFor(i, 1.0)
                drawLine(colors.plotGrid, Offset(cx, cy), p, strokeWidth = 1f)
                val measured = textMeasurer.measure(axisLabels[i], style = TextStyle(fontSize = 10.sp, color = colors.plotTick))
                val labelPos = pointFor(i, 1.16)
                drawText(
                    measured,
                    topLeft = Offset((labelPos.x - measured.size.width / 2f).coerceIn(0f, size.width - measured.size.width), labelPos.y - measured.size.height / 2f),
                )
            }

            series.forEach { s ->
                val path = Path()
                for (i in 0 until n) {
                    val v = s.values.getOrElse(i) { 0.0 }
                    val p = pointFor(i, v / maxValue * reveal)
                    if (i == 0) path.moveTo(p.x, p.y) else path.lineTo(p.x, p.y)
                }
                path.close()
                drawPath(path, s.color.copy(alpha = 0.18f), style = Fill)
                drawPath(path, s.color, style = Stroke(width = 2f))
            }
        }
        if (showLegend && series.size > 1) {
            ChartLegend(series.map { it.name to it.color }, modifier = Modifier.padding(top = 6.dp))
        }
    }
}
