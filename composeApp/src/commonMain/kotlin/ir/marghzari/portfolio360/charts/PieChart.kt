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
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import ir.marghzari.portfolio360.theme.LocalChartColors
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.sin

data class PieSlice(val label: String, val value: Double, val color: Color)

@Composable
fun DonutChart(
    slices: List<PieSlice>,
    modifier: Modifier = Modifier,
    title: String? = null,
    holeFraction: Float = 0.55f,
    height: Dp = 280.dp,
    showLegend: Boolean = true,
) {
    val colors = LocalChartColors.current
    val textMeasurer = rememberTextMeasurer()
    val total = slices.sumOf { it.value }.takeIf { it > 0 } ?: 1.0

    Column(modifier = modifier.fillMaxWidth()) {
        if (title != null) {
            Text(title, style = MaterialTheme.typography.titleSmall, color = colors.plotText, modifier = Modifier.padding(bottom = 6.dp))
        }
        Canvas(modifier = Modifier.fillMaxWidth().height(height)) {
            val diameter = min(size.width, size.height) * 0.86f
            val topLeft = Offset((size.width - diameter) / 2f, (size.height - diameter) / 2f)
            var startAngle = -90f
            slices.forEach { slice ->
                val sweep = (slice.value / total * 360.0).toFloat()
                drawArc(
                    color = slice.color, startAngle = startAngle, sweepAngle = sweep, useCenter = true,
                    topLeft = topLeft, size = Size(diameter, diameter),
                )
                startAngle += sweep
            }
            // Punch the hole.
            drawCircle(colors.plotBg, radius = diameter * holeFraction / 2f, center = Offset(size.width / 2f, size.height / 2f))

            // Percent labels for slices above a visibility threshold.
            startAngle = -90f
            val radius = diameter / 2f
            val labelRadius = (radius + diameter * holeFraction / 2f) / 2f
            slices.forEach { slice ->
                val sweep = (slice.value / total * 360.0).toFloat()
                if (sweep > 14f) {
                    val mid = Math.toRadians((startAngle + sweep / 2).toDouble())
                    val cx = size.width / 2f + labelRadius * cos(mid).toFloat()
                    val cy = size.height / 2f + labelRadius * sin(mid).toFloat()
                    val pct = "%.0f%%".format(slice.value / total * 100)
                    val measured = textMeasurer.measure(pct, style = TextStyle(fontSize = 11.sp, color = Color.White))
                    drawText(measured, topLeft = Offset(cx - measured.size.width / 2f, cy - measured.size.height / 2f))
                }
                startAngle += sweep
            }
        }
        if (showLegend) {
            ChartLegend(slices.map { "${it.label} ${(it.value / total * 100).let { p -> "%.1f".format(p) }}%" to it.color }, modifier = Modifier.padding(top = 6.dp))
        }
    }
}
