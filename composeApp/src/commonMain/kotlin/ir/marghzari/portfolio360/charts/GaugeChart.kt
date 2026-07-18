package ir.marghzari.portfolio360.charts

import androidx.compose.foundation.Canvas
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.graphics.drawscope.Stroke
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import ir.marghzari.portfolio360.theme.LocalChartColors
import ir.marghzari.portfolio360.ui.motion.rememberChartReveal
import kotlin.math.cos
import kotlin.math.min
import kotlin.math.sin

data class GaugeZone(val from: Double, val to: Double, val color: Color)

/** Semi-circle gauge (mode="gauge+number" equivalent), used for the Fear & Greed index. */
@Composable
fun GaugeChart(
    value: Double,
    valueRange: ClosedFloatingPointRange<Double>,
    zones: List<GaugeZone>,
    needleColor: Color,
    modifier: Modifier = Modifier,
    centerLabel: String? = null,
    height: Dp = 200.dp,
) {
    val colors = LocalChartColors.current
    val textMeasurer = rememberTextMeasurer()
    val reveal by rememberChartReveal(value)
    Column(modifier = modifier.fillMaxWidth()) {
        Canvas(modifier = Modifier.fillMaxWidth().height(height)) {
            val diameter = min(size.width * 0.9f, size.height * 1.8f)
            val cx = size.width / 2f
            val cy = size.height * 0.86f
            val radius = diameter / 2f
            val strokeW = radius * 0.24f

            fun angleFor(v: Double): Float {
                val frac = ((v - valueRange.start) / (valueRange.endInclusive - valueRange.start)).coerceIn(0.0, 1.0)
                return (180.0 + frac * 180.0).toFloat()
            }

            zones.forEach { z ->
                val startAngle = angleFor(z.from)
                val sweep = angleFor(z.to) - startAngle
                drawArc(
                    color = z.color, startAngle = startAngle, sweepAngle = sweep, useCenter = false,
                    topLeft = Offset(cx - radius, cy - radius), size = Size(diameter, diameter),
                    style = Stroke(width = strokeW, cap = androidx.compose.ui.graphics.StrokeCap.Butt),
                )
            }

            // The needle sweeps in from the range's start to its true value as the gauge reveals.
            val animatedValue = valueRange.start + (value - valueRange.start) * reveal
            val needleAngleDeg = angleFor(animatedValue)
            val rad = Math.toRadians(needleAngleDeg.toDouble())
            val needleLen = radius * 0.86f
            val tipX = cx + (needleLen * cos(rad)).toFloat()
            val tipY = cy + (needleLen * sin(rad)).toFloat()
            drawLine(needleColor, Offset(cx, cy), Offset(tipX, tipY), strokeWidth = 4.5f, cap = androidx.compose.ui.graphics.StrokeCap.Round)
            drawCircle(needleColor, radius = 7f, center = Offset(cx, cy))

            val valueText = "%.0f".format(animatedValue)
            val measured = textMeasurer.measure(valueText, style = TextStyle(fontSize = 30.sp, color = colors.plotText))
            drawText(measured, topLeft = Offset(cx - measured.size.width / 2f, cy - radius * 0.55f))

            if (reveal > 0.7f) {
                centerLabel?.let {
                    val labelMeasured = textMeasurer.measure(it, style = TextStyle(fontSize = 13.sp, color = needleColor))
                    drawText(labelMeasured, topLeft = Offset(cx - labelMeasured.size.width / 2f, cy - radius * 0.55f + measured.size.height + 2f))
                }
            }
        }
    }
}
