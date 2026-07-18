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
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import ir.marghzari.portfolio360.theme.LocalChartColors

data class BarSeries(val name: String, val color: Color, val values: List<Double>)

/** Grouped/single categorical bar chart. [categories] length must match each series' value count. */
@Composable
fun BarChart(
    categories: List<String>,
    series: List<BarSeries>,
    modifier: Modifier = Modifier,
    title: String? = null,
    yFormatter: (Double) -> String = ::formatCompactNumber,
    height: Dp = 260.dp,
    showLegend: Boolean = true,
    zeroLine: Boolean = true,
    perBarColorOverride: ((seriesIndex: Int, barIndex: Int, value: Double) -> Color)? = null,
) {
    val colors = LocalChartColors.current
    val textMeasurer = rememberTextMeasurer()

    Column(modifier = modifier.fillMaxWidth()) {
        if (title != null) {
            Text(title, style = MaterialTheme.typography.titleSmall, color = colors.plotText, modifier = Modifier.padding(bottom = 6.dp))
        }
        Canvas(modifier = Modifier.fillMaxWidth().height(height)) {
            if (categories.isEmpty() || series.isEmpty()) return@Canvas
            val allValues = series.flatMap { it.values }
            val dataMin = minOf(0.0, allValues.minOrNull() ?: 0.0)
            val dataMax = maxOf(0.0, allValues.maxOrNull() ?: 0.0)
            val ticks = niceAxisTicks(dataMin, dataMax)

            val leftPad = 54.dp.toPx()
            val rightPad = 8.dp.toPx()
            val topPad = 10.dp.toPx()
            val bottomPad = 26.dp.toPx()
            val plotW = size.width - leftPad - rightPad
            val plotH = size.height - topPad - bottomPad
            if (plotW <= 0 || plotH <= 0) return@Canvas

            fun yToPx(v: Double): Float = topPad + plotH * (1f - ((v - ticks.min) / (ticks.max - ticks.min)).toFloat())

            ticks.values.forEach { t ->
                val y = yToPx(t)
                drawLine(colors.plotGrid, Offset(leftPad, y), Offset(size.width - rightPad, y), strokeWidth = 1f)
                val label = yFormatter(t)
                val measured = textMeasurer.measure(label, style = TextStyle(fontSize = 10.sp, color = colors.plotTick))
                drawText(measured, topLeft = Offset(4.dp.toPx(), y - measured.size.height / 2f))
            }

            val groupWidth = plotW / categories.size
            val barGap = groupWidth * 0.15f
            val innerWidth = (groupWidth - barGap) / series.size

            for (c in categories.indices) {
                val groupLeft = leftPad + groupWidth * c
                for ((si, s) in series.withIndex()) {
                    val v = s.values.getOrElse(c) { 0.0 }
                    val barLeft = groupLeft + barGap / 2 + innerWidth * si
                    val yTop = yToPx(maxOf(0.0, v))
                    val yBase = yToPx(minOf(0.0, v))
                    val color = perBarColorOverride?.invoke(si, c, v) ?: s.color
                    drawRect(color, topLeft = Offset(barLeft + innerWidth * 0.08f, yTop), size = androidx.compose.ui.geometry.Size(innerWidth * 0.84f, (yBase - yTop).coerceAtLeast(1.5f)))
                }
                val label = categories[c]
                if (categories.size <= 14) {
                    val measured = textMeasurer.measure(label, style = TextStyle(fontSize = 9.sp, color = colors.plotTick))
                    val cx = groupLeft + groupWidth / 2f
                    drawText(measured, topLeft = Offset((cx - measured.size.width / 2f).coerceIn(0f, size.width - measured.size.width), size.height - bottomPad + 6f))
                }
            }

            if (zeroLine) {
                val y0 = yToPx(0.0)
                drawLine(colors.plotTick.copy(alpha = 0.5f), Offset(leftPad, y0), Offset(size.width - rightPad, y0), strokeWidth = 1.2f)
            }
        }
        if (showLegend && series.size > 1) {
            ChartLegend(series.map { it.name to it.color }, modifier = Modifier.padding(top = 6.dp))
        }
    }
}
