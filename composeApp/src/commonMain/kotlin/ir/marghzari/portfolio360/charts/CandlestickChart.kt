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
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import ir.marghzari.portfolio360.theme.LocalChartColors

data class Candle(val open: Double, val high: Double, val low: Double, val close: Double)

@Composable
fun CandlestickChart(
    candles: List<Candle>,
    modifier: Modifier = Modifier,
    title: String? = null,
    ma20: List<Double?>? = null,
    ma50: List<Double?>? = null,
    yFormatter: (Double) -> String = ::formatCompactNumber,
    height: Dp = 320.dp,
) {
    val colors = LocalChartColors.current
    val textMeasurer = rememberTextMeasurer()

    Column(modifier = modifier.fillMaxWidth()) {
        if (title != null) {
            Text(title, style = MaterialTheme.typography.titleSmall, color = colors.plotText, modifier = Modifier.padding(bottom = 6.dp))
        }
        Canvas(modifier = Modifier.fillMaxWidth().height(height)) {
            if (candles.isEmpty()) return@Canvas
            val dataMin = candles.minOf { it.low }
            val dataMax = candles.maxOf { it.high }
            val ticks = niceAxisTicks(dataMin, dataMax)

            val leftPad = 58.dp.toPx()
            val rightPad = 8.dp.toPx()
            val topPad = 10.dp.toPx()
            val bottomPad = 8.dp.toPx()
            val plotW = size.width - leftPad - rightPad
            val plotH = size.height - topPad - bottomPad
            if (plotW <= 0 || plotH <= 0) return@Canvas

            fun yToPx(v: Double): Float = topPad + plotH * (1f - ((v - ticks.min) / (ticks.max - ticks.min)).toFloat())
            val slot = plotW / candles.size
            val bodyWidth = (slot * 0.6f).coerceAtLeast(1f)

            ticks.values.forEach { t ->
                val y = yToPx(t)
                drawLine(colors.plotGrid, Offset(leftPad, y), Offset(size.width - rightPad, y), strokeWidth = 1f)
                val measured = textMeasurer.measure(yFormatter(t), style = TextStyle(fontSize = 10.sp, color = colors.plotTick))
                drawText(measured, topLeft = Offset(4.dp.toPx(), y - measured.size.height / 2f))
            }

            candles.forEachIndexed { i, c ->
                val cx = leftPad + slot * i + slot / 2f
                val up = c.close >= c.open
                val color = if (up) colors.green else colors.red
                drawLine(color, Offset(cx, yToPx(c.high)), Offset(cx, yToPx(c.low)), strokeWidth = 1.4f)
                val top = yToPx(maxOf(c.open, c.close))
                val bottom = yToPx(minOf(c.open, c.close))
                drawRect(color, topLeft = Offset(cx - bodyWidth / 2f, top), size = androidx.compose.ui.geometry.Size(bodyWidth, (bottom - top).coerceAtLeast(1.2f)))
            }

            fun drawMa(values: List<Double?>, color: Color) {
                var started = false
                val path = androidx.compose.ui.graphics.Path()
                values.forEachIndexed { i, v ->
                    if (v == null) return@forEachIndexed
                    val x = leftPad + slot * i + slot / 2f
                    val y = yToPx(v)
                    if (!started) { path.moveTo(x, y); started = true } else path.lineTo(x, y)
                }
                if (started) drawPath(path, color, style = androidx.compose.ui.graphics.drawscope.Stroke(width = 1.6f))
            }
            ma20?.let { drawMa(it, colors.blueAccent) }
            ma50?.let { drawMa(it, colors.gold) }
        }
    }
}

@Composable
fun VolumeBarChart(
    volumes: List<Double>,
    upFlags: List<Boolean>,
    modifier: Modifier = Modifier,
    title: String? = null,
    height: Dp = 130.dp,
) {
    val colors = LocalChartColors.current
    Column(modifier = modifier.fillMaxWidth()) {
        if (title != null) Text(title, style = MaterialTheme.typography.titleSmall, color = colors.plotText, modifier = Modifier.padding(bottom = 6.dp))
        Canvas(modifier = Modifier.fillMaxWidth().height(height)) {
            if (volumes.isEmpty()) return@Canvas
            val maxV = volumes.maxOrNull()?.takeIf { it > 0 } ?: 1.0
            val leftPad = 58.dp.toPx()
            val plotW = size.width - leftPad - 8.dp.toPx()
            val slot = plotW / volumes.size
            val barW = (slot * 0.6f).coerceAtLeast(1f)
            volumes.forEachIndexed { i, v ->
                val h = (v / maxV * size.height).toFloat()
                val x = leftPad + slot * i + slot / 2f
                val color = if (upFlags.getOrElse(i) { true }) colors.green else colors.red
                drawRect(color.copy(alpha = 0.75f), topLeft = Offset(x - barW / 2f, size.height - h), size = androidx.compose.ui.geometry.Size(barW, h))
            }
        }
    }
}
