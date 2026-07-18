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
import androidx.compose.ui.graphics.lerp
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import ir.marghzari.portfolio360.theme.LocalChartColors
import kotlin.math.abs

/** Diverging red -> background -> green heatmap, used for correlation matrices and seasonality grids. */
@Composable
fun HeatmapChart(
    rowLabels: List<String>,
    colLabels: List<String>,
    values: Array<DoubleArray>, // [row][col]
    modifier: Modifier = Modifier,
    title: String? = null,
    valueRange: Pair<Double, Double> = -1.0 to 1.0,
    height: Dp = 320.dp,
    valueFormatter: (Double) -> String = { "%.2f".format(it) },
) {
    val colors = LocalChartColors.current
    val textMeasurer = rememberTextMeasurer()
    val (vMin, vMax) = valueRange
    val maxAbs = maxOf(abs(vMin), abs(vMax)).takeIf { it > 0 } ?: 1.0

    Column(modifier = modifier.fillMaxWidth()) {
        if (title != null) {
            Text(title, style = MaterialTheme.typography.titleSmall, color = colors.plotText, modifier = Modifier.padding(bottom = 6.dp))
        }
        Canvas(modifier = Modifier.fillMaxWidth().height(height)) {
            if (rowLabels.isEmpty() || colLabels.isEmpty()) return@Canvas
            val leftPad = 62.dp.toPx()
            val bottomPad = 46.dp.toPx()
            val topPad = 4.dp.toPx()
            val rightPad = 4.dp.toPx()
            val plotW = size.width - leftPad - rightPad
            val plotH = size.height - topPad - bottomPad
            if (plotW <= 0 || plotH <= 0) return@Canvas

            val cellW = plotW / colLabels.size
            val cellH = plotH / rowLabels.size

            for (r in rowLabels.indices) {
                for (c in colLabels.indices) {
                    val v = values.getOrNull(r)?.getOrNull(c) ?: 0.0
                    val frac = (v / maxAbs).coerceIn(-1.0, 1.0)
                    val color = if (frac >= 0) lerp(colors.plotBg, colors.green, frac.toFloat()) else lerp(colors.plotBg, colors.red, (-frac).toFloat())
                    val x = leftPad + cellW * c
                    val y = topPad + cellH * r
                    drawRect(color, topLeft = Offset(x, y), size = Size(cellW - 1.5f, cellH - 1.5f))
                    if (cellW > 30f && cellH > 20f) {
                        val label = valueFormatter(v)
                        val measured = textMeasurer.measure(label, style = TextStyle(fontSize = 9.sp, color = colors.plotText))
                        drawText(measured, topLeft = Offset(x + cellW / 2f - measured.size.width / 2f, y + cellH / 2f - measured.size.height / 2f))
                    }
                }
                val rowLabel = rowLabels[r]
                val measured = textMeasurer.measure(rowLabel, style = TextStyle(fontSize = 9.sp, color = colors.plotTick))
                drawText(measured, topLeft = Offset(leftPad - measured.size.width - 6f, topPad + cellH * r + cellH / 2f - measured.size.height / 2f))
            }
            for (c in colLabels.indices) {
                val label = colLabels[c]
                val measured = textMeasurer.measure(label, style = TextStyle(fontSize = 9.sp, color = colors.plotTick))
                drawText(
                    measured,
                    topLeft = Offset(leftPad + cellW * c + cellW / 2f - measured.size.width / 2f, topPad + plotH + 6f),
                )
            }
        }
    }
}
