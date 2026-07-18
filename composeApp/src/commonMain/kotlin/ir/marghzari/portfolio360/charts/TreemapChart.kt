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
import androidx.compose.ui.geometry.Rect
import androidx.compose.ui.geometry.Size
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.drawText
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.rememberTextMeasurer
import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import ir.marghzari.portfolio360.theme.LocalChartColors
import kotlin.math.max
import kotlin.math.min

data class TreemapItem(val label: String, val value: Double, val color: Color)

private fun worstRatio(row: List<Double>, length: Double): Double {
    val sum = row.sum()
    val maxV = row.max()
    val minV = row.min()
    val sq = length * length
    return max((sq * maxV) / (sum * sum), (sum * sum) / (sq * minV))
}

/** Classic squarified-treemap layout (Bruls, Huizing, van Wijk 2000) over a fixed pixel rect. */
private fun squarify(values: List<Double>, rect: Rect, out: MutableList<Rect>) {
    if (values.isEmpty()) return
    if (values.size == 1) {
        out.add(rect)
        return
    }
    var remaining = values
    var currentRect = rect
    val row = mutableListOf<Double>()

    while (remaining.isNotEmpty()) {
        val length = min(currentRect.width, currentRect.height).toDouble()
        val candidate = remaining.first()
        if (row.isEmpty() || worstRatio(row + candidate, length) <= worstRatio(row, length)) {
            row.add(candidate)
            remaining = remaining.drop(1)
        } else {
            currentRect = layoutRow(row, currentRect, out)
            row.clear()
        }
        if (remaining.isEmpty() && row.isNotEmpty()) {
            currentRect = layoutRow(row, currentRect, out)
            row.clear()
        }
    }
}

private fun layoutRow(row: List<Double>, rect: Rect, out: MutableList<Rect>): Rect {
    val rowSum = row.sum()
    val horizontal = rect.width >= rect.height
    return if (horizontal) {
        val rowHeight = if (rect.width > 0) (rowSum / rect.width).toFloat() else 0f
        var x = rect.left
        row.forEach { v ->
            val w = if (rowHeight > 0) (v / rowHeight).toFloat() else 0f
            out.add(Rect(x, rect.top, x + w, rect.top + rowHeight))
            x += w
        }
        Rect(rect.left, rect.top + rowHeight, rect.right, rect.bottom)
    } else {
        val rowWidth = if (rect.height > 0) (rowSum / rect.height).toFloat() else 0f
        var y = rect.top
        row.forEach { v ->
            val h = if (rowWidth > 0) (v / rowWidth).toFloat() else 0f
            out.add(Rect(rect.left, y, rect.left + rowWidth, y + h))
            y += h
        }
        Rect(rect.left + rowWidth, rect.top, rect.right, rect.bottom)
    }
}

@Composable
fun TreemapChart(
    items: List<TreemapItem>,
    modifier: Modifier = Modifier,
    title: String? = null,
    height: Dp = 300.dp,
) {
    val colors = LocalChartColors.current
    val textMeasurer = rememberTextMeasurer()
    val sorted = items.sortedByDescending { it.value }
    val total = sorted.sumOf { it.value }.takeIf { it > 0 } ?: 1.0

    Column(modifier = modifier.fillMaxWidth()) {
        if (title != null) {
            Text(title, style = MaterialTheme.typography.titleSmall, color = colors.plotText, modifier = Modifier.padding(bottom = 6.dp))
        }
        Canvas(modifier = Modifier.fillMaxWidth().height(height)) {
            if (sorted.isEmpty()) return@Canvas
            // Normalize values to the pixel-area budget so the squarify algorithm's arithmetic stays well-scaled.
            val area = size.width.toDouble() * size.height.toDouble()
            val scaled = sorted.map { it.value / total * area }
            val rects = mutableListOf<Rect>()
            squarify(scaled, Rect(0f, 0f, size.width, size.height), rects)

            sorted.forEachIndexed { i, item ->
                val r = rects.getOrNull(i) ?: return@forEachIndexed
                drawRect(item.color, topLeft = Offset(r.left, r.top), size = Size(max(r.width - 2f, 0f), max(r.height - 2f, 0f)))
                if (r.width > 46f && r.height > 26f) {
                    val pct = "%.1f%%".format(item.value / total * 100)
                    val labelMeasured = textMeasurer.measure(item.label, style = TextStyle(fontSize = 11.sp, color = Color.White))
                    val pctMeasured = textMeasurer.measure(pct, style = TextStyle(fontSize = 10.sp, color = Color.White.copy(alpha = 0.85f)))
                    drawText(labelMeasured, topLeft = Offset(r.left + 6f, r.top + 6f))
                    if (r.height > 44f) drawText(pctMeasured, topLeft = Offset(r.left + 6f, r.top + 6f + labelMeasured.size.height + 2f))
                }
            }
        }
    }
}
