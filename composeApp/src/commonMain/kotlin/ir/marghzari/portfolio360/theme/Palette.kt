package ir.marghzari.portfolio360.theme

import androidx.compose.ui.graphics.Color
import ir.marghzari.portfolio360.core.model.ChartPalette

/** The 15-color cyclic chart palette from app.py's `COLORS` list, as Compose [Color]s. */
val ChartSeriesColors: List<Color> = ChartPalette.map { Color(it) }

fun chartColor(index: Int): Color = ChartSeriesColors[index.mod(ChartSeriesColors.size)]
