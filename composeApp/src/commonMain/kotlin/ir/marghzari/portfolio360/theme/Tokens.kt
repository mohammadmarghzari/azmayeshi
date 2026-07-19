package ir.marghzari.portfolio360.theme

import androidx.compose.ui.unit.Dp
import androidx.compose.ui.unit.dp

/**
 * Design-system spacing scale. Components and screens use these named steps instead of ad-hoc
 * dp literals so vertical rhythm and gutters stay consistent app-wide; migrate call sites to the
 * scale as they are touched rather than in one sweeping rename.
 */
object Spacing {
    val xs: Dp = 4.dp
    val sm: Dp = 8.dp
    val md: Dp = 12.dp
    val lg: Dp = 16.dp
    val xl: Dp = 20.dp
    val xxl: Dp = 24.dp
}

/** Corner-radius scale: sm for chips/badges, md for inputs, lg for cards, pill for buttons. */
object Radii {
    val sm: Dp = 10.dp
    val md: Dp = 14.dp
    val lg: Dp = 18.dp
    val pill: Dp = 999.dp
}

/** Shadow elevation scale: resting cards vs raised (dialog/sheet) surfaces. */
object Elevations {
    val card: Dp = 8.dp
    val raised: Dp = 16.dp
}
