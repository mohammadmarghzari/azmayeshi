package ir.marghzari.portfolio360.ui.background

import androidx.compose.runtime.compositionLocalOf
import dev.chrisbanes.haze.HazeState

/**
 * The current screen's [HazeState], set by [AnimatedBackground] so that cards drawn over it (via
 * `Modifier.hazeChild`) show a genuine blurred glimpse of the background photo — true frosted
 * glass — instead of just a translucent flat tint. Null outside any [AnimatedBackground] (e.g. an
 * isolated preview), in which case callers should fall back to a plain translucent background.
 */
val LocalHazeState = compositionLocalOf<HazeState?> { null }
