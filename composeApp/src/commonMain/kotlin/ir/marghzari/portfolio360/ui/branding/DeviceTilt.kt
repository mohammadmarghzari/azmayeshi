package ir.marghzari.portfolio360.ui.branding

import androidx.compose.runtime.State
import androidx.compose.runtime.mutableStateOf
import androidx.compose.ui.geometry.Offset

/**
 * Ambient device tilt in degrees, clamped to a small range. Only Android actually updates this
 * (see `AndroidDeviceTilt.kt` in androidMain, a lightweight accelerometer listener installed once
 * from MainActivity) — Desktop has no motion sensor, so it simply stays at zero there.
 */
object DeviceTilt {
    private val internalDegrees = mutableStateOf(Offset.Zero)
    val degrees: State<Offset> get() = internalDegrees

    fun update(value: Offset) {
        internalDegrees.value = value
    }
}
