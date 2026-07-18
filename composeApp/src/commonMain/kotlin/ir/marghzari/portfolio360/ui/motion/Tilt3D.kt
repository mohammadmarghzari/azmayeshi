package ir.marghzari.portfolio360.ui.motion

import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Modifier
import androidx.compose.ui.graphics.graphicsLayer
import ir.marghzari.portfolio360.ui.branding.DeviceTilt

/**
 * A small passive 3D tilt driven by the shared [DeviceTilt] reading (Android accelerometer; always
 * zero on Desktop, which has no motion sensor). Deliberately does NOT add its own drag/hover
 * gesture detector here: this is meant for cards sitting inside scrolling lists across all 16
 * screens, and a per-card drag detector would fight the list's own scroll gesture. The standalone
 * nav icon (`PremiumIconMotion`) isn't embedded in a long scrolling list the same way, so it still
 * keeps its own touch-drag tilt.
 */
@Composable
fun Modifier.tilt3D(maxDegrees: Float = 4f): Modifier {
    if (LocalReducedMotion.current) return this
    val deviceTilt by DeviceTilt.degrees
    val scale = maxDegrees / 6f
    val targetX by animateFloatAsState((deviceTilt.y * scale).coerceIn(-maxDegrees, maxDegrees), label = "tilt3d-x")
    val targetY by animateFloatAsState((deviceTilt.x * scale).coerceIn(-maxDegrees, maxDegrees), label = "tilt3d-y")
    return this.graphicsLayer {
        rotationX = targetX
        rotationY = targetY
        cameraDistance = 32f * density
    }
}
