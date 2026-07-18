package ir.marghzari.portfolio360.ui.branding

import android.content.Context
import android.hardware.Sensor
import android.hardware.SensorEvent
import android.hardware.SensorEventListener
import android.hardware.SensorManager
import androidx.compose.runtime.Composable
import androidx.compose.runtime.DisposableEffect
import androidx.compose.ui.geometry.Offset
import androidx.compose.ui.platform.LocalContext
import kotlin.math.abs

private const val MAX_TILT_DEGREES = 6f
private const val GRAVITY = SensorManager.GRAVITY_EARTH

/**
 * Installs a single accelerometer listener for as long as this is in composition (call once, near
 * the app root) and feeds [DeviceTilt]. Runs at [SensorManager.SENSOR_DELAY_UI] (~60ms) and only
 * publishes a new value when it moved enough to matter, so icons reading [DeviceTilt] don't
 * recompose on every tiny sensor jitter — cheap enough to keep registered for the app's lifetime.
 */
@Composable
fun InstallDeviceTiltSensor() {
    val context = LocalContext.current
    DisposableEffect(context) {
        val sensorManager = context.getSystemService(Context.SENSOR_SERVICE) as? SensorManager
        val sensor = sensorManager?.getDefaultSensor(Sensor.TYPE_ACCELEROMETER)
        val listener = object : SensorEventListener {
            override fun onSensorChanged(event: SensorEvent) {
                val x = event.values.getOrElse(0) { 0f }
                val y = event.values.getOrElse(1) { 0f }
                val next = Offset(
                    (-x / GRAVITY * MAX_TILT_DEGREES).coerceIn(-MAX_TILT_DEGREES, MAX_TILT_DEGREES),
                    (y / GRAVITY * MAX_TILT_DEGREES).coerceIn(-MAX_TILT_DEGREES, MAX_TILT_DEGREES),
                )
                val current = DeviceTilt.degrees.value
                if (abs(next.x - current.x) > 0.15f || abs(next.y - current.y) > 0.15f) {
                    DeviceTilt.update(next)
                }
            }
            override fun onAccuracyChanged(sensor: Sensor?, accuracy: Int) {}
        }
        if (sensor != null) {
            sensorManager.registerListener(listener, sensor, SensorManager.SENSOR_DELAY_UI)
        }
        onDispose { sensorManager?.unregisterListener(listener) }
    }
}
