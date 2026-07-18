package ir.marghzari.portfolio360.desktop

import androidx.compose.ui.window.Window
import androidx.compose.ui.window.application
import androidx.compose.ui.window.rememberWindowState
import androidx.compose.ui.unit.DpSize
import androidx.compose.ui.unit.dp
import ir.marghzari.portfolio360.App

fun main() = application {
    val windowState = rememberWindowState(size = DpSize(1280.dp, 860.dp))
    Window(onCloseRequest = ::exitApplication, title = "Portfolio360", state = windowState) {
        App()
    }
}
