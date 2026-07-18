package ir.marghzari.portfolio360

import android.content.Intent
import android.os.Bundle
import android.os.Process
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.enableEdgeToEdge
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.ui.Modifier
import androidx.compose.ui.unit.dp
import ir.marghzari.portfolio360.theme.Portfolio360Theme
import ir.marghzari.portfolio360.ui.components.CrashScreen
import kotlin.system.exitProcess

/**
 * Compose forbids try/catch around composable calls, so a screen that throws would otherwise
 * take down the whole Activity with no way to see why. Instead, any uncaught exception is
 * captured here, the app relaunches itself with the stack trace attached, and that fresh launch
 * renders a plain (crash-safe) error screen instead of the screen that failed — so instead of
 * being kicked out silently, you get the exact error to screenshot and report.
 */
class MainActivity : ComponentActivity() {
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        enableEdgeToEdge()

        val crashTrace = intent?.getStringExtra(EXTRA_CRASH_TRACE)

        if (crashTrace == null) {
            Thread.setDefaultUncaughtExceptionHandler { _, throwable ->
                try {
                    val restartIntent = Intent(this, MainActivity::class.java).apply {
                        addFlags(Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TASK)
                        putExtra(EXTRA_CRASH_TRACE, throwable.stackTraceToString())
                    }
                    startActivity(restartIntent)
                } catch (e: Throwable) {
                    // Best-effort: if even the restart fails, just fall through to process death below.
                }
                Process.killProcess(Process.myPid())
                exitProcess(1)
            }
        }

        setContent {
            if (crashTrace != null) {
                Portfolio360Theme {
                    Box(modifier = Modifier.fillMaxSize().padding(16.dp)) {
                        CrashScreen("برنامه", crashTrace) {
                            startActivity(Intent(this@MainActivity, MainActivity::class.java))
                            finish()
                        }
                    }
                }
            } else {
                App()
            }
        }
    }

    companion object {
        private const val EXTRA_CRASH_TRACE = "crash_trace"
    }
}
