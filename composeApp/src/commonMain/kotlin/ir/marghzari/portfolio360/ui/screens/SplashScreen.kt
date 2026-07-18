package ir.marghzari.portfolio360.ui.screens

import androidx.compose.animation.core.FastOutSlowInEasing
import androidx.compose.animation.core.RepeatMode
import androidx.compose.animation.core.animateFloat
import androidx.compose.animation.core.animateFloatAsState
import androidx.compose.animation.core.infiniteRepeatable
import androidx.compose.animation.core.rememberInfiniteTransition
import androidx.compose.animation.core.tween
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.fillMaxWidth
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.material3.LinearProgressIndicator
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.runtime.getValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.scale
import androidx.compose.ui.graphics.Color
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.text.style.TextAlign
import androidx.compose.ui.unit.dp
import androidx.compose.ui.unit.sp
import ir.marghzari.portfolio360.ui.background.AnimatedBackground
import ir.marghzari.portfolio360.ui.background.BackgroundArt

/** Premium animated splash shown for a couple of seconds while the app spins up. */
@Composable
fun SplashScreen(progress: Float) {
    val transition = rememberInfiniteTransition(label = "splash-glow")
    val glow by transition.animateFloat(
        initialValue = 0.55f,
        targetValue = 1f,
        animationSpec = infiniteRepeatable(tween(1500, easing = FastOutSlowInEasing), RepeatMode.Reverse),
        label = "splash-glow-value",
    )
    val entrance by animateFloatAsState(
        targetValue = 1f,
        animationSpec = tween(1100, easing = FastOutSlowInEasing),
        label = "splash-entrance",
    )

    AnimatedBackground(image = BackgroundArt.splash, intensity = 1.5f, scrimAlpha = 0.62f) {
        Column(
            modifier = Modifier.fillMaxSize().padding(32.dp),
            horizontalAlignment = Alignment.CenterHorizontally,
            verticalArrangement = Arrangement.Center,
        ) {
            Text(
                "PORTFOLIO 360",
                fontSize = 30.sp,
                fontWeight = FontWeight.Bold,
                letterSpacing = 4.sp,
                color = Color(0xFFF4E4B0).copy(alpha = glow),
                modifier = Modifier.scale(0.92f + entrance * 0.08f),
            )
            androidx.compose.foundation.layout.Spacer(Modifier.height(12.dp))
            Text(
                "تحلیل حرفه‌ای پرتفوی، اختیار معامله و بازار ایران",
                style = MaterialTheme.typography.bodyMedium,
                color = Color.White.copy(alpha = 0.78f * entrance),
                textAlign = TextAlign.Center,
            )
            androidx.compose.foundation.layout.Spacer(Modifier.height(40.dp))
            LinearProgressIndicator(
                progress = progress,
                modifier = Modifier.fillMaxWidth(0.55f).height(3.dp),
                color = Color(0xFFE8C87A),
                trackColor = Color.White.copy(alpha = 0.12f),
            )
        }
    }
}
