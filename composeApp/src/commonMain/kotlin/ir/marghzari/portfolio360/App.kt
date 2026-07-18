package ir.marghzari.portfolio360

import androidx.compose.animation.AnimatedContent
import androidx.compose.animation.core.tween
import androidx.compose.animation.fadeIn
import androidx.compose.animation.fadeOut
import androidx.compose.animation.scaleIn
import androidx.compose.animation.scaleOut
import androidx.compose.animation.slideInVertically
import androidx.compose.animation.togetherWith
import androidx.compose.animation.Crossfade
import androidx.compose.foundation.BorderStroke
import androidx.compose.foundation.ExperimentalFoundationApi
import androidx.compose.foundation.background
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Box
import androidx.compose.foundation.layout.BoxWithConstraints
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Row
import androidx.compose.foundation.layout.RowScope
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.lazy.LazyColumn
import androidx.compose.foundation.lazy.items
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.shape.RoundedCornerShape
import androidx.compose.foundation.verticalScroll
import androidx.compose.material.icons.Icons
import androidx.compose.material.icons.filled.DarkMode
import androidx.compose.material.icons.filled.LightMode
import androidx.compose.material.icons.filled.Menu
import androidx.compose.material.icons.filled.Animation
import androidx.compose.material.icons.filled.MotionPhotosOff
import androidx.compose.material3.Divider
import androidx.compose.material3.Icon
import androidx.compose.material3.IconButton
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.ModalNavigationDrawer
import androidx.compose.material3.ModalDrawerSheet
import androidx.compose.material3.NavigationBar
import androidx.compose.material3.NavigationBarItem
import androidx.compose.material3.NavigationRail
import androidx.compose.material3.NavigationRailItem
import androidx.compose.material3.NavigationDrawerItem
import androidx.compose.material3.Scaffold
import androidx.compose.material3.Text
import androidx.compose.material3.TopAppBar
import androidx.compose.material3.rememberDrawerState
import androidx.compose.material3.DrawerValue
import androidx.compose.runtime.Composable
import androidx.compose.runtime.CompositionLocalProvider
import androidx.compose.runtime.LaunchedEffect
import androidx.compose.runtime.remember
import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.rememberCoroutineScope
import androidx.compose.runtime.setValue
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.draw.clip
import androidx.compose.ui.unit.dp
import ir.marghzari.portfolio360.nav.Destination
import ir.marghzari.portfolio360.state.AppState
import ir.marghzari.portfolio360.theme.LocalChartColors
import ir.marghzari.portfolio360.theme.Portfolio360Theme
import ir.marghzari.portfolio360.ui.background.AnimatedBackground
import ir.marghzari.portfolio360.ui.background.BackgroundArt
import ir.marghzari.portfolio360.ui.branding.PremiumIconMotion
import ir.marghzari.portfolio360.ui.motion.LocalMotionClock
import ir.marghzari.portfolio360.ui.motion.LocalReducedMotion
import ir.marghzari.portfolio360.ui.motion.rememberMotionClock
import ir.marghzari.portfolio360.ui.screens.AllocationScreen
import ir.marghzari.portfolio360.ui.screens.AdvancedOptionsScreen
import ir.marghzari.portfolio360.ui.screens.AlertsScreen
import ir.marghzari.portfolio360.ui.screens.BenchmarkScreen
import ir.marghzari.portfolio360.ui.screens.BlackLittermanScreen
import ir.marghzari.portfolio360.ui.screens.BourseOptionsScreen
import ir.marghzari.portfolio360.ui.screens.CertificatesScreen
import ir.marghzari.portfolio360.ui.screens.EfficientFrontierScreen
import ir.marghzari.portfolio360.ui.screens.ImeLiveScreen
import ir.marghzari.portfolio360.ui.screens.IranToolsScreen
import ir.marghzari.portfolio360.ui.screens.LiveDataScreen
import ir.marghzari.portfolio360.ui.screens.PriceChartScreen
import ir.marghzari.portfolio360.ui.screens.RebalanceScreen
import ir.marghzari.portfolio360.ui.screens.RiskReturnScreen
import ir.marghzari.portfolio360.ui.screens.SavePortfolioScreen
import ir.marghzari.portfolio360.ui.screens.SplashScreen
import ir.marghzari.portfolio360.ui.screens.StressMonteCarloScreen
import ir.marghzari.portfolio360.ui.screens.StyleCompareScreen
import kotlinx.coroutines.delay
import kotlinx.coroutines.launch

private const val SPLASH_DURATION_MS = 2000

/** Root composable, shared verbatim across the Android and Desktop targets. */
@Composable
fun App() {
    val appState = remember { AppState() }
    var showSplash by remember { mutableStateOf(true) }
    var splashProgress by remember { mutableStateOf(0f) }

    LaunchedEffect(Unit) {
        val steps = 40
        repeat(steps) { i ->
            splashProgress = (i + 1) / steps.toFloat()
            delay((SPLASH_DURATION_MS / steps).toLong())
        }
        showSplash = false
    }

    Portfolio360Theme(darkTheme = appState.isDarkTheme) {
        val motionClock = rememberMotionClock()
        CompositionLocalProvider(
            LocalMotionClock provides motionClock,
            LocalReducedMotion provides appState.reducedMotion,
        ) {
            Crossfade(targetState = showSplash, animationSpec = tween(500), label = "splash-crossfade") { splashing ->
                if (splashing) {
                    SplashScreen(progress = splashProgress)
                } else {
                    BoxWithConstraints(modifier = Modifier.fillMaxSize()) {
                        val isWide = maxWidth >= 800.dp
                        if (isWide) {
                            WideLayout(appState)
                        } else {
                            CompactLayout(appState)
                        }
                    }
                }
            }
        }
    }
}

@Composable
private fun WideLayout(appState: AppState) {
    var selected by remember { mutableStateOf(Destination.ALLOCATION) }
    val colors = LocalChartColors.current
    Row(modifier = Modifier.fillMaxSize().background(colors.bg)) {
        NavigationRail(containerColor = colors.sidebarBg, contentColor = colors.textPrimary) {
            Column(
                modifier = Modifier.fillMaxSize().verticalScroll(rememberScrollState()).padding(vertical = 12.dp),
                horizontalAlignment = Alignment.CenterHorizontally,
                verticalArrangement = Arrangement.spacedBy(2.dp),
            ) {
                Text("P360", style = MaterialTheme.typography.titleMedium, color = colors.blueAccent, modifier = Modifier.padding(bottom = 12.dp))
                Destination.entries.forEach { dest ->
                    NavigationRailItem(
                        selected = dest == selected,
                        onClick = { selected = dest },
                        icon = {
                            PremiumIconMotion(
                                icon = dest.icon, contentDescription = dest.labelFa, active = dest == selected,
                                tint = if (dest == selected) colors.blueAccent else colors.muted,
                            )
                        },
                        label = { Text(dest.labelFa, style = MaterialTheme.typography.labelSmall) },
                    )
                }
                IconButton(onClick = { appState.isDarkTheme = !appState.isDarkTheme }, modifier = Modifier.padding(top = 12.dp)) {
                    Icon(if (appState.isDarkTheme) Icons.Filled.LightMode else Icons.Filled.DarkMode, contentDescription = "Theme")
                }
                IconButton(onClick = { appState.reducedMotion = !appState.reducedMotion }) {
                    Icon(
                        if (appState.reducedMotion) Icons.Filled.MotionPhotosOff else Icons.Filled.Animation,
                        contentDescription = "کاهش انیمیشن",
                        tint = if (appState.reducedMotion) colors.muted else colors.blueAccent,
                    )
                }
            }
        }
        Column(modifier = Modifier.fillMaxSize().padding(20.dp)) {
            ScreenHost(selected, appState)
        }
    }
}

@OptIn(ExperimentalFoundationApi::class, androidx.compose.material3.ExperimentalMaterial3Api::class)
@Composable
private fun CompactLayout(appState: AppState) {
    var selected by remember { mutableStateOf(Destination.ALLOCATION) }
    val colors = LocalChartColors.current
    val drawerState = rememberDrawerState(DrawerValue.Closed)
    val scope = rememberCoroutineScope()

    ModalNavigationDrawer(
        drawerState = drawerState,
        drawerContent = {
            ModalDrawerSheet(drawerContainerColor = colors.sidebarBg) {
                Text(
                    "Portfolio360", style = MaterialTheme.typography.titleLarge, color = colors.blueAccent,
                    modifier = Modifier.padding(20.dp),
                )
                LazyColumn {
                    items(Destination.entries) { dest ->
                        NavigationDrawerItem(
                            label = { Text(dest.labelFa) },
                            selected = dest == selected,
                            icon = {
                                PremiumIconMotion(
                                    icon = dest.icon, contentDescription = null, active = dest == selected,
                                    tint = if (dest == selected) colors.blueAccent else colors.muted,
                                )
                            },
                            onClick = { selected = dest; scope.launch { drawerState.close() } },
                            modifier = Modifier.padding(horizontal = 8.dp),
                        )
                    }
                }
            }
        },
    ) {
        Scaffold(
            topBar = {
                TopAppBar(
                    title = { Text(selected.labelFa) },
                    navigationIcon = {
                        IconButton(onClick = { scope.launch { drawerState.open() } }) {
                            Icon(Icons.Filled.Menu, contentDescription = "Menu")
                        }
                    },
                    actions = {
                        IconButton(onClick = { appState.isDarkTheme = !appState.isDarkTheme }) {
                            Icon(if (appState.isDarkTheme) Icons.Filled.LightMode else Icons.Filled.DarkMode, contentDescription = "Theme")
                        }
                        IconButton(onClick = { appState.reducedMotion = !appState.reducedMotion }) {
                            Icon(
                                if (appState.reducedMotion) Icons.Filled.MotionPhotosOff else Icons.Filled.Animation,
                                contentDescription = "کاهش انیمیشن",
                                tint = if (appState.reducedMotion) colors.muted else colors.blueAccent,
                            )
                        }
                    },
                )
            },
        ) { padding ->
            Column(modifier = Modifier.fillMaxSize().padding(padding).padding(16.dp)) {
                ScreenHost(selected, appState)
            }
        }
    }
}

@Composable
private fun ScreenHost(destination: Destination, appState: AppState) {
    AnimatedContent(
        targetState = destination,
        transitionSpec = {
            (fadeIn(tween(300)) + scaleIn(tween(300), initialScale = 0.97f) + slideInVertically(tween(300)) { it / 20 }) togetherWith
                (fadeOut(tween(250)) + scaleOut(tween(250), targetScale = 1.02f))
        },
        label = "screen-transition",
    ) { dest ->
        Box(modifier = Modifier.fillMaxSize().clip(RoundedCornerShape(22.dp))) {
            AnimatedBackground(image = BackgroundArt.forDestination(dest)) {
                Box(modifier = Modifier.fillMaxSize().padding(4.dp)) {
                    when (dest) {
                        Destination.ALLOCATION -> AllocationScreen(appState)
                        Destination.RISK_RETURN -> RiskReturnScreen(appState)
                        Destination.PRICE_CHART -> PriceChartScreen(appState)
                        Destination.STYLE_COMPARE -> StyleCompareScreen(appState)
                        Destination.EFFICIENT_FRONTIER -> EfficientFrontierScreen(appState)
                        Destination.ADVANCED_OPTIONS -> AdvancedOptionsScreen(appState)
                        Destination.BLACK_LITTERMAN -> BlackLittermanScreen(appState)
                        Destination.STRESS_MC -> StressMonteCarloScreen(appState)
                        Destination.REBALANCE -> RebalanceScreen(appState)
                        Destination.BENCHMARK -> BenchmarkScreen(appState)
                        Destination.LIVE_DATA -> LiveDataScreen(appState)
                        Destination.SAVE_PORTFOLIO -> SavePortfolioScreen(appState)
                        Destination.ALERTS -> AlertsScreen(appState)
                        Destination.IRAN_TOOLS -> IranToolsScreen(appState)
                        Destination.CERTIFICATES -> CertificatesScreen(appState)
                        Destination.BOURSE_OPTIONS -> BourseOptionsScreen(appState)
                        Destination.IME_LIVE -> ImeLiveScreen(appState)
                    }
                }
            }
        }
    }
}
