package ir.marghzari.portfolio360.ui.background

import ir.marghzari.portfolio360.generated.resources.Res
import ir.marghzari.portfolio360.generated.resources.bg_ancient_wisdom
import ir.marghzari.portfolio360.generated.resources.bg_bitcoin_king
import ir.marghzari.portfolio360.generated.resources.bg_breakthrough
import ir.marghzari.portfolio360.generated.resources.bg_crypto_orbit
import ir.marghzari.portfolio360.generated.resources.bg_eth_power
import ir.marghzari.portfolio360.generated.resources.bg_sisyphus
import ir.marghzari.portfolio360.generated.resources.bg_stairway
import ir.marghzari.portfolio360.generated.resources.bg_stoic
import ir.marghzari.portfolio360.generated.resources.bg_thinking
import ir.marghzari.portfolio360.generated.resources.bg_treasure_cave
import ir.marghzari.portfolio360.generated.resources.bg_vortex
import ir.marghzari.portfolio360.nav.Destination
import org.jetbrains.compose.resources.DrawableResource

/**
 * Assigns one of the eleven supplied crypto/Bitcoin artworks to each section, chosen for thematic
 * fit (e.g. the orbiting-logos piece for allocation/diversification, the man pushing a boulder for
 * risk & drawdown, the burning-book statue for the Black-Litterman model). A handful of images are
 * intentionally reused across less central screens since 16 destinations outnumber the 11 pieces.
 */
object BackgroundArt {
    val splash: DrawableResource get() = Res.drawable.bg_bitcoin_king

    fun forDestination(destination: Destination): DrawableResource = when (destination) {
        Destination.ALLOCATION -> Res.drawable.bg_crypto_orbit
        Destination.RISK_RETURN -> Res.drawable.bg_sisyphus
        Destination.PRICE_CHART -> Res.drawable.bg_vortex
        Destination.STYLE_COMPARE -> Res.drawable.bg_thinking
        Destination.EFFICIENT_FRONTIER -> Res.drawable.bg_eth_power
        Destination.ADVANCED_OPTIONS -> Res.drawable.bg_breakthrough
        Destination.BLACK_LITTERMAN -> Res.drawable.bg_ancient_wisdom
        Destination.STRESS_MC -> Res.drawable.bg_stairway
        Destination.REBALANCE -> Res.drawable.bg_treasure_cave
        Destination.BENCHMARK -> Res.drawable.bg_stoic
        Destination.LIVE_DATA -> Res.drawable.bg_bitcoin_king
        Destination.SAVE_PORTFOLIO -> Res.drawable.bg_treasure_cave
        Destination.ALERTS -> Res.drawable.bg_breakthrough
        Destination.IRAN_TOOLS -> Res.drawable.bg_vortex
        Destination.CERTIFICATES -> Res.drawable.bg_treasure_cave
        Destination.BOURSE_OPTIONS -> Res.drawable.bg_eth_power
        Destination.IME_LIVE -> Res.drawable.bg_crypto_orbit
    }
}
