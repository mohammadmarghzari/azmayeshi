package ir.marghzari.portfolio360.core.math

import kotlin.math.PI
import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.sqrt

/** Standard normal PDF/CDF, used throughout the Black-Scholes pricing/greeks code. */
object NormalDistribution {

    fun pdf(x: Double): Double = exp(-0.5 * x * x) / sqrt(2.0 * PI)

    /** Abramowitz & Stegun 7.1.26 erf approximation (max error ~1.5e-7) via the standard CDF identity. */
    fun cdf(x: Double): Double {
        val sign = if (x < 0) -1.0 else 1.0
        val ax = abs(x) / sqrt(2.0)
        val t = 1.0 / (1.0 + 0.3275911 * ax)
        val y = 1.0 - (((((1.061405429 * t - 1.453152027) * t) + 1.421413741) * t - 0.284496736) * t + 0.254829592) * t * exp(-ax * ax)
        return 0.5 * (1.0 + sign * y)
    }
}
