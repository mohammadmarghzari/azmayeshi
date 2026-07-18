package ir.marghzari.portfolio360.core.math

import kotlin.math.abs
import kotlin.math.exp
import kotlin.math.ln
import kotlin.math.max
import kotlin.math.sqrt

enum class OptionType { CALL, PUT }

data class OptionGreeks(
    val price: Double,
    val delta: Double,
    val gamma: Double,
    /** Per-day theta, already divided by 365 (matches the Python `theta` return value). */
    val theta: Double,
    /** Per-1%-vol-point vega, already divided by 100 (matches the Python `vega` return value). */
    val vega: Double,
)

/** Direct port of `bs_price` / `black_scholes_call` from app.py. */
object BlackScholes {

    fun price(S: Double, K: Double, T: Double, r: Double, sigma: Double, type: OptionType = OptionType.CALL): OptionGreeks {
        val isCall = type == OptionType.CALL
        if (T <= 0 || sigma <= 0 || S <= 0 || K <= 0) {
            val intrinsic = if (isCall) max(S - K, 0.0) else max(K - S, 0.0)
            val delta = if (isCall) (if (S > K) 1.0 else 0.0) else (if (S < K) -1.0 else 0.0)
            return OptionGreeks(intrinsic, delta, 0.0, 0.0, 0.0)
        }
        val d1 = (ln(S / K) + (r + 0.5 * sigma * sigma) * T) / (sigma * sqrt(T))
        val d2 = d1 - sigma * sqrt(T)
        val n = NormalDistribution

        val price: Double
        val delta: Double
        if (isCall) {
            price = S * n.cdf(d1) - K * exp(-r * T) * n.cdf(d2)
            delta = n.cdf(d1)
        } else {
            price = K * exp(-r * T) * n.cdf(-d2) - S * n.cdf(-d1)
            delta = n.cdf(d1) - 1.0
        }
        val gamma = n.pdf(d1) / (S * sigma * sqrt(T) + 1e-9)
        val theta = (
            -(S * n.pdf(d1) * sigma) / (2 * sqrt(T + 1e-9)) -
                r * K * exp(-r * T) * (if (isCall) n.cdf(d2) else n.cdf(-d2))
            ) / 365.0
        val vega = S * n.pdf(d1) * sqrt(T) / 100.0
        return OptionGreeks(price, delta, gamma, theta, vega)
    }

    /** Newton-Raphson implied volatility solver, matching `bc_implied_vol`. Returns null if it can't converge. */
    fun impliedVolatility(
        marketPrice: Double,
        S: Double, K: Double, T: Double, r: Double,
        type: OptionType = OptionType.CALL,
        tol: Double = 1e-5,
        maxIter: Int = 200,
    ): Double? {
        if (marketPrice <= 0 || T <= 0) return null
        val intrinsic = if (type == OptionType.CALL) max(S - K, 0.0) else max(K - S, 0.0)
        if (marketPrice < intrinsic) return null
        var sigma = 0.30
        repeat(maxIter) {
            val g = price(S, K, T, r, sigma, type)
            val diff = g.price - marketPrice
            if (abs(diff) < tol) return sigma
            val vegaFull = g.vega * 100.0
            if (abs(vegaFull) < 1e-10) return@repeat
            sigma -= diff / (vegaFull + 1e-9)
            sigma = sigma.coerceIn(0.001, 10.0)
        }
        return sigma
    }
}
