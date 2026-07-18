package ir.marghzari.portfolio360.core.math

import kotlin.test.Test
import kotlin.test.assertEquals
import kotlin.test.assertTrue

class BlackScholesTest {

    @Test
    fun `call price matches known textbook value`() {
        // S=100, K=100, T=1, r=5%, sigma=20% -> call ~= 10.4506, delta ~= 0.6368 (widely-cited reference values).
        val g = BlackScholes.price(100.0, 100.0, 1.0, 0.05, 0.20, OptionType.CALL)
        assertEquals(10.4506, g.price, 0.01)
        assertEquals(0.6368, g.delta, 0.001)
    }

    @Test
    fun `put price matches known textbook value`() {
        val g = BlackScholes.price(100.0, 100.0, 1.0, 0.05, 0.20, OptionType.PUT)
        assertEquals(5.5735, g.price, 0.01)
    }

    @Test
    fun `put-call parity holds`() {
        val call = BlackScholes.price(120.0, 110.0, 0.5, 0.03, 0.35, OptionType.CALL).price
        val put = BlackScholes.price(120.0, 110.0, 0.5, 0.03, 0.35, OptionType.PUT).price
        val lhs = call - put
        val rhs = 120.0 - 110.0 * kotlin.math.exp(-0.03 * 0.5)
        assertEquals(rhs, lhs, 0.01)
    }

    @Test
    fun `implied volatility round-trips`() {
        val sigma = 0.28
        val price = BlackScholes.price(100.0, 105.0, 0.5, 0.05, sigma, OptionType.CALL).price
        val iv = BlackScholes.impliedVolatility(price, 100.0, 105.0, 0.5, 0.05, OptionType.CALL)
        assertTrue(iv != null)
        assertEquals(sigma, iv!!, 0.001)
    }

    @Test
    fun `zero time to expiry returns intrinsic value`() {
        val g = BlackScholes.price(110.0, 100.0, 0.0, 0.05, 0.2, OptionType.CALL)
        assertEquals(10.0, g.price, 1e-9)
    }
}
