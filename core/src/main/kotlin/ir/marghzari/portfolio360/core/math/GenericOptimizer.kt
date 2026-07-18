package ir.marghzari.portfolio360.core.math

/**
 * Generic derivative-free portfolio optimizer used by every allocation style.
 *
 * The source Python app relies on SciPy's SLSQP for each style's own objective/constraint
 * combination. Rather than porting a full general-purpose SQP solver, every style here is
 * expressed as a plain objective(w) -> Double (constraints folded in as penalty terms where
 * the Python code used inequality constraints), and solved with projected gradient descent:
 * a numeric (central-difference) gradient step followed by Euclidean projection onto the
 * capped simplex { sum(w) = 1, lo <= w_i <= hi }. This is a standard, robust approach for the
 * small (n <= ~30 asset), well-conditioned convex/quasi-convex problems that show up here.
 */
object GenericOptimizer {

    fun projectedGradientDescent(
        n: Int,
        lo: Double,
        hi: Double,
        iterations: Int = 300,
        initialLr: Double = 0.08,
        init: DoubleArray? = null,
        objective: (DoubleArray) -> Double,
    ): DoubleArray {
        var w = init?.let { LinAlg.projectCappedSimplex(it, lo, hi) }
            ?: LinAlg.projectCappedSimplex(DoubleArray(n) { 1.0 / n }, lo, hi)
        var lr = initialLr
        var bestW = w.copyOf()
        var bestVal = objective(w)
        val h = 1e-5
        for (iter in 0 until iterations) {
            val grad = DoubleArray(n)
            for (i in 0 until n) {
                val wPlus = w.copyOf(); wPlus[i] += h
                val wMinus = w.copyOf(); wMinus[i] -= h
                grad[i] = (objective(wPlus) - objective(wMinus)) / (2 * h)
            }
            val gNorm = LinAlg.norm2(grad)
            val step = if (gNorm > 1e-9) LinAlg.scale(grad, lr / gNorm) else grad
            val newW = LinAlg.projectCappedSimplex(LinAlg.sub(w, step), lo, hi)
            val newVal = objective(newW)
            if (newVal < bestVal) {
                bestVal = newVal
                bestW = newW.copyOf()
            }
            w = newW
            lr *= 0.99
        }
        return bestW
    }

    /** Soft penalty for the "expected annual return >= target" inequality constraint used across styles. */
    fun returnFloorPenalty(w: DoubleArray, meanReturns: DoubleArray, targetAnnualReturn: Double, weight: Double = 40.0): Double {
        if (targetAnnualReturn <= 0.001) return 0.0
        val achieved = LinAlg.dot(w, meanReturns)
        val shortfall = (targetAnnualReturn - achieved).coerceAtLeast(0.0)
        return weight * shortfall * shortfall
    }
}
