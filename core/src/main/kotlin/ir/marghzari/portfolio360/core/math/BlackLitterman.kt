package ir.marghzari.portfolio360.core.math

data class BlackLittermanResult(val weights: DoubleArray, val impliedReturns: DoubleArray)

/** Direct port of `black_litterman` from app.py. [views] maps ticker -> expected annual return (e.g. 0.20 = +20%). */
object BlackLitterman {

    fun compute(
        weightsMkt: DoubleArray,
        cov: Array<DoubleArray>,
        meanReturnsDaily: DoubleArray,
        tickers: List<String>,
        views: Map<String, Double>,
        tau: Double = 0.05,
    ): BlackLittermanResult {
        val n = weightsMkt.size
        val tauCov = Array(n) { i -> DoubleArray(n) { j -> tau * cov[i][j] } }
        val pi = LinAlg.matVec(tauCov, weightsMkt)

        val viewAssets = tickers.filter { views.containsKey(it) }
        if (viewAssets.isEmpty()) {
            return BlackLittermanResult(weightsMkt, DoubleArray(n) { meanReturnsDaily[it] * 252.0 })
        }

        val k = viewAssets.size
        val p = Array(k) { DoubleArray(n) }
        val q = DoubleArray(k)
        for (i in viewAssets.indices) {
            val j = tickers.indexOf(viewAssets[i])
            p[i][j] = 1.0
            q[i] = views.getValue(viewAssets[i])
        }

        return try {
            val pt = LinAlg.transpose(p)
            val pTauCovPt = LinAlg.matMul(LinAlg.matMul(p, tauCov), pt)
            val omega = Array(k) { i -> DoubleArray(k) { j -> if (i == j) pTauCovPt[i][i] else 0.0 } }
            val omegaInv = LinAlg.invert(omega)
            val tauCovInv = LinAlg.invert(tauCov)

            val mInv = LinAlg.invert(LinAlg.addMatrices(tauCovInv, LinAlg.matMul(LinAlg.matMul(pt, omegaInv), p)))
            val rhs = LinAlg.add(
                LinAlg.matVec(tauCovInv, pi),
                LinAlg.matVec(LinAlg.matMul(pt, omegaInv), q),
            )
            val muBl = LinAlg.matVec(mInv, rhs)
            val covInv = LinAlg.invert(cov)
            var wBl = LinAlg.matVec(covInv, muBl)
            wBl = DoubleArray(n) { wBl[it].coerceAtLeast(0.0) }
            val s = wBl.sum()
            wBl = if (s > 0) DoubleArray(n) { wBl[it] / s } else weightsMkt
            BlackLittermanResult(wBl, muBl)
        } catch (e: Exception) {
            BlackLittermanResult(weightsMkt, pi)
        }
    }
}
