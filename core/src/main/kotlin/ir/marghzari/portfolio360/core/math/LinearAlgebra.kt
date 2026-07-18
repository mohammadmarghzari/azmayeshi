package ir.marghzari.portfolio360.core.math

import kotlin.math.sqrt

/** Minimal dense linear-algebra helpers sized for small (n <= ~30 asset) portfolio problems. */
object LinAlg {

    fun matVec(m: Array<DoubleArray>, v: DoubleArray): DoubleArray {
        val n = m.size
        val out = DoubleArray(n)
        for (i in 0 until n) {
            var s = 0.0
            val row = m[i]
            for (j in row.indices) s += row[j] * v[j]
            out[i] = s
        }
        return out
    }

    fun quadForm(v: DoubleArray, m: Array<DoubleArray>): Double {
        val mv = matVec(m, v)
        var s = 0.0
        for (i in v.indices) s += v[i] * mv[i]
        return s
    }

    fun dot(a: DoubleArray, b: DoubleArray): Double {
        var s = 0.0
        for (i in a.indices) s += a[i] * b[i]
        return s
    }

    fun scale(v: DoubleArray, k: Double): DoubleArray = DoubleArray(v.size) { v[it] * k }

    fun add(a: DoubleArray, b: DoubleArray): DoubleArray = DoubleArray(a.size) { a[it] + b[it] }

    fun sub(a: DoubleArray, b: DoubleArray): DoubleArray = DoubleArray(a.size) { a[it] - b[it] }

    fun addMatrices(a: Array<DoubleArray>, b: Array<DoubleArray>): Array<DoubleArray> =
        Array(a.size) { i -> DoubleArray(a[i].size) { j -> a[i][j] + b[i][j] } }

    fun transpose(m: Array<DoubleArray>): Array<DoubleArray> {
        val rows = m.size
        val cols = if (rows == 0) 0 else m[0].size
        return Array(cols) { j -> DoubleArray(rows) { i -> m[i][j] } }
    }

    fun matMul(a: Array<DoubleArray>, b: Array<DoubleArray>): Array<DoubleArray> {
        val n = a.size
        val k = if (n == 0) 0 else a[0].size
        val m = if (b.isNotEmpty()) b[0].size else 0
        val out = Array(n) { DoubleArray(m) }
        for (i in 0 until n) {
            for (p in 0 until k) {
                val aip = a[i][p]
                if (aip == 0.0) continue
                val row = b[p]
                for (j in 0 until m) out[i][j] += aip * row[j]
            }
        }
        return out
    }

    /** Gauss-Jordan inversion. Throws if singular. Fine for the small (<=30x30) matrices used here. */
    fun invert(matrix: Array<DoubleArray>): Array<DoubleArray> {
        val n = matrix.size
        val a = Array(n) { i -> DoubleArray(2 * n) { j -> if (j < n) matrix[i][j] else if (j - n == i) 1.0 else 0.0 } }
        for (col in 0 until n) {
            var pivotRow = col
            var maxAbs = kotlin.math.abs(a[col][col])
            for (r in col + 1 until n) {
                val v = kotlin.math.abs(a[r][col])
                if (v > maxAbs) { maxAbs = v; pivotRow = r }
            }
            if (maxAbs < 1e-12) {
                // Regularize a hair to avoid a hard failure on near-singular covariance matrices.
                a[col][col] += 1e-8
                maxAbs = kotlin.math.abs(a[col][col])
            }
            if (pivotRow != col) {
                val tmp = a[col]; a[col] = a[pivotRow]; a[pivotRow] = tmp
            }
            val pivot = a[col][col]
            for (j in 0 until 2 * n) a[col][j] /= pivot
            for (r in 0 until n) {
                if (r == col) continue
                val factor = a[r][col]
                if (factor == 0.0) continue
                for (j in 0 until 2 * n) a[r][j] -= factor * a[col][j]
            }
        }
        return Array(n) { i -> DoubleArray(n) { j -> a[i][j + n] } }
    }

    fun identity(n: Int): Array<DoubleArray> = Array(n) { i -> DoubleArray(n) { j -> if (i == j) 1.0 else 0.0 } }

    fun norm2(v: DoubleArray): Double = sqrt(v.sumOf { it * it })

    /**
     * Euclidean projection of [v] onto { w : sum(w) = 1, lo <= w_i <= hi }.
     * Standard bisection-on-the-multiplier algorithm for a capped simplex projection.
     */
    fun projectCappedSimplex(v: DoubleArray, lo: Double, hi: Double): DoubleArray {
        val n = v.size
        fun clipSum(tau: Double): Double {
            var s = 0.0
            for (x in v) s += (x - tau).coerceIn(lo, hi)
            return s
        }
        var lowTau = v.min() - hi
        var highTau = v.max() - lo
        // Bisection: sum(clip(v - tau)) is monotonically decreasing in tau.
        repeat(100) {
            val mid = (lowTau + highTau) / 2.0
            val s = clipSum(mid)
            if (s > 1.0) lowTau = mid else highTau = mid
        }
        val tau = (lowTau + highTau) / 2.0
        return DoubleArray(n) { (v[it] - tau).coerceIn(lo, hi) }
    }
}
