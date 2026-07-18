package ir.marghzari.portfolio360.core.math

import ir.marghzari.portfolio360.core.model.ImeQuote
import ir.marghzari.portfolio360.core.model.ImeVerdict

/** Direct port of the IME Live tab's rule-based "smart analyzer" (⑤ تحلیل‌گر هوشمند). */
object ImeSignalAnalyzer {

    fun analyze(q: ImeQuote): ImeVerdict {
        var score = 0
        val reasons = mutableListOf<String>()

        when {
            q.plChangePct >= 4 -> { score += 2; reasons += "📈 صعود قوی" }
            q.plChangePct >= 1 -> { score += 1; reasons += "↗ صعود ملایم" }
            q.plChangePct <= -4 -> { score -= 2; reasons += "📉 ریزش قوی" }
            q.plChangePct <= -1 -> { score -= 1; reasons += "↘ ریزش ملایم" }
        }

        val range = q.pMax - q.pMin
        if (range > 0) {
            val posRng = (q.pl - q.pMin) / (range + 1e-9)
            when {
                posRng > 0.85 -> { score += 1; reasons += "🔝 نزدیک سقف روز" }
                posRng < 0.15 -> { score -= 1; reasons += "⬇ نزدیک کف روز" }
            }
        }

        val qd1 = q.bidQtys.getOrElse(0) { 0.0 }
        val qo1 = q.askQtys.getOrElse(0) { 0.0 }
        when {
            qd1 > 0 && qo1 == 0.0 -> { score += 2; reasons += "💚 صف خرید — بدون عرضه" }
            qo1 > 0 && qd1 == 0.0 -> { score -= 2; reasons += "🔴 صف فروش — بدون تقاضا" }
            qd1 > 0 && qo1 > 0 -> {
                val ratio = qd1 / (qo1 + 1e-9)
                when {
                    ratio > 3 -> { score += 1; reasons += "📗 فشار خرید قوی (%.1fx)".format(ratio) }
                    ratio < 0.33 -> { score -= 1; reasons += "📛 فشار فروش قوی (%.1fx)".format(ratio) }
                }
            }
        }

        if (q.tradeCount > 1000) reasons += "🔥 حجم بالا"

        val label = when {
            score >= 3 -> "🟢 خرید"
            score >= 1 -> "💛 تمایل خرید"
            score <= -3 -> "🔴 فروش"
            score <= -1 -> "🟠 تمایل فروش"
            else -> "⚪ خنثی"
        }
        return ImeVerdict(label, score, reasons)
    }
}
