package ir.marghzari.portfolio360.core.math

/** Fair-value dollar & commodity-certificate bubble calculators from the "ابزار ایران" tab. */
object IranTools {

    data class InflationFairValue(val inflationDiffPct: Double, val fairDollarToman: Double, val gapToman: Double, val gapPct: Double)

    fun fairValueByInflation(baseDollarToman: Double, iranInflationPct: Double, usInflationPct: Double, marketDollarToman: Double): InflationFairValue {
        val diff = iranInflationPct - usInflationPct
        val fair = baseDollarToman * (1 + diff / 100.0)
        val gap = fair - marketDollarToman
        return InflationFairValue(diff, fair, gap, gap / marketDollarToman * 100.0)
    }

    data class GoldFairValue(val fairDollarToman: Double, val gapToman: Double, val gapPct: Double)

    /** Bahar Azadi coin price ÷ (world ounce price × 4) = fair dollar price. */
    fun fairValueByGold(sekkePriceToman: Double, goldOzUsd: Double, marketDollarToman: Double): GoldFairValue {
        val fair = sekkePriceToman / (goldOzUsd * 4.0)
        val gap = fair - marketDollarToman
        return GoldFairValue(fair, gap, gap / marketDollarToman * 100.0)
    }

    const val TROY_OZ_GRAMS = 31.1035
    const val SEKKE_WEIGHT_GRAMS = 8.133

    enum class CommodityKey { GoldBar, GoldCoin, SilverBar, CopperCthd, ZincIngot, SteelRebar, Bitumen }

    data class CommodityDef(val key: CommodityKey, val labelFa: String, val unitFa: String, val defaultPriceToman: Long, val categoryFa: String, val icon: String)

    val IRAN_COMMODITIES = listOf(
        CommodityDef(CommodityKey.GoldBar, "شمش طلا 995+", "گرم", 20_964_990, "فلزات گرانبها", "🥇"),
        CommodityDef(CommodityKey.GoldCoin, "سکه بهار آزادی", "سکه", 1_612_393_490, "فلزات گرانبها", "🪙"),
        CommodityDef(CommodityKey.SilverBar, "شمش نقره 999.9", "گرم", 3_748_900, "فلزات گرانبها", "🥈"),
        CommodityDef(CommodityKey.CopperCthd, "مس کاتد", "کیلوگرم", 22_200_000, "فلزات صنعتی", "🟠"),
        CommodityDef(CommodityKey.ZincIngot, "شمش روی", "کیلوگرم", 4_790_100, "فلزات صنعتی", "🔘"),
        CommodityDef(CommodityKey.SteelRebar, "میلگرد", "کیلوگرم", 580_000, "فلزات صنعتی", "⚙"),
        CommodityDef(CommodityKey.Bitumen, "قیر", "کیلوگرم", 751_150, "نفت و گاز", "🛢"),
    )

    /** World reference prices, in USD, needed for [fairPrice]. Copper/zinc must already be converted to $/kg. */
    data class WorldPrices(val goldOzUsd: Double, val silverOzUsd: Double, val copperUsdPerKg: Double, val zincUsdPerKg: Double)

    /** null when there is no reliable world reference (SteelRebar, Bitumen). */
    fun fairPrice(key: CommodityKey, world: WorldPrices, usdTomanRate: Double): Double? = when (key) {
        CommodityKey.GoldBar -> (world.goldOzUsd * usdTomanRate) / TROY_OZ_GRAMS
        CommodityKey.GoldCoin -> (world.goldOzUsd * usdTomanRate / TROY_OZ_GRAMS) * SEKKE_WEIGHT_GRAMS
        CommodityKey.SilverBar -> (world.silverOzUsd * usdTomanRate) / TROY_OZ_GRAMS
        CommodityKey.CopperCthd -> world.copperUsdPerKg * usdTomanRate
        CommodityKey.ZincIngot -> world.zincUsdPerKg * usdTomanRate
        CommodityKey.SteelRebar, CommodityKey.Bitumen -> null
    }

    data class BubbleResult(val marketPriceToman: Double, val fairPriceToman: Double?, val bubbleToman: Double?, val bubblePct: Double?)

    fun bubble(marketPriceToman: Double, fairPriceToman: Double?): BubbleResult {
        if (fairPriceToman == null || fairPriceToman <= 0) return BubbleResult(marketPriceToman, null, null, null)
        val bubble = marketPriceToman - fairPriceToman
        return BubbleResult(marketPriceToman, fairPriceToman, bubble, bubble / fairPriceToman * 100.0)
    }

    data class PnlTargetResult(
        val costToman: Double, val pnlCurrentToman: Double, val pnlCurrentPct: Double,
        val pnlTargetToman: Double, val pnlTargetPct: Double,
        val pnlStopToman: Double, val pnlStopPct: Double, val riskRewardRatio: Double,
    )

    fun pnlTarget(buyPriceToman: Double, quantity: Double, currentPriceToman: Double, targetPriceToman: Double, stopPriceToman: Double): PnlTargetResult {
        val cost = buyPriceToman * quantity
        val pnlCur = (currentPriceToman - buyPriceToman) * quantity
        val pnlTarget = (targetPriceToman - buyPriceToman) * quantity
        val pnlStop = (stopPriceToman - buyPriceToman) * quantity
        val rr = if (pnlStop != 0.0) kotlin.math.abs(pnlTarget / pnlStop) else Double.POSITIVE_INFINITY
        return PnlTargetResult(
            cost, pnlCur, pnlCur / cost * 100.0,
            pnlTarget, pnlTarget / cost * 100.0,
            pnlStop, pnlStop / cost * 100.0, rr,
        )
    }

    fun formatToman(v: Double): String = "${"%,.0f".format(v)} تومان"
}
