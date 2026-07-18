package ir.marghzari.portfolio360.core.network

import ir.marghzari.portfolio360.core.math.IranTools

/** Wraps Yahoo futures quotes into the world reference prices needed by the Iran commodity bubble calculator. */
class WorldCommodityClient(private val yahoo: YahooFinanceClient = YahooFinanceClient()) {

    suspend fun fetch(): IranTools.WorldPrices? {
        val goldOz = yahoo.fetchLastClose("GC=F") ?: return null
        val silverOz = yahoo.fetchLastClose("SI=F") ?: return null
        val copperLb = yahoo.fetchLastClose("HG=F") ?: return null
        val zincTon = yahoo.fetchLastClose("ZN=F") ?: return null
        return IranTools.WorldPrices(
            goldOzUsd = goldOz,
            silverOzUsd = silverOz,
            copperUsdPerKg = copperLb * 2.20462,
            zincUsdPerKg = zincTon / 1000.0,
        )
    }

    suspend fun fetchGoldOz(): Double? = yahoo.fetchLastClose("GC=F")
}
