package ir.marghzari.portfolio360.core.math

import ir.marghzari.portfolio360.core.model.RiskInputs
import kotlin.math.abs
import kotlin.math.max

data class CoveredCallResult(
    val bsPrice: Double, val premium: Double, val delta: Double, val gamma: Double, val theta: Double, val vega: Double,
    val totalPremium: Double, val costBasis: Double,
    val profitBelow: Double, val retBelow: Double,
    val cappedGain: Double, val retCapped: Double,
    val maxLoss: Double, val retMaxLoss: Double,
    val annPremiumYield: Double, val breakeven: Double, val moneyness: Double,
    val ccPeriodRet: Double, val ccAnnRet: Double, val ccAdjRet: Double,
    val expectedAnnAdj: Double, val worthwhileScore: Double, val riskPenaltyPct: Double,
    val ivOk: Boolean, val otmOk: Boolean, val timeOk: Boolean, val shares: Int,
)

data class ProtectivePutResult(
    val bsPrice: Double, val premium: Double, val delta: Double, val gamma: Double, val theta: Double, val vega: Double,
    val totalCost: Double, val costPct: Double, val costAnnPct: Double,
    val maxLossInsured: Double, val breakeven: Double,
    val netRetAfterInsurancePct: Double, val expectedAdjPct: Double, val riskPenaltyPct: Double,
    val worthwhile: Boolean,
)

data class IronCondorResult(
    val netCredit: Double, val totalCredit: Double, val maxLoss: Double,
    val beLower: Double, val beUpper: Double, val profitZonePct: Double,
    val retOnRiskPct: Double, val annRetPct: Double, val adjAnnRetPct: Double,
    val expectedAdjPct: Double, val worthwhileScorePct: Double, val riskPenaltyPct: Double,
    val putBuyPrice: Double, val putSellPrice: Double, val callSellPrice: Double, val callBuyPrice: Double,
)

data class RollingCcCycle(
    val dayIndex: Int, val spot: Double, val strike: Double, val premium: Double,
    val premiumEarned: Double, val spotAtExpiry: Double, val exercised: Boolean,
    val stockPnl: Double, val optionPnl: Double, val totalPnl: Double, val delta: Double,
)

/** Port of the covered-call / protective-put / iron-condor / rolling-CC analytics from app.py. */
object OptionsStrategies {

    fun analyzeCoveredCall(
        S: Double, K: Double, tDays: Int, r: Double, sigma: Double,
        premiumInput: Double, contracts: Int, risk: RiskInputs,
    ): CoveredCallResult {
        val T = tDays / 365.0
        val g = BlackScholes.price(S, K, T, r, sigma, OptionType.CALL)
        val premium = if (premiumInput <= 0) g.price else premiumInput
        val shares = contracts * 100
        val totalPremium = premium * shares
        val costBasis = S * shares

        val profitBelow = totalPremium
        val retBelow = profitBelow / costBasis
        val cappedGain = (K - S) * shares + totalPremium
        val retCapped = cappedGain / costBasis
        val maxLoss = costBasis - totalPremium
        val retMaxLoss = -maxLoss / costBasis
        val annPremiumYield = (premium / S) * (365.0 / tDays)
        val breakeven = S - premium

        val riskPenalty = PortfolioEngine.riskPenalty(risk)
        val expectedAnnAdj = (risk.expectedReturnPct / 100.0) * (1 - riskPenalty)
        val ccPeriodRet = premium / S
        val ccAnnRet = ccPeriodRet * (365.0 / tDays)
        val ccAdjRet = ccAnnRet * (1 - riskPenalty * 0.5)
        val worthwhileScore = ccAdjRet - expectedAnnAdj
        val moneyness = (K - S) / S * 100.0

        return CoveredCallResult(
            bsPrice = g.price, premium = premium, delta = g.delta, gamma = g.gamma, theta = g.theta, vega = g.vega,
            totalPremium = totalPremium, costBasis = costBasis,
            profitBelow = profitBelow, retBelow = retBelow,
            cappedGain = cappedGain, retCapped = retCapped,
            maxLoss = maxLoss, retMaxLoss = retMaxLoss,
            annPremiumYield = annPremiumYield, breakeven = breakeven, moneyness = moneyness,
            ccPeriodRet = ccPeriodRet, ccAnnRet = ccAnnRet, ccAdjRet = ccAdjRet,
            expectedAnnAdj = expectedAnnAdj, worthwhileScore = worthwhileScore, riskPenaltyPct = riskPenalty * 100,
            ivOk = sigma >= 0.20, otmOk = K >= S * 1.02, timeOk = tDays >= 21, shares = shares,
        )
    }

    fun analyzeProtectivePut(
        S: Double, kPut: Double, tDays: Int, r: Double, sigmaPut: Double,
        premiumInput: Double, sharesOwned: Int, risk: RiskInputs,
    ): ProtectivePutResult {
        val T = tDays / 365.0
        val g = BlackScholes.price(S, kPut, T, r, sigmaPut, OptionType.PUT)
        val premium = if (premiumInput <= 0) g.price else premiumInput
        val totalCost = premium * sharesOwned
        val costPct = premium / S
        val maxLossInsured = (S - kPut + premium) * sharesOwned
        val breakeven = S + premium
        val riskPenalty = PortfolioEngine.riskPenalty(risk)
        val expectedAdj = (risk.expectedReturnPct / 100.0) * (1 - riskPenalty)
        val costAnn = costPct * (365.0 / tDays)
        val netRet = expectedAdj - costAnn
        return ProtectivePutResult(
            bsPrice = g.price, premium = premium, delta = g.delta, gamma = g.gamma, theta = g.theta, vega = g.vega,
            totalCost = totalCost, costPct = costPct * 100, costAnnPct = costAnn * 100,
            maxLossInsured = maxLossInsured, breakeven = breakeven,
            netRetAfterInsurancePct = netRet * 100, expectedAdjPct = expectedAdj * 100, riskPenaltyPct = riskPenalty * 100,
            worthwhile = netRet > 0,
        )
    }

    /** Floor return for a hedged asset: `max(r, K/S - 1) - dailyPremiumCost`. */
    fun hedgeFloorReturn(kPut: Double, s: Double): Double = kPut / s - 1.0
    fun hedgeDailyCost(premiumPerShare: Double, s: Double): Double = (premiumPerShare / s) / 365.0

    fun effectiveHedgedVol(rawReturns: DoubleArray, kPut: Double, s: Double, premiumPerShare: Double): Double {
        val floor = hedgeFloorReturn(kPut, s)
        val cost = hedgeDailyCost(premiumPerShare, s)
        val hedged = DoubleArray(rawReturns.size) { max(rawReturns[it], floor) - cost }
        return Stats.std(hedged) * kotlin.math.sqrt(252.0)
    }

    fun applyHedgeToColumn(column: DoubleArray, kPut: Double, s: Double, premiumPerShare: Double): DoubleArray {
        val floor = hedgeFloorReturn(kPut, s)
        val cost = hedgeDailyCost(premiumPerShare, s)
        return DoubleArray(column.size) { max(column[it], floor) - cost }
    }

    fun analyzeIronCondor(
        S: Double, kPutBuy: Double, kPutSell: Double, kCallSell: Double, kCallBuy: Double,
        tDays: Int, r: Double, sigma: Double, contracts: Int, risk: RiskInputs,
    ): IronCondorResult {
        val T = tDays / 365.0
        val pb = BlackScholes.price(S, kPutBuy, T, r, sigma, OptionType.PUT).price
        val ps = BlackScholes.price(S, kPutSell, T, r, sigma, OptionType.PUT).price
        val cs = BlackScholes.price(S, kCallSell, T, r, sigma, OptionType.CALL).price
        val cb = BlackScholes.price(S, kCallBuy, T, r, sigma, OptionType.CALL).price
        val netCredit = ps - pb + cs - cb
        val totalCredit = netCredit * contracts * 100
        val spreadPut = kPutSell - kPutBuy
        val spreadCall = kCallBuy - kCallSell
        val maxLoss = (max(spreadPut, spreadCall) - netCredit) * contracts * 100
        val beLower = kPutSell - netCredit
        val beUpper = kCallSell + netCredit
        val profitZonePct = (beUpper - beLower) / S * 100
        val retOnRisk = netCredit / (max(spreadPut, spreadCall) - netCredit + 1e-9)
        val annRet = retOnRisk * (365.0 / tDays)
        val riskPenalty = PortfolioEngine.riskPenalty(risk)
        val adjAnnRet = annRet * (1 - riskPenalty * 0.3)
        val expectedAdj = (risk.expectedReturnPct / 100.0) * (1 - riskPenalty)
        val worthwhileScore = adjAnnRet - expectedAdj
        return IronCondorResult(
            netCredit = netCredit, totalCredit = totalCredit, maxLoss = maxLoss,
            beLower = beLower, beUpper = beUpper, profitZonePct = profitZonePct,
            retOnRiskPct = retOnRisk * 100, annRetPct = annRet * 100, adjAnnRetPct = adjAnnRet * 100,
            expectedAdjPct = expectedAdj * 100, worthwhileScorePct = worthwhileScore * 100, riskPenaltyPct = riskPenalty * 100,
            putBuyPrice = pb, putSellPrice = ps, callSellPrice = cs, callBuyPrice = cb,
        )
    }

    /** Iron condor P&L at a given expiry spot price, per contract-adjusted dollar terms. */
    fun ironCondorPnlAt(S: Double, kPutBuy: Double, kPutSell: Double, kCallSell: Double, kCallBuy: Double, netCredit: Double, contracts: Int): Double {
        val putSpread = -max(kPutSell - S, 0.0) + max(kPutBuy - S, 0.0)
        val callSpread = -max(S - kCallSell, 0.0) + max(S - kCallBuy, 0.0)
        return (putSpread + callSpread + netCredit) * contracts * 100
    }

    fun simulateRollingCc(
        spotSeries: DoubleArray, kOffsetPct: Double, dte: Int, r: Double, sigma: Double, contracts: Int,
    ): List<RollingCcCycle> {
        val step = max(dte, 5)
        val n = spotSeries.size
        val results = mutableListOf<RollingCcCycle>()
        var i = 0
        while (i < n - step) {
            val S = spotSeries[i]
            val K = S * (1 + kOffsetPct / 100.0)
            val T = dte / 365.0
            val g = BlackScholes.price(S, K, T, r, sigma, OptionType.CALL)
            val premiumEarned = g.price * contracts * 100
            val sExp = spotSeries[minOf(i + step, n - 1)]
            val exercised = sExp >= K
            val stockPnl = (sExp - S) * contracts * 100
            val optionPnl = if (exercised) -(sExp - K) * contracts * 100 + premiumEarned else premiumEarned
            results.add(
                RollingCcCycle(
                    dayIndex = i, spot = S, strike = K, premium = g.price,
                    premiumEarned = premiumEarned, spotAtExpiry = sExp, exercised = exercised,
                    stockPnl = stockPnl, optionPnl = optionPnl, totalPnl = stockPnl + optionPnl, delta = g.delta,
                )
            )
            i += step
        }
        return results
    }

    /** Generic long/short call/put P&L at expiration used by the Iran commodity-exchange options tab. */
    fun pnlAtExpiry(S: Double, K: Double, premiumPaid: Double, lotSize: Int, nContracts: Int, type: OptionType, isLong: Boolean): Double {
        val intrinsic = if (type == OptionType.CALL) max(S - K, 0.0) else max(K - S, 0.0)
        val perUnit = if (isLong) intrinsic - premiumPaid else premiumPaid - intrinsic
        return perUnit * lotSize * nContracts
    }
}
