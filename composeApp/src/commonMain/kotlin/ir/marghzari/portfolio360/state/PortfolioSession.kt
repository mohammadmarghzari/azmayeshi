package ir.marghzari.portfolio360.state

import androidx.compose.runtime.getValue
import androidx.compose.runtime.mutableStateOf
import androidx.compose.runtime.setValue
import ir.marghzari.portfolio360.core.model.PortfolioMetrics
import ir.marghzari.portfolio360.core.model.PriceSeries
import ir.marghzari.portfolio360.core.model.RiskInputs

/**
 * Everything produced by the fetch → optimize pipeline, extracted from the former AppState
 * god-object into its own feature state-holder. Screens still read these through `AppState`'s
 * delegating properties, so this refactor changed no call sites; new code can depend on this
 * narrower type directly instead of all of AppState.
 */
class PortfolioSession {
    var prices by mutableStateOf<PriceSeries?>(null)
    var fetchFailedTickers by mutableStateOf<List<String>>(emptyList())
    var weights by mutableStateOf<DoubleArray?>(null)
    var covariance by mutableStateOf<Array<DoubleArray>?>(null)
    var metrics by mutableStateOf<PortfolioMetrics?>(null)
    var styleLabelUsed by mutableStateOf("")
    var lastUsedRisk by mutableStateOf(RiskInputs())

    var isFetching by mutableStateOf(false)
    var isCalculating by mutableStateOf(false)
    var lastError by mutableStateOf<String?>(null)

    fun resetComputed() {
        weights = null; covariance = null; metrics = null
    }
}
