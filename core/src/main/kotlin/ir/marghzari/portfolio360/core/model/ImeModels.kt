package ir.marghzari.portfolio360.core.model

/** One row of the IME (Iran Mercantile Exchange) commodity-deposit-certificate live board. */
data class ImeQuote(
    val commodity: String,
    val contractCode: String,
    val contractDescription: String,
    val contractSize: Double,
    val contractSizeUnit: String,
    val contractCurrency: String,
    val py: Double, // previous close
    val pf: Double, val pfChange: Double, val pfChangePct: Double, // first price / change / change%
    val pMax: Double, val pMaxChange: Double, val pMaxChangePct: Double,
    val pMin: Double, val pMinChange: Double, val pMinChangePct: Double,
    val pl: Double, val plChange: Double, val plChangePct: Double, // last price / change / change%
    val plTime: String, val pfTime: String,
    val tradeCount: Long, val volume: Double, val tradeValue: Double, val tradeValueUnit: String,
    val dateOrder: String, val timeOrder: String,
    val bidPrices: List<Double>, val bidQtys: List<Double>,
    val askPrices: List<Double>, val askQtys: List<Double>,
    val dateUpdate: String, val timeUpdate: String,
)

data class ImeCandle(val dateLabel: String, val open: Double, val high: Double, val low: Double, val close: Double, val volume: Double)

data class ImeSignal(val labelFa: String, val score: Int)
data class ImeVerdict(val labelFa: String, val score: Int, val reasonsFa: List<String>)
