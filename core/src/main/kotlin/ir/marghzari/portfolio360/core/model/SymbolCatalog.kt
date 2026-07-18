package ir.marghzari.portfolio360.core.model

data class SymbolCategory(val labelFa: String, val symbols: List<Pair<String, String>>)

/** Direct port of the `SYMBOLS` dict from app.py: ticker -> display name, grouped by category. */
object SymbolCatalog {
    val CATEGORIES: List<SymbolCategory> = listOf(
        SymbolCategory(
            "💰 ارزهای دیجیتال",
            listOf(
                "BTC-USD" to "Bitcoin", "ETH-USD" to "Ethereum", "BNB-USD" to "BNB", "SOL-USD" to "Solana",
                "XRP-USD" to "XRP", "ADA-USD" to "Cardano", "AVAX-USD" to "Avalanche", "DOGE-USD" to "Dogecoin",
                "DOT-USD" to "Polkadot", "MATIC-USD" to "Polygon", "LINK-USD" to "Chainlink", "LTC-USD" to "Litecoin",
                "UNI7083-USD" to "Uniswap", "ATOM-USD" to "Cosmos", "XLM-USD" to "Stellar",
            ),
        ),
        SymbolCategory(
            "📈 سهام آمریکا",
            listOf(
                "AAPL" to "Apple", "MSFT" to "Microsoft", "GOOGL" to "Alphabet", "AMZN" to "Amazon",
                "NVDA" to "NVIDIA", "META" to "Meta", "TSLA" to "Tesla", "BERKB" to "Berkshire B",
                "JPM" to "JPMorgan", "V" to "Visa", "JNJ" to "J&J", "WMT" to "Walmart", "XOM" to "Exxon",
                "BAC" to "Bank of America", "MA" to "Mastercard", "PG" to "P&G", "HD" to "Home Depot",
                "CVX" to "Chevron", "ABBV" to "AbbVie", "KO" to "Coca-Cola", "PEP" to "PepsiCo",
                "LLY" to "Eli Lilly", "MRK" to "Merck", "CRM" to "Salesforce", "AMD" to "AMD",
                "INTC" to "Intel", "NFLX" to "Netflix", "DIS" to "Disney", "PYPL" to "PayPal", "UBER" to "Uber",
            ),
        ),
        SymbolCategory(
            "🌍 سهام جهانی",
            listOf(
                "TSM" to "TSMC (Taiwan)", "ASML" to "ASML (Netherlands)", "SAP" to "SAP (Germany)",
                "TM" to "Toyota (Japan)", "NVO" to "Novo Nordisk (Denmark)", "HSBC" to "HSBC (UK)",
                "BP" to "BP (UK)", "SHEL" to "Shell (UK)", "UL" to "Unilever (UK)", "RIO" to "Rio Tinto (UK)",
                "BABA" to "Alibaba (China)", "JD" to "JD.com (China)", "SONY" to "Sony (Japan)",
                "HMC" to "Honda (Japan)", "BCS" to "Barclays (UK)",
            ),
        ),
        SymbolCategory(
            "🏦 ETF و شاخص",
            listOf(
                "SPY" to "S&P 500 ETF", "QQQ" to "Nasdaq 100 ETF", "DIA" to "Dow Jones ETF",
                "IWM" to "Russell 2000 ETF", "VTI" to "Total Market ETF", "EEM" to "Emerging Markets ETF",
                "VEA" to "Developed Markets ETF", "AGG" to "Bond Aggregate ETF", "TLT" to "20Y Treasury ETF",
                "HYG" to "High Yield Bond ETF", "GLD" to "Gold ETF", "SLV" to "Silver ETF", "USO" to "Oil ETF",
                "XLE" to "Energy ETF", "XLF" to "Financials ETF", "XLK" to "Technology ETF",
                "XLV" to "Healthcare ETF", "ARKK" to "ARK Innovation ETF", "VNQ" to "Real Estate ETF",
                "PDBC" to "Commodity ETF",
            ),
        ),
        SymbolCategory(
            "🥇 کامودیتی و فارکس",
            listOf(
                "GC=F" to "Gold Futures", "SI=F" to "Silver Futures", "CL=F" to "Crude Oil (WTI)",
                "BZ=F" to "Brent Oil", "NG=F" to "Natural Gas", "HG=F" to "Copper", "PL=F" to "Platinum",
                "ZW=F" to "Wheat", "ZC=F" to "Corn", "ZS=F" to "Soybeans", "EURUSD=X" to "EUR/USD",
                "GBPUSD=X" to "GBP/USD", "USDJPY=X" to "USD/JPY", "USDCHF=X" to "USD/CHF",
                "AUDUSD=X" to "AUD/USD", "USDCAD=X" to "USD/CAD", "USDIRR=X" to "USD/IRR",
            ),
        ),
    )

    fun displayName(ticker: String): String = CATEGORIES.flatMap { it.symbols }.firstOrNull { it.first == ticker }?.second ?: ticker

    val ALL_TICKERS: List<String> get() = CATEGORIES.flatMap { cat -> cat.symbols.map { it.first } }
}
