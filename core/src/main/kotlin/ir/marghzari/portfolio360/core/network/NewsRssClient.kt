package ir.marghzari.portfolio360.core.network

import io.ktor.client.call.body
import io.ktor.client.request.get
import io.ktor.client.request.header
import org.w3c.dom.Element
import java.io.ByteArrayInputStream
import javax.xml.parsers.DocumentBuilderFactory

data class NewsItem(val title: String, val link: String, val pubDate: String, val description: String)

/** Yahoo Finance RSS headlines per ticker, matching `fetch_news_rss` in app.py (15-minute cache). */
class NewsRssClient(private val cache: TtlCache<List<NewsItem>> = TtlCache()) {

    suspend fun fetch(ticker: String, limit: Int = 7): List<NewsItem> = cache.getOrPut("news|$ticker", 900_000) {
        try {
            val xml: String = HttpClientProvider.client.get("https://feeds.finance.yahoo.com/rss/2.0/headline") {
                url {
                    parameters.append("s", ticker)
                    parameters.append("region", "US")
                    parameters.append("lang", "en-US")
                }
                header("Accept", "application/rss+xml, text/xml")
            }.body()

            val factory = DocumentBuilderFactory.newInstance().apply {
                isNamespaceAware = false
                setFeature("http://apache.org/xml/features/disallow-doctype-decl", true)
            }
            val doc = factory.newDocumentBuilder().parse(ByteArrayInputStream(xml.toByteArray(Charsets.UTF_8)))
            val items = doc.getElementsByTagName("item")
            val out = mutableListOf<NewsItem>()
            for (i in 0 until minOf(items.length, limit)) {
                val el = items.item(i) as Element
                fun text(tag: String) = el.getElementsByTagName(tag).item(0)?.textContent?.trim().orEmpty()
                out.add(NewsItem(text("title"), text("link"), text("pubDate"), text("description")))
            }
            out
        } catch (e: Exception) {
            emptyList()
        }
    }
}
