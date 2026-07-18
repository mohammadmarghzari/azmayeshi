package ir.marghzari.portfolio360.core.network

import kotlinx.datetime.Clock

/**
 * Minimal in-memory cache keyed by an arbitrary string, mirroring the `@st.cache_data(ttl=...)`
 * decorators sprinkled across app.py (Yahoo history: 1h, Fear&Greed: 30m, news: 15m, IME board: 60s,
 * IME history: 5m, world commodity prices: 30m).
 */
class TtlCache<T> {
    private data class Entry<T>(val value: T, val expiresAtMs: Long)
    private val lock = Any()
    private val store = HashMap<String, Entry<T>>()

    suspend fun getOrPut(key: String, ttlMs: Long, compute: suspend () -> T): T {
        val now = Clock.System.now().toEpochMilliseconds()
        val cached = synchronized(lock) { store[key] }
        if (cached != null && cached.expiresAtMs > now) return cached.value
        val value = compute()
        synchronized(lock) { store[key] = Entry(value, now + ttlMs) }
        return value
    }

    fun clear() = synchronized(lock) { store.clear() }
}
