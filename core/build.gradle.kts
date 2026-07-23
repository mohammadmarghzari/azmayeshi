import java.util.Properties

plugins {
    alias(libs.plugins.kotlinJvm)
    alias(libs.plugins.kotlinSerialization)
}

// Secrets are NEVER hardcoded or committed: a gitignored `secrets.properties` at the repo root
// (for local/dev builds) or the `COINGLASS_API_KEY` environment variable (for CI, sourced from a
// GitHub Actions secret) supplies the real value; anything else compiles with an empty key, which
// CoinglassClient treats as "not configured" rather than crashing.
val secretsFile = rootProject.file("secrets.properties")
val secretsProps = Properties().apply {
    if (secretsFile.exists()) secretsFile.inputStream().use { load(it) }
}
val coinglassApiKey: String =
    (secretsProps.getProperty("coinglassApiKey") ?: System.getenv("COINGLASS_API_KEY") ?: "")

val generatedSecretsDir = layout.buildDirectory.dir("generated/secrets/kotlin")

val generateSecrets = tasks.register("generateSecrets") {
    val outputDir = generatedSecretsDir
    inputs.property("coinglassApiKey", coinglassApiKey)
    outputs.dir(outputDir)
    doLast {
        val escaped = coinglassApiKey.replace("\\", "\\\\").replace("\"", "\\\"")
        val dir = outputDir.get().asFile.resolve("ir/marghzari/portfolio360/core/network")
        dir.mkdirs()
        dir.resolve("BuildSecrets.kt").writeText(
            """
            package ir.marghzari.portfolio360.core.network

            // GENERATED at build time by :core's generateSecrets task — do not edit, do not commit real values here.
            internal object BuildSecrets {
                const val COINGLASS_API_KEY: String = "$escaped"
            }
            """.trimIndent() + "\n",
        )
    }
}

kotlin {
    jvmToolchain(21)
    sourceSets["main"].kotlin.srcDir(generateSecrets.map { generatedSecretsDir.get() })
}

dependencies {
    implementation(libs.kotlinx.coroutines.core)
    implementation(libs.kotlinx.serialization.json)
    implementation(libs.kotlinx.datetime)

    implementation(libs.ktor.client.core)
    implementation(libs.ktor.client.cio)
    implementation(libs.ktor.client.content.negotiation)
    implementation(libs.ktor.serialization.kotlinx.json)
    implementation(libs.ktor.client.logging)

    testImplementation(libs.kotlin.test)
    testImplementation(libs.kotlin.test.junit)
}

tasks.test {
    useJUnit()
}
