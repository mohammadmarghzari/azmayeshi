rootProject.name = "Portfolio360"

pluginManagement {
    repositories {
        google()
        gradlePluginPortal()
        mavenCentral()
    }
}

dependencyResolutionManagement {
    repositories {
        google()
        mavenCentral()
    }
}

include(":core")
include(":composeApp")

// Local-only development helper: compiles the composeApp module's shared UI as a plain JVM app so
// it can be built/run without the Android Gradle Plugin. Not part of the shipped project; skipped
// automatically when the directory isn't present (e.g. after a fresh clone).
if (file("devpreview/build.gradle.kts").exists()) {
    include(":devpreview")
}
