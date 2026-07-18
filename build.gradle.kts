// Intentionally empty: each module declares and applies its own plugins.
// Keeping this file free of android/compose plugin declarations means a plain
// `:core:test` run never needs to resolve the Android Gradle Plugin (which requires
// network access to Google's Maven repository).
