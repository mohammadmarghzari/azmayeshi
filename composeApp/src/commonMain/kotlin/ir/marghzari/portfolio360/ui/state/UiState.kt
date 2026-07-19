package ir.marghzari.portfolio360.ui.state

import androidx.compose.runtime.Composable
import androidx.compose.ui.Modifier
import ir.marghzari.portfolio360.ui.components.EmptyState
import ir.marghzari.portfolio360.ui.components.ErrorState
import ir.marghzari.portfolio360.ui.motion.SkeletonCard

/**
 * The four states every data-driven screen must handle. Screens hold a `UiState<T>` (usually in
 * a `mutableStateOf`) and render it through [StateHost], which guarantees the app never shows a
 * blank area while loading, a silent failure, or an unexplained empty screen.
 */
sealed interface UiState<out T> {
    data object Loading : UiState<Nothing>
    data object Empty : UiState<Nothing>
    data class Error(val message: String) : UiState<Nothing>
    data class Success<T>(val data: T) : UiState<T>
}

/** Convenience: wrap a fetch result — null/empty payloads become [UiState.Empty]. */
inline fun <T> uiStateOf(data: T?, isEmpty: (T) -> Boolean = { false }): UiState<T> = when {
    data == null -> UiState.Empty
    isEmpty(data) -> UiState.Empty
    else -> UiState.Success(data)
}

/**
 * Uniform renderer for [UiState]: shimmer skeleton while loading, [ErrorState] with retry on
 * failure, [EmptyState] with guidance when there is nothing to show, and [content] on success.
 */
@Composable
fun <T> StateHost(
    state: UiState<T>,
    modifier: Modifier = Modifier,
    onRetry: (() -> Unit)? = null,
    emptyTitle: String = "داده‌ای برای نمایش نیست",
    emptyHint: String? = null,
    emptyActionText: String? = null,
    onEmptyAction: (() -> Unit)? = null,
    skeleton: @Composable () -> Unit = { SkeletonCard() },
    content: @Composable (T) -> Unit,
) {
    when (state) {
        is UiState.Loading -> skeleton()
        is UiState.Error -> ErrorState(message = state.message, modifier = modifier, onRetry = onRetry)
        is UiState.Empty -> EmptyState(
            title = emptyTitle, modifier = modifier, hint = emptyHint,
            actionText = emptyActionText, onAction = onEmptyAction,
        )
        is UiState.Success -> content(state.data)
    }
}
