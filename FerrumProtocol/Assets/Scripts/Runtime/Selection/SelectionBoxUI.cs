using UnityEngine;
using UnityEngine.UI;

namespace FerrumProtocol.Selection
{
    /// <summary>Draws the drag-select rectangle overlay while <see cref="SelectionManager.IsBoxSelecting"/> is true.</summary>
    [RequireComponent(typeof(RectTransform))]
    public class SelectionBoxUI : MonoBehaviour
    {
        [SerializeField] private SelectionManager selectionManager;
        [SerializeField] private Image image;
        [SerializeField] private Canvas parentCanvas;

        private RectTransform _rect;

        private void Awake()
        {
            _rect = GetComponent<RectTransform>();
            _rect.anchorMin = Vector2.zero;
            _rect.anchorMax = Vector2.zero;
            _rect.pivot = Vector2.zero;

            if (image == null)
            {
                image = GetComponent<Image>();
            }

            SetVisible(false);
        }

        private void Update()
        {
            if (selectionManager == null)
            {
                return;
            }

            if (!selectionManager.IsBoxSelecting)
            {
                SetVisible(false);
                return;
            }

            SetVisible(true);
            Vector2 mousePos = UnityEngine.InputSystem.Mouse.current != null
                ? UnityEngine.InputSystem.Mouse.current.position.ReadValue()
                : (Vector2)Input.mousePosition;

            Rect box = selectionManager.GetCurrentBoxRect(mousePos);
            // GetCurrentBoxRect is in GUI-space (Y flipped); convert back to canvas (bottom-up) space for anchoredPosition.
            float yBottomUp = Screen.height - box.y - box.height;
            _rect.anchoredPosition = new Vector2(box.x, yBottomUp);
            _rect.sizeDelta = new Vector2(box.width, box.height);
        }

        private void SetVisible(bool visible)
        {
            if (image != null)
            {
                image.enabled = visible;
            }
        }
    }
}
