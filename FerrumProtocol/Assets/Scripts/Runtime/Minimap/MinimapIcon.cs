using FerrumProtocol.Selection;
using UnityEngine;
using UnityEngine.UI;

namespace FerrumProtocol.Minimap
{
    /// <summary>Attach to any unit/building that should show up as a dot on the minimap.</summary>
    public class MinimapIcon : MonoBehaviour
    {
        [SerializeField] private Sprite iconSprite;
        [SerializeField] private Vector2 iconSize = new Vector2(6f, 6f);
        [SerializeField] private Selectable selectable;
        [SerializeField] private Color enemyColor = Color.red;
        [SerializeField] private Color allyColor = Color.green;

        private RectTransform _dotRect;
        private Image _dotImage;

        private void Start()
        {
            if (MinimapController.Instance == null || MinimapController.Instance.IconsContainer == null)
            {
                enabled = false;
                return;
            }

            var go = new GameObject($"MinimapDot_{name}", typeof(RectTransform), typeof(Image));
            go.transform.SetParent(MinimapController.Instance.IconsContainer, worldPositionStays: false);

            _dotRect = go.GetComponent<RectTransform>();
            _dotRect.sizeDelta = iconSize;

            _dotImage = go.GetComponent<Image>();
            _dotImage.sprite = iconSprite;
            _dotImage.color = GetIconColor();
        }

        private Color GetIconColor()
        {
            if (selectable == null || Core.GameManager.Instance == null)
            {
                return Color.white;
            }
            return selectable.OwnerPlayerId == Core.GameManager.Instance.LocalPlayerId ? allyColor : enemyColor;
        }

        private void Update()
        {
            if (_dotRect == null || MinimapController.Instance == null)
            {
                return;
            }

            _dotRect.anchoredPosition = MinimapController.Instance.WorldToIconLocalPosition(transform.position);
        }

        private void OnDestroy()
        {
            if (_dotRect != null)
            {
                Destroy(_dotRect.gameObject);
            }
        }
    }
}
