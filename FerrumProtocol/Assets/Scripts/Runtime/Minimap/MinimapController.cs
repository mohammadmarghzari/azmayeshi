using FerrumProtocol.CameraSystem;
using FerrumProtocol.Core;
using UnityEngine;
using UnityEngine.EventSystems;
using UnityEngine.UI;

namespace FerrumProtocol.Minimap
{
    /// <summary>
    /// Drives the minimap: a top-down orthographic camera renders the map into a RenderTexture
    /// shown on a UI RawImage; clicking/dragging on that image jumps the main RTS camera to the
    /// corresponding world position. <see cref="MinimapIcon"/> instances register here to be
    /// placed as small dots inside <see cref="iconsContainer"/>.
    /// </summary>
    public class MinimapController : MonoBehaviour, IPointerClickHandler, IDragHandler
    {
        public static MinimapController Instance { get; private set; }

        [SerializeField] private Camera minimapCamera;
        [SerializeField] private RawImage minimapImage;
        [SerializeField] private RectTransform iconsContainer;
        [SerializeField] private RTSCameraController mainCameraController;

        private RectTransform _imageRect;

        private void Awake()
        {
            Instance = this;
            ServiceLocator.Register(this);
            _imageRect = minimapImage != null ? minimapImage.rectTransform : GetComponent<RectTransform>();
        }

        private void OnDestroy()
        {
            if (Instance == this) Instance = null;
            ServiceLocator.Unregister<MinimapController>();
        }

        /// <summary>World-space half-extents visible on the minimap, derived from the top-down camera's orthographic size.</summary>
        private Vector2 GetWorldHalfExtents()
        {
            float halfHeight = minimapCamera.orthographicSize;
            float halfWidth = halfHeight * minimapCamera.aspect;
            return new Vector2(halfWidth, halfHeight);
        }

        public Vector2 WorldToIconLocalPosition(Vector3 worldPos)
        {
            Vector2 half = GetWorldHalfExtents();
            Vector3 camPos = minimapCamera.transform.position;

            float normalizedX = (worldPos.x - camPos.x) / (half.x * 2f) + 0.5f;
            float normalizedZ = (worldPos.z - camPos.z) / (half.y * 2f) + 0.5f;

            Rect rect = iconsContainer.rect;
            return new Vector2(rect.width * (normalizedX - 0.5f), rect.height * (normalizedZ - 0.5f));
        }

        public void OnPointerClick(PointerEventData eventData) => JumpMainCameraToClick(eventData);
        public void OnDrag(PointerEventData eventData) => JumpMainCameraToClick(eventData);

        private void JumpMainCameraToClick(PointerEventData eventData)
        {
            if (mainCameraController == null || minimapCamera == null)
            {
                return;
            }

            if (!RectTransformUtility.ScreenPointToLocalPointInRectangle(_imageRect, eventData.position, eventData.pressEventCamera, out Vector2 localPoint))
            {
                return;
            }

            Rect rect = _imageRect.rect;
            float normalizedX = (localPoint.x - rect.x) / rect.width;
            float normalizedY = (localPoint.y - rect.y) / rect.height;

            Vector2 half = GetWorldHalfExtents();
            Vector3 camPos = minimapCamera.transform.position;
            float worldX = camPos.x + (normalizedX - 0.5f) * half.x * 2f;
            float worldZ = camPos.z + (normalizedY - 0.5f) * half.y * 2f;

            mainCameraController.JumpTo(new Vector3(worldX, 0f, worldZ));
        }

        public RectTransform IconsContainer => iconsContainer;
    }
}
