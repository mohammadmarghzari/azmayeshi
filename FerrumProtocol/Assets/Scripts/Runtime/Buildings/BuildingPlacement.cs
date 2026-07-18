using FerrumProtocol.Core;
using FerrumProtocol.Data;
using FerrumProtocol.Resources;
using UnityEngine;
using UnityEngine.InputSystem;

namespace FerrumProtocol.Buildings
{
    /// <summary>
    /// Ghost-preview building placement: call <see cref="BeginPlacement"/> (e.g. from a build
    /// menu button) to start following the mouse with a translucent preview, left-click to
    /// confirm (if the site is clear and affordable), right-click/Escape to cancel.
    /// </summary>
    public class BuildingPlacement : MonoBehaviour
    {
        [SerializeField] private Camera targetCamera;
        [SerializeField] private LayerMask groundLayer = ~0;
        [SerializeField] private LayerMask obstructionLayers = ~0;
        [SerializeField] private Material validGhostMaterial;
        [SerializeField] private Material invalidGhostMaterial;
        [SerializeField] private int localPlayerId = 0;

        private GameObject _ghostInstance;
        private BuildingDataSO _pendingData;
        private bool _isPlacing;
        private Renderer[] _ghostRenderers;

        public bool IsPlacing => _isPlacing;

        private void Awake()
        {
            if (targetCamera == null)
            {
                targetCamera = Camera.main;
            }
        }

        public void BeginPlacement(BuildingDataSO data)
        {
            if (data == null || data.prefab == null)
            {
                return;
            }

            CancelPlacement();

            _pendingData = data;
            _ghostInstance = Instantiate(data.prefab);
            SetGhostComponentsEnabled(_ghostInstance, false);
            _ghostRenderers = _ghostInstance.GetComponentsInChildren<Renderer>();
            _isPlacing = true;
        }

        public void CancelPlacement()
        {
            if (_ghostInstance != null)
            {
                Destroy(_ghostInstance);
            }
            _ghostInstance = null;
            _pendingData = null;
            _isPlacing = false;
        }

        private void Update()
        {
            if (!_isPlacing)
            {
                return;
            }

            if (Keyboard.current != null && Keyboard.current.escapeKey.wasPressedThisFrame)
            {
                CancelPlacement();
                return;
            }

            if (Mouse.current == null || targetCamera == null)
            {
                return;
            }

            if (Mouse.current.rightButton.wasPressedThisFrame)
            {
                CancelPlacement();
                return;
            }

            Vector2 screenPos = Mouse.current.position.ReadValue();
            Ray ray = targetCamera.ScreenPointToRay(screenPos);

            if (!Physics.Raycast(ray, out RaycastHit hit, 1000f, groundLayer))
            {
                return;
            }

            _ghostInstance.transform.position = hit.point;
            bool valid = IsSiteValid(hit.point);
            ApplyGhostMaterial(valid);

            if (valid && Mouse.current.leftButton.wasPressedThisFrame)
            {
                TryConfirmPlacement(hit.point);
            }
        }

        private bool IsSiteValid(Vector3 position)
        {
            // Approximate footprint check - a 4x4 world unit box centered on the cursor. Filtered
            // to actual placeable obstacles (units/buildings/resource nodes) rather than every
            // collider on obstructionLayers, since the flat ground/terrain collider itself would
            // otherwise always overlap and make every site look invalid.
            Vector3 halfExtents = new Vector3(2f, 2f, 2f);
            var overlaps = Physics.OverlapBox(position + Vector3.up, halfExtents, Quaternion.identity, obstructionLayers);

            foreach (var col in overlaps)
            {
                if (col.GetComponentInParent<Selection.Selectable>() != null ||
                    col.GetComponentInParent<ResourceNode>() != null)
                {
                    return false;
                }
            }

            return true;
        }

        private void TryConfirmPlacement(Vector3 position)
        {
            var economy = GameManager.Instance != null ? GameManager.Instance.GetPlayerEconomy(localPlayerId) : null;
            if (economy == null || !economy.TrySpend(_pendingData.ferriteCost, _pendingData.voltiumCost))
            {
                return;
            }

            var instance = Instantiate(_pendingData.prefab, position, Quaternion.identity);
            if (instance.TryGetComponent<Selection.Selectable>(out var selectable))
            {
                selectable.SetOwner(localPlayerId);
            }

            CancelPlacement();
        }

        private void ApplyGhostMaterial(bool valid)
        {
            if (_ghostRenderers == null)
            {
                return;
            }

            Material mat = valid ? validGhostMaterial : invalidGhostMaterial;
            if (mat == null)
            {
                return;
            }

            foreach (var renderer in _ghostRenderers)
            {
                renderer.material = mat;
            }
        }

        private static void SetGhostComponentsEnabled(GameObject go, bool enabled)
        {
            foreach (var behaviour in go.GetComponentsInChildren<MonoBehaviour>())
            {
                behaviour.enabled = enabled;
            }
            foreach (var collider in go.GetComponentsInChildren<Collider>())
            {
                collider.enabled = false;
            }
        }
    }
}
