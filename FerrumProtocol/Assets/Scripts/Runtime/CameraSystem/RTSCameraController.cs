using UnityEngine;

namespace FerrumProtocol.CameraSystem
{
    /// <summary>
    /// Classic RTS camera rig: a pivot point on the ground that the actual camera orbits/hovers
    /// above at a pitched angle. Supports keyboard/edge-scroll panning, scroll-wheel zoom,
    /// free rotation, and smooth interpolation on every axis so input never feels snappy/raw.
    /// Drive this via <see cref="RequestPan"/>/<see cref="RequestZoom"/>/<see cref="RequestRotate"/>
    /// from an input handler - this class has no direct Input System dependency so it stays testable
    /// and reusable (e.g. for a replay free-camera later).
    /// </summary>
    [RequireComponent(typeof(Camera))]
    public class RTSCameraController : MonoBehaviour
    {
        [Header("Rig")]
        [SerializeField] private Transform pivot;
        [SerializeField] private float pitchDegrees = 55f;

        [Header("Pan")]
        [SerializeField] private float panSpeed = 30f;
        [SerializeField] private float edgeScrollBorderPixels = 12f;
        [SerializeField] private bool edgeScrollEnabled = true;
        [SerializeField] private CameraBounds bounds = CameraBounds.FromCenterExtents(Vector3.zero, new Vector2(100f, 100f));

        [Header("Zoom")]
        [SerializeField] private float minDistance = 10f;
        [SerializeField] private float maxDistance = 60f;
        [SerializeField] private float zoomSpeed = 4f;

        [Header("Rotate")]
        [SerializeField] private float rotateSpeed = 90f;

        [Header("Smoothing")]
        [SerializeField] private float positionSmoothTime = 0.12f;
        [SerializeField] private float zoomSmoothTime = 0.15f;
        [SerializeField] private float rotationSmoothTime = 0.1f;

        private Camera _camera;
        private Vector3 _targetPivotPosition;
        private float _targetDistance;
        private float _targetYaw;

        private Vector3 _pivotVelocity;
        private float _distanceVelocity;
        private float _yawVelocity;
        private float _currentDistance;
        private float _currentYaw;

        public void SetBounds(CameraBounds newBounds) => bounds = newBounds;

        private void Awake()
        {
            _camera = GetComponent<Camera>();

            if (pivot == null)
            {
                var pivotGo = new GameObject($"{name}_Pivot");
                pivot = pivotGo.transform;
            }

            _targetPivotPosition = pivot.position;
            _targetDistance = Mathf.Clamp((minDistance + maxDistance) * 0.5f, minDistance, maxDistance);
            _currentDistance = _targetDistance;
            _targetYaw = pivot.eulerAngles.y;
            _currentYaw = _targetYaw;

            ApplyTransform(immediate: true);
        }

        private void Update()
        {
            if (edgeScrollEnabled)
            {
                ApplyEdgeScroll();
            }

            SmoothAndApply();
        }

        /// <summary>Direction is in camera-relative XZ space, magnitude 0-1 per axis.</summary>
        public void RequestPan(Vector2 direction, float deltaTime)
        {
            if (direction.sqrMagnitude < 0.0001f)
            {
                return;
            }

            Vector3 forward = Quaternion.Euler(0f, _targetYaw, 0f) * Vector3.forward;
            Vector3 right = Quaternion.Euler(0f, _targetYaw, 0f) * Vector3.right;
            Vector3 move = (forward * direction.y + right * direction.x) * (panSpeed * deltaTime);
            _targetPivotPosition = bounds.Clamp(_targetPivotPosition + move);
        }

        public void RequestZoom(float scrollDelta)
        {
            _targetDistance = Mathf.Clamp(_targetDistance - scrollDelta * zoomSpeed, minDistance, maxDistance);
        }

        public void RequestRotate(float yawDelta)
        {
            _targetYaw += yawDelta * rotateSpeed * Time.deltaTime;
        }

        /// <summary>Directly jump the camera focus to a world point (e.g. control-group double-tap, minimap click).</summary>
        public void JumpTo(Vector3 worldPoint)
        {
            _targetPivotPosition = bounds.Clamp(worldPoint);
            pivot.position = _targetPivotPosition;
            ApplyTransform(immediate: true);
        }

        private void ApplyEdgeScroll()
        {
            Vector2 mousePos = UnityEngine.InputSystem.Mouse.current != null
                ? UnityEngine.InputSystem.Mouse.current.position.ReadValue()
                : (Vector2)Input.mousePosition;

            Vector2 direction = Vector2.zero;
            if (mousePos.x <= edgeScrollBorderPixels) direction.x -= 1f;
            if (mousePos.x >= Screen.width - edgeScrollBorderPixels) direction.x += 1f;
            if (mousePos.y <= edgeScrollBorderPixels) direction.y -= 1f;
            if (mousePos.y >= Screen.height - edgeScrollBorderPixels) direction.y += 1f;

            if (direction.sqrMagnitude > 0f)
            {
                RequestPan(direction.normalized, Time.deltaTime);
            }
        }

        private void SmoothAndApply()
        {
            pivot.position = Vector3.SmoothDamp(pivot.position, _targetPivotPosition, ref _pivotVelocity, positionSmoothTime);
            _currentDistance = Mathf.SmoothDamp(_currentDistance, _targetDistance, ref _distanceVelocity, zoomSmoothTime);
            _currentYaw = Mathf.SmoothDampAngle(_currentYaw, _targetYaw, ref _yawVelocity, rotationSmoothTime);
            ApplyTransform(immediate: false);
        }

        private void ApplyTransform(bool immediate)
        {
            if (immediate)
            {
                _currentDistance = _targetDistance;
                _currentYaw = _targetYaw;
            }

            Quaternion rotation = Quaternion.Euler(pitchDegrees, _currentYaw, 0f);
            Vector3 offset = rotation * new Vector3(0f, 0f, -_currentDistance);
            transform.position = pivot.position + offset;
            transform.rotation = rotation;
        }
    }
}
