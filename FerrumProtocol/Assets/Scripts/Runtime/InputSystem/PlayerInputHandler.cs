using FerrumProtocol.CameraSystem;
using FerrumProtocol.Core;
using FerrumProtocol.Selection;
using UnityEngine;
using UnityEngine.InputSystem;

namespace FerrumProtocol.InputSystem
{
    /// <summary>
    /// Single entry point translating raw Input System events into camera moves, selection
    /// actions, and unit commands. Reads the actions by name from an assigned
    /// <see cref="InputActionAsset"/> rather than relying on a generated wrapper class, since
    /// that generated class is produced by the Unity Editor on import and can't be hand-authored.
    /// </summary>
    public class PlayerInputHandler : MonoBehaviour
    {
        [SerializeField] private InputActionAsset controls;
        [SerializeField] private RTSCameraController cameraController;
        [SerializeField] private SelectionManager selectionManager;
        [SerializeField] private Camera targetCamera;
        [SerializeField] private LayerMask commandRaycastLayers = ~0;

        private InputAction _move;
        private InputAction _zoom;
        private InputAction _rotateDelta;
        private InputAction _rotateHeld;
        private InputAction _mouseDelta;
        private InputAction _select;
        private InputAction _selectModifier;
        private InputAction _command;
        private InputAction _deselect;

        private bool _isDragSelecting;
        private Vector2 _mousePos;

        private void Awake()
        {
            if (controls == null)
            {
                Debug.LogError($"{nameof(PlayerInputHandler)} has no InputActionAsset assigned.");
                enabled = false;
                return;
            }

            var map = controls.FindActionMap("Gameplay", throwIfNotFound: true);
            _move = map.FindAction("Move");
            _zoom = map.FindAction("Zoom");
            _rotateDelta = map.FindAction("RotateDelta");
            _rotateHeld = map.FindAction("RotateHeld");
            _mouseDelta = map.FindAction("MouseDelta");
            _select = map.FindAction("Select");
            _selectModifier = map.FindAction("SelectModifier");
            _command = map.FindAction("Command");
            _deselect = map.FindAction("Deselect");

            _select.started += OnSelectStarted;
            _select.canceled += OnSelectCanceled;
            _command.performed += OnCommandPerformed;
            _deselect.performed += OnDeselectPerformed;

            if (targetCamera == null)
            {
                targetCamera = cameraController != null ? cameraController.GetComponent<Camera>() : Camera.main;
            }
        }

        private void OnEnable() => controls?.Enable();
        private void OnDisable() => controls?.Disable();

        private void Update()
        {
            _mousePos = Mouse.current != null ? Mouse.current.position.ReadValue() : (Vector2)Input.mousePosition;

            if (cameraController != null)
            {
                cameraController.RequestPan(_move.ReadValue<Vector2>(), Time.deltaTime);
                cameraController.RequestZoom(_zoom.ReadValue<float>());

                float keyboardRotate = _rotateDelta.ReadValue<float>();
                if (Mathf.Abs(keyboardRotate) > 0.01f)
                {
                    cameraController.RequestRotate(keyboardRotate);
                }
                else if (_rotateHeld.IsPressed())
                {
                    float mouseYaw = _mouseDelta.ReadValue<Vector2>().x * 0.2f;
                    cameraController.RequestRotate(mouseYaw);
                }
            }

            if (_isDragSelecting)
            {
                // Visual feedback handled by SelectionBoxUI polling SelectionManager each frame.
            }
        }

        private void OnSelectStarted(InputAction.CallbackContext ctx)
        {
            _isDragSelecting = true;
            selectionManager?.BeginBoxSelect(_mousePos);
        }

        private void OnSelectCanceled(InputAction.CallbackContext ctx)
        {
            if (!_isDragSelecting || selectionManager == null || targetCamera == null)
            {
                return;
            }

            _isDragSelecting = false;
            bool additive = _selectModifier.IsPressed();
            selectionManager.EndBoxSelect(_mousePos, additive, targetCamera);
        }

        private void OnCommandPerformed(InputAction.CallbackContext ctx)
        {
            if (selectionManager == null || targetCamera == null || selectionManager.CurrentSelection.Count == 0)
            {
                return;
            }

            Ray ray = targetCamera.ScreenPointToRay(_mousePos);
            if (!Physics.Raycast(ray, out RaycastHit hit, 1000f, commandRaycastLayers))
            {
                return;
            }

            var damageable = hit.collider.GetComponentInParent<IDamageable>();
            var resourceNode = hit.collider.GetComponentInParent<FerrumProtocol.Resources.ResourceNode>();
            bool queued = _selectModifier.IsPressed();

            CommandType type;
            Transform targetTransform = null;

            if (resourceNode != null && !resourceNode.IsDepleted)
            {
                type = CommandType.Harvest;
                targetTransform = resourceNode.transform;
            }
            else if (damageable != null && !damageable.IsDead)
            {
                type = CommandType.Attack;
                targetTransform = hit.collider.transform;
            }
            else
            {
                type = CommandType.AttackMove;
            }

            var command = new UnitCommand
            {
                TargetPoint = hit.point,
                Queued = queued,
                TargetNetId = -1,
                TargetTransform = targetTransform,
                Type = type
            };

            selectionManager.IssueCommandToSelection(command);
        }

        private void OnDeselectPerformed(InputAction.CallbackContext ctx)
        {
            selectionManager?.ClearSelection();
        }
    }
}
