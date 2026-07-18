using System.Collections.Generic;
using FerrumProtocol.Core;
using UnityEngine;

namespace FerrumProtocol.Selection
{
    /// <summary>
    /// Owns the local player's current selection: single click, drag-box, and forwards
    /// commands (move/attack-move/etc.) to every selected <see cref="ICommandable"/>.
    /// One instance per local player (in split-screen/hotseat this would be per-viewport;
    /// for Phase 1 there is a single local player).
    /// </summary>
    public class SelectionManager : MonoBehaviour
    {
        public static SelectionManager Instance { get; private set; }

        [SerializeField] private int localPlayerId = 0;
        [SerializeField] private LayerMask groundLayer = ~0;
        [SerializeField] private LayerMask selectableLayer = ~0;

        private readonly List<Selectable> _registered = new List<Selectable>();
        private readonly List<Selectable> _currentSelection = new List<Selectable>();

        public IReadOnlyList<Selectable> CurrentSelection => _currentSelection;
        public event System.Action<IReadOnlyList<Selectable>> OnSelectionChanged;

        private Vector2 _boxStartScreenPos;
        private bool _boxSelectActive;
        public bool IsBoxSelecting => _boxSelectActive;

        private void Awake()
        {
            Instance = this;
            ServiceLocator.Register(this);
        }

        private void OnDestroy()
        {
            if (Instance == this)
            {
                Instance = null;
            }
            ServiceLocator.Unregister<SelectionManager>();
        }

        public void RegisterSelectable(Selectable selectable) => _registered.Add(selectable);

        /// <summary>All registered selectables owned by a given player - used by AI systems to survey their own army/base without a per-frame scene scan.</summary>
        public IEnumerable<Selectable> GetAllOwnedBy(int ownerId)
        {
            foreach (var s in _registered)
            {
                if (s != null && s.OwnerPlayerId == ownerId)
                {
                    yield return s;
                }
            }
        }

        public void UnregisterSelectable(Selectable selectable)
        {
            _registered.Remove(selectable);
            _currentSelection.Remove(selectable);
        }

        public void BeginBoxSelect(Vector2 screenPos)
        {
            _boxStartScreenPos = screenPos;
            _boxSelectActive = true;
        }

        public Rect GetCurrentBoxRect(Vector2 currentScreenPos)
        {
            float xMin = Mathf.Min(_boxStartScreenPos.x, currentScreenPos.x);
            float yMin = Mathf.Min(_boxStartScreenPos.y, currentScreenPos.y);
            float width = Mathf.Abs(currentScreenPos.x - _boxStartScreenPos.x);
            float height = Mathf.Abs(currentScreenPos.y - _boxStartScreenPos.y);
            return new Rect(xMin, yMin, width, height);
        }

        private const float ClickDragThresholdPixels = 6f;

        public void EndBoxSelect(Vector2 screenPos, bool additive, Camera cam)
        {
            _boxSelectActive = false;
            Rect box = GetCurrentBoxRect(screenPos);

            if (box.width < ClickDragThresholdPixels && box.height < ClickDragThresholdPixels)
            {
                SelectSingleAtScreenPoint(screenPos, cam, additive);
                return;
            }

            if (!additive)
            {
                ClearSelectionInternal();
            }

            foreach (var selectable in _registered)
            {
                if (selectable.OwnerPlayerId != localPlayerId)
                {
                    continue;
                }

                Vector3 screenPoint = cam.WorldToScreenPoint(selectable.Transform.position);
                if (screenPoint.z < 0f)
                {
                    continue;
                }

                // Screen-space Y is bottom-up for WorldToScreenPoint, Rect uses top-down for GUI;
                // callers pass GUI-space rects, so flip once here.
                Vector2 flipped = new Vector2(screenPoint.x, Screen.height - screenPoint.y);
                if (box.Contains(flipped) && !_currentSelection.Contains(selectable))
                {
                    AddToSelection(selectable);
                }
            }

            NotifySelectionChanged();
        }

        public void SelectSingleAtScreenPoint(Vector2 screenPos, Camera cam, bool additive)
        {
            if (!additive)
            {
                ClearSelectionInternal();
            }

            Ray ray = cam.ScreenPointToRay(screenPos);
            if (Physics.Raycast(ray, out RaycastHit hit, 1000f, selectableLayer))
            {
                var selectable = hit.collider.GetComponentInParent<Selectable>();
                if (selectable != null && selectable.OwnerPlayerId == localPlayerId)
                {
                    if (additive && _currentSelection.Contains(selectable))
                    {
                        RemoveFromSelection(selectable);
                    }
                    else
                    {
                        AddToSelection(selectable);
                    }
                }
            }

            NotifySelectionChanged();
        }

        public void SetSelection(IEnumerable<Selectable> selectables)
        {
            ClearSelectionInternal();
            foreach (var s in selectables)
            {
                AddToSelection(s);
            }
            NotifySelectionChanged();
        }

        public void ClearSelection()
        {
            ClearSelectionInternal();
            NotifySelectionChanged();
        }

        private void AddToSelection(Selectable selectable)
        {
            _currentSelection.Add(selectable);
            selectable.SetSelected(true);
        }

        private void RemoveFromSelection(Selectable selectable)
        {
            _currentSelection.Remove(selectable);
            selectable.SetSelected(false);
        }

        private void ClearSelectionInternal()
        {
            foreach (var s in _currentSelection)
            {
                s.SetSelected(false);
            }
            _currentSelection.Clear();
        }

        private void NotifySelectionChanged()
        {
            OnSelectionChanged?.Invoke(_currentSelection);
        }

        public void IssueCommandToSelection(UnitCommand command)
        {
            foreach (var selectable in _currentSelection)
            {
                var commandable = selectable.GetComponent<ICommandable>();
                commandable?.IssueCommand(command);
            }
        }
    }
}
