using FerrumProtocol.Core;
using UnityEngine;

namespace FerrumProtocol.Selection
{
    /// <summary>Attach to any unit/building GameObject that should be selectable by the player.</summary>
    public class Selectable : MonoBehaviour, ISelectable
    {
        [SerializeField] private int ownerPlayerId;
        [SerializeField] private GameObject selectionIndicator;

        public Transform Transform => transform;
        public int OwnerPlayerId => ownerPlayerId;
        public bool IsSelected { get; private set; }

        public event System.Action<bool> OnSelectionChanged;

        public void SetOwner(int playerId) => ownerPlayerId = playerId;

        private void Awake()
        {
            if (SelectionManager.Instance != null)
            {
                SelectionManager.Instance.RegisterSelectable(this);
            }
        }

        private void OnDestroy()
        {
            if (SelectionManager.Instance != null)
            {
                SelectionManager.Instance.UnregisterSelectable(this);
            }
        }

        public void SetSelected(bool selected)
        {
            if (IsSelected == selected)
            {
                return;
            }

            IsSelected = selected;

            if (selectionIndicator != null)
            {
                selectionIndicator.SetActive(selected);
            }

            OnSelectionChanged?.Invoke(selected);
        }
    }
}
