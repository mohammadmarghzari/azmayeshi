using UnityEngine;

namespace FerrumProtocol.Core
{
    /// <summary>Anything that can be box-selected / single-clicked by the player.</summary>
    public interface ISelectable
    {
        Transform Transform { get; }
        int OwnerPlayerId { get; }
        bool IsSelected { get; }
        void SetSelected(bool selected);
    }
}
