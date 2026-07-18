using FerrumProtocol.Selection;
using UnityEngine;

namespace FerrumProtocol.FogOfWar
{
    /// <summary>Attach to any unit/building that should push back the fog of war (i.e. everything with sight).</summary>
    public class FogRevealer : MonoBehaviour
    {
        [SerializeField] private float sightRadius = 15f;
        [SerializeField] private Selectable selectable;

        public float SightRadius => sightRadius;
        public int OwnerPlayerId => selectable != null ? selectable.OwnerPlayerId : -1;

        public void SetSightRadius(float radius) => sightRadius = radius;

        private void OnEnable()
        {
            if (FogOfWarManager.Instance != null)
            {
                FogOfWarManager.Instance.Register(this);
            }
        }

        private void OnDisable()
        {
            if (FogOfWarManager.Instance != null)
            {
                FogOfWarManager.Instance.Unregister(this);
            }
        }
    }
}
