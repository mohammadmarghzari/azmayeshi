using FerrumProtocol.Resources;
using UnityEngine;

namespace FerrumProtocol.Data
{
    [CreateAssetMenu(menuName = "Ferrum Protocol/Economy/Resource Type Info", fileName = "NewResourceTypeInfo")]
    public class ResourceTypeDataSO : ScriptableObject
    {
        public ResourceType type;
        public string displayName = "Resource";
        public Sprite icon;
        [Tooltip("Strategic resources are scarce, fixed, non-respawning map pickups (see GDD).")]
        public bool isStrategic;
    }
}
