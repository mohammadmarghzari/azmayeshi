using UnityEngine;

namespace FerrumProtocol.FogOfWar
{
    /// <summary>
    /// Bridges <see cref="FogOfWarManager"/>'s grid texture and world bounds into the
    /// <c>Shaders/FogOfWarURP.shader</c> material on this object's Renderer. Drop this on a
    /// large flat quad positioned above the terrain (see SceneBootstrapper for the exact
    /// placement math once wired into the prototype scene).
    /// </summary>
    [RequireComponent(typeof(Renderer))]
    public class FogOfWarRenderer : MonoBehaviour
    {
        private static readonly int FogTexId = Shader.PropertyToID("_FogTex");
        private static readonly int FogWorldMinId = Shader.PropertyToID("_FogWorldMin");
        private static readonly int FogWorldSizeId = Shader.PropertyToID("_FogWorldSize");

        [SerializeField] private Vector2 worldOrigin = new Vector2(-100f, -100f);
        [SerializeField] private Vector2 worldSize = new Vector2(200f, 200f);

        private Renderer _renderer;
        private MaterialPropertyBlock _block;

        private void Awake()
        {
            _renderer = GetComponent<Renderer>();
            _block = new MaterialPropertyBlock();
        }

        private void LateUpdate()
        {
            if (FogOfWarManager.Instance == null || FogOfWarManager.Instance.FogTexture == null)
            {
                return;
            }

            _renderer.GetPropertyBlock(_block);
            _block.SetTexture(FogTexId, FogOfWarManager.Instance.FogTexture);
            _block.SetVector(FogWorldMinId, new Vector4(worldOrigin.x, worldOrigin.y, 0f, 0f));
            _block.SetVector(FogWorldSizeId, new Vector4(worldSize.x, worldSize.y, 0f, 0f));
            _renderer.SetPropertyBlock(_block);
        }
    }
}
