using System.Collections.Generic;
using FerrumProtocol.Core;
using UnityEngine;

namespace FerrumProtocol.FogOfWar
{
    public enum FogState : byte
    {
        Unseen = 0,
        Explored = 1,
        Visible = 2
    }

    /// <summary>
    /// Grid-based fog of war: a low-resolution byte grid over the map's XZ extents tracks
    /// Unseen/Explored/Visible per cell, updated from registered <see cref="FogRevealer"/>
    /// sight radii on a fixed tick (not every frame - fog doesn't need per-frame precision).
    /// The grid is uploaded to a Texture2D that <c>Shaders/FogOfWarURP.shader</c> samples as
    /// a world-aligned overlay. This is far cheaper than per-object visibility raycasts once
    /// unit counts get into the hundreds.
    /// </summary>
    public class FogOfWarManager : MonoBehaviour
    {
        public static FogOfWarManager Instance { get; private set; }

        [SerializeField] private Vector2 worldOrigin = new Vector2(-100f, -100f);
        [SerializeField] private Vector2 worldSize = new Vector2(200f, 200f);
        [SerializeField] private int cellsPerWorldUnit = 1;
        [SerializeField] private float updateIntervalSeconds = 0.2f;
        [SerializeField] private int localPlayerId = 0;

        private byte[] _grid;
        private int _width;
        private int _height;
        private Texture2D _fogTexture;
        private float _timer;

        private readonly List<FogRevealer> _revealers = new List<FogRevealer>();

        public Texture2D FogTexture => _fogTexture;

        private void Awake()
        {
            Instance = this;
            ServiceLocator.Register(this);

            _width = Mathf.Max(1, Mathf.RoundToInt(worldSize.x * cellsPerWorldUnit));
            _height = Mathf.Max(1, Mathf.RoundToInt(worldSize.y * cellsPerWorldUnit));
            _grid = new byte[_width * _height];
            _fogTexture = new Texture2D(_width, _height, TextureFormat.R8, mipChain: false)
            {
                wrapMode = TextureWrapMode.Clamp,
                filterMode = FilterMode.Bilinear
            };
        }

        private void OnDestroy()
        {
            if (Instance == this)
            {
                Instance = null;
            }
            ServiceLocator.Unregister<FogOfWarManager>();
        }

        public void Register(FogRevealer revealer) => _revealers.Add(revealer);
        public void Unregister(FogRevealer revealer) => _revealers.Remove(revealer);

        private void Update()
        {
            _timer -= Time.deltaTime;
            if (_timer > 0f)
            {
                return;
            }
            _timer = updateIntervalSeconds;
            RecomputeVisibility();
        }

        private void RecomputeVisibility()
        {
            for (int i = 0; i < _grid.Length; i++)
            {
                if (_grid[i] == (byte)FogState.Visible)
                {
                    _grid[i] = (byte)FogState.Explored;
                }
            }

            foreach (var revealer in _revealers)
            {
                if (revealer.OwnerPlayerId != localPlayerId)
                {
                    continue;
                }

                RevealCircle(revealer.transform.position, revealer.SightRadius);
            }

            _fogTexture.SetPixelData(_grid, 0);
            _fogTexture.Apply(false);
        }

        private void RevealCircle(Vector3 worldPos, float radius)
        {
            WorldToCell(worldPos, out int cx, out int cy);
            int cellRadius = Mathf.CeilToInt(radius * cellsPerWorldUnit);

            int minX = Mathf.Clamp(cx - cellRadius, 0, _width - 1);
            int maxX = Mathf.Clamp(cx + cellRadius, 0, _width - 1);
            int minY = Mathf.Clamp(cy - cellRadius, 0, _height - 1);
            int maxY = Mathf.Clamp(cy + cellRadius, 0, _height - 1);
            float radiusSqr = cellRadius * cellRadius;

            for (int y = minY; y <= maxY; y++)
            {
                for (int x = minX; x <= maxX; x++)
                {
                    float dx = x - cx;
                    float dy = y - cy;
                    if (dx * dx + dy * dy <= radiusSqr)
                    {
                        _grid[y * _width + x] = (byte)FogState.Visible;
                    }
                }
            }
        }

        private void WorldToCell(Vector3 worldPos, out int cx, out int cy)
        {
            cx = Mathf.Clamp(Mathf.RoundToInt((worldPos.x - worldOrigin.x) * cellsPerWorldUnit), 0, _width - 1);
            cy = Mathf.Clamp(Mathf.RoundToInt((worldPos.z - worldOrigin.y) * cellsPerWorldUnit), 0, _height - 1);
        }

        public FogState GetState(Vector3 worldPos)
        {
            WorldToCell(worldPos, out int cx, out int cy);
            return (FogState)_grid[cy * _width + cx];
        }

        public bool IsVisible(Vector3 worldPos) => GetState(worldPos) == FogState.Visible;
        public bool IsExplored(Vector3 worldPos) => GetState(worldPos) != FogState.Unseen;
    }
}
