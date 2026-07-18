using System;
using FerrumProtocol.Core;
using UnityEngine;

namespace FerrumProtocol.Save
{
    /// <summary>Serializes/restores <see cref="SaveData"/> through a pluggable <see cref="ICloudSaveProvider"/> (local disk by default).</summary>
    public class SaveManager : MonoBehaviour
    {
        private ICloudSaveProvider _provider;

        private void Awake()
        {
            _provider = new LocalSaveProvider();
            ServiceLocator.Register(this);
        }

        private void OnDestroy() => ServiceLocator.Unregister<SaveManager>();

        /// <summary>Swap in a real backend later (Steam Cloud, custom service) without touching call sites.</summary>
        public void SetProvider(ICloudSaveProvider provider) => _provider = provider;

        public void Save(string slotName, SaveData data)
        {
            data.savedAtIso8601 = DateTime.UtcNow.ToString("o");
            string json = JsonUtility.ToJson(data, prettyPrint: true);
            _provider.Write(slotName, json);
        }

        public SaveData Load(string slotName)
        {
            string json = _provider.Read(slotName);
            return string.IsNullOrEmpty(json) ? null : JsonUtility.FromJson<SaveData>(json);
        }

        public bool HasSave(string slotName) => _provider.Exists(slotName);

        public void DeleteSave(string slotName) => _provider.Delete(slotName);
    }
}
