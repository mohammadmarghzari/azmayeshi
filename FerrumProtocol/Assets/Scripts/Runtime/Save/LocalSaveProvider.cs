using System.IO;
using UnityEngine;

namespace FerrumProtocol.Save
{
    /// <summary>Default provider: writes JSON to Application.persistentDataPath. Always available, works offline, no third-party dependency.</summary>
    public class LocalSaveProvider : ICloudSaveProvider
    {
        private static string PathFor(string slotName) => Path.Combine(Application.persistentDataPath, $"{slotName}.json");

        public void Write(string slotName, string json) => File.WriteAllText(PathFor(slotName), json);

        public string Read(string slotName) => Exists(slotName) ? File.ReadAllText(PathFor(slotName)) : null;

        public bool Exists(string slotName) => File.Exists(PathFor(slotName));

        public void Delete(string slotName)
        {
            if (Exists(slotName))
            {
                File.Delete(PathFor(slotName));
            }
        }
    }
}
