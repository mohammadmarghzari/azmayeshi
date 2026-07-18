using System.Collections.Generic;
using UnityEngine;

namespace FerrumProtocol.Save
{
    /// <summary>
    /// Local achievement tracking with a provider seam for a real platform backend later
    /// (Steamworks.NET, etc.) - mirrors the <see cref="ICloudSaveProvider"/> pattern so the
    /// same swap-the-backend-later approach applies to achievements too.
    /// </summary>
    public interface IAchievementProvider
    {
        void Unlock(string achievementId);
        bool IsUnlocked(string achievementId);
    }

    public class LocalAchievementProvider : IAchievementProvider
    {
        private readonly HashSet<string> _unlocked = new HashSet<string>();
        public void Unlock(string achievementId) => _unlocked.Add(achievementId);
        public bool IsUnlocked(string achievementId) => _unlocked.Contains(achievementId);
    }

    public class AchievementManager : MonoBehaviour
    {
        private IAchievementProvider _provider = new LocalAchievementProvider();

        public event System.Action<string> OnAchievementUnlocked;

        public void SetProvider(IAchievementProvider provider) => _provider = provider;

        public void Unlock(string achievementId)
        {
            if (_provider.IsUnlocked(achievementId))
            {
                return;
            }

            _provider.Unlock(achievementId);
            OnAchievementUnlocked?.Invoke(achievementId);
        }

        public bool IsUnlocked(string achievementId) => _provider.IsUnlocked(achievementId);
    }
}
