namespace FerrumProtocol.Save
{
    /// <summary>
    /// Abstraction over "where saves actually live" so a real backend (Steam Cloud, a custom
    /// service, etc.) can replace <see cref="LocalSaveProvider"/> in Phase 5 without touching
    /// any call site - everything goes through <see cref="SaveManager"/>.
    /// </summary>
    public interface ICloudSaveProvider
    {
        void Write(string slotName, string json);
        string Read(string slotName);
        bool Exists(string slotName);
        void Delete(string slotName);
    }
}
