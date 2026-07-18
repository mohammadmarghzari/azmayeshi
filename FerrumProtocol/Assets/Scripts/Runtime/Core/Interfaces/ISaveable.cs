namespace FerrumProtocol.Core
{
    /// <summary>Implemented by systems that need to persist/restore state across saves.</summary>
    public interface ISaveable
    {
        string SaveKey { get; }
        string CaptureState();
        void RestoreState(string serializedState);
    }
}
