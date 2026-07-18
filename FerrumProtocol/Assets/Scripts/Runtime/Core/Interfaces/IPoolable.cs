namespace FerrumProtocol.Core
{
    /// <summary>Optional hooks a pooled component can implement to reset itself on reuse.</summary>
    public interface IPoolable
    {
        void OnSpawnFromPool();
        void OnReturnToPool();
    }
}
