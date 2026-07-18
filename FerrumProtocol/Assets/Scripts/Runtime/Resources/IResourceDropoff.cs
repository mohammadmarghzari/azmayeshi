namespace FerrumProtocol.Resources
{
    /// <summary>Implemented by buildings (Refinery, Command Center) that harvesters can deposit resources at.</summary>
    public interface IResourceDropoff
    {
        int OwnerPlayerId { get; }
        UnityEngine.Transform Transform { get; }
        void Deposit(ResourceType type, int amount);
    }
}
