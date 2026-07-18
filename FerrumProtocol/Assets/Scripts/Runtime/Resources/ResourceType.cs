namespace FerrumProtocol.Resources
{
    /// <summary>
    /// Two primary resources (Ferrite, Voltium) plus one scarce secondary strategic resource
    /// (CommandCells) per the GDD - see Documentation/GDD.md section 3.
    /// </summary>
    public enum ResourceType
    {
        Ferrite,
        Voltium,
        CommandCells
    }
}
