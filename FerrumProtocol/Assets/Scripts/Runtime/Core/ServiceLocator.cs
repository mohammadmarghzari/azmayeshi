using System;
using System.Collections.Generic;

namespace FerrumProtocol.Core
{
    /// <summary>
    /// Minimal static service registry used in place of a heavyweight DI container.
    /// Systems should depend on interfaces and resolve them here (usually once, cached
    /// in Awake/Start) rather than reaching for concrete singletons directly. This keeps
    /// gameplay classes testable in EditMode without a running scene.
    /// </summary>
    public static class ServiceLocator
    {
        private static readonly Dictionary<Type, object> Services = new Dictionary<Type, object>();

        public static void Register<T>(T instance) where T : class
        {
            Services[typeof(T)] = instance ?? throw new ArgumentNullException(nameof(instance));
        }

        public static bool TryResolve<T>(out T instance) where T : class
        {
            if (Services.TryGetValue(typeof(T), out var raw))
            {
                instance = (T)raw;
                return true;
            }

            instance = null;
            return false;
        }

        public static T Resolve<T>() where T : class
        {
            if (TryResolve<T>(out var instance))
            {
                return instance;
            }

            throw new InvalidOperationException($"No service of type {typeof(T).Name} is registered.");
        }

        public static void Unregister<T>() where T : class
        {
            Services.Remove(typeof(T));
        }

        /// <summary>Call when leaving play mode / tearing down a match so stale references don't leak between sessions.</summary>
        public static void Clear()
        {
            Services.Clear();
        }
    }
}
