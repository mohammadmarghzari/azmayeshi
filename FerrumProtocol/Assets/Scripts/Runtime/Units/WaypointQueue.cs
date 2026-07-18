using System.Collections.Generic;
using FerrumProtocol.Core;
using UnityEngine;

namespace FerrumProtocol.Units
{
    /// <summary>Shift-queue of pending orders for a single unit (classic RTS "shift-click to queue waypoints").</summary>
    public class WaypointQueue
    {
        private readonly Queue<UnitCommand> _pending = new Queue<UnitCommand>();

        public int Count => _pending.Count;

        /// <summary>Appends a shift-queued order. Non-queued orders are handled entirely by the
        /// caller (UnitController clears and executes them directly) and never reach this queue.</summary>
        public void SetOrder(UnitCommand command) => _pending.Enqueue(command);

        public bool TryDequeueNext(out UnitCommand command)
        {
            if (_pending.Count > 0)
            {
                command = _pending.Dequeue();
                return true;
            }

            command = default;
            return false;
        }

        public void Clear() => _pending.Clear();
    }
}
