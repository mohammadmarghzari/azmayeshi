using System.Collections.Generic;
using FerrumProtocol.CameraSystem;
using FerrumProtocol.Core;
using UnityEngine;
using UnityEngine.InputSystem;

namespace FerrumProtocol.Selection
{
    /// <summary>
    /// Classic control groups: Ctrl+0-9 assigns the current selection to a slot,
    /// pressing the digit alone recalls it, and pressing it twice quickly also
    /// snaps the camera to the group (a small quality-of-life staple of the genre).
    /// </summary>
    public class ControlGroupManager : MonoBehaviour
    {
        [SerializeField] private RTSCameraController cameraController;
        [SerializeField] private float doubleTapWindow = 0.35f;

        private readonly Dictionary<int, List<Selectable>> _groups = new Dictionary<int, List<Selectable>>();
        private int _lastRecalledGroup = -1;
        private float _lastRecallTime;

        private void Update()
        {
            var keyboard = Keyboard.current;
            if (keyboard == null)
            {
                return;
            }

            bool ctrlHeld = keyboard.leftCtrlKey.isPressed || keyboard.rightCtrlKey.isPressed;

            for (int i = 0; i <= 9; i++)
            {
                var key = GetDigitKey(keyboard, i);
                if (key == null || !key.wasPressedThisFrame)
                {
                    continue;
                }

                if (ctrlHeld)
                {
                    AssignGroup(i);
                }
                else
                {
                    RecallGroup(i);
                }
            }
        }

        private static KeyControl GetDigitKey(Keyboard keyboard, int digit)
        {
            switch (digit)
            {
                case 0: return keyboard.digit0Key;
                case 1: return keyboard.digit1Key;
                case 2: return keyboard.digit2Key;
                case 3: return keyboard.digit3Key;
                case 4: return keyboard.digit4Key;
                case 5: return keyboard.digit5Key;
                case 6: return keyboard.digit6Key;
                case 7: return keyboard.digit7Key;
                case 8: return keyboard.digit8Key;
                case 9: return keyboard.digit9Key;
                default: return null;
            }
        }

        private void AssignGroup(int slot)
        {
            if (SelectionManager.Instance == null || SelectionManager.Instance.CurrentSelection.Count == 0)
            {
                return;
            }

            _groups[slot] = new List<Selectable>(SelectionManager.Instance.CurrentSelection);
        }

        private void RecallGroup(int slot)
        {
            if (!_groups.TryGetValue(slot, out var members) || members.Count == 0)
            {
                return;
            }

            members.RemoveAll(m => m == null);
            if (members.Count == 0)
            {
                _groups.Remove(slot);
                return;
            }

            SelectionManager.Instance.SetSelection(members);

            bool isDoubleTap = slot == _lastRecalledGroup && Time.unscaledTime - _lastRecallTime <= doubleTapWindow;
            _lastRecalledGroup = slot;
            _lastRecallTime = Time.unscaledTime;

            if (isDoubleTap && cameraController != null)
            {
                Vector3 average = Vector3.zero;
                foreach (var m in members) average += m.Transform.position;
                average /= members.Count;
                cameraController.JumpTo(average);
            }
        }
    }
}
