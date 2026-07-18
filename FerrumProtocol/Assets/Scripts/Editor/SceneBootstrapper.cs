using FerrumProtocol.AI;
using FerrumProtocol.Buildings;
using FerrumProtocol.CameraSystem;
using FerrumProtocol.Combat;
using FerrumProtocol.Core;
using FerrumProtocol.Data;
using FerrumProtocol.FogOfWar;
using FerrumProtocol.InputSystem;
using FerrumProtocol.Minimap;
using FerrumProtocol.Resources;
using FerrumProtocol.Selection;
using Unity.AI.Navigation;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.AI;
using UnityEngine.InputSystem;
using UnityEngine.SceneManagement;
using UnityEngine.UI;

namespace FerrumProtocol.EditorTools
{
    /// <summary>
    /// One-click prototype assembly: builds a complete playable scene (terrain, baked NavMesh,
    /// RTS camera, input, selection, fog of war, minimap, two factions' worth of placeholder
    /// units/buildings, a resource economy, and one AI opponent) entirely from code. This is
    /// the safe alternative to hand-authoring .unity/.prefab YAML (see ARCHITECTURE.md).
    /// Menu: Ferrum Protocol > Build Prototype Scene.
    /// </summary>
    public static class SceneBootstrapper
    {
        private const string SceneOutputPath = "Assets/Scenes/Prototype.unity";
        private const string DataFolder = "Assets/Data/Generated";

        [MenuItem("Ferrum Protocol/Build Prototype Scene")]
        public static void BuildPrototypeScene()
        {
            var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);

            BuildGround();
            BuildLighting();
            var controls = LoadControls();
            var cameraGo = BuildCamera(controls);
            BuildSystems(cameraGo);
            BuildUI(cameraGo);

            EnsureFolder(DataFolder);
            var factionAData = BuildFactionAssets("AuroraConcord", new Color(0.25f, 0.75f, 1f));
            var factionBData = BuildFactionAssets("KesslerDominion", new Color(0.85f, 0.4f, 0.15f));

            PlacePlayerBase(factionAData, ownerId: 0, origin: new Vector3(-30f, 0f, -30f));
            PlacePlayerBase(factionBData, ownerId: 1, origin: new Vector3(30f, 0f, 30f), spawnAiController: true);

            PlaceResourceNodes();
            BakeNavMesh();

            EnsureFolder("Assets/Scenes");
            EditorSceneManager.SaveScene(scene, SceneOutputPath);
            Debug.Log($"Ferrum Protocol: prototype scene built and saved to {SceneOutputPath}. Press Play to test it.");
        }

        private static void EnsureFolder(string path)
        {
            if (!AssetDatabase.IsValidFolder(path))
            {
                System.IO.Directory.CreateDirectory(path);
                AssetDatabase.ImportAsset(path);
            }
        }

        private static void BuildGround()
        {
            var ground = GameObject.CreatePrimitive(PrimitiveType.Plane);
            ground.name = "Ground";
            ground.transform.localScale = new Vector3(20f, 1f, 20f); // a Unity Plane is 10x10 units at scale 1
            var groundSurface = ground.AddComponent<NavMeshSurface>();
            // "Children" (this object + descendants) rather than "All" - the ground plane has no
            // children, so this bakes purely from the flat terrain mesh and is unaffected by
            // whatever units/buildings get placed on top of it later in this same bootstrap pass.
            groundSurface.collectObjects = CollectObjects.Children;
        }

        private static void BuildLighting()
        {
            var lightGo = new GameObject("Directional Light");
            var light = lightGo.AddComponent<Light>();
            light.type = LightType.Directional;
            light.intensity = 1.1f;
            lightGo.transform.rotation = Quaternion.Euler(50f, -30f, 0f);
        }

        private static InputActionAsset LoadControls()
        {
            return AssetDatabase.LoadAssetAtPath<InputActionAsset>(
                "Assets/Scripts/Runtime/InputSystem/RTSControls.inputactions");
        }

        private static GameObject BuildCamera(InputActionAsset controls)
        {
            var cameraGo = new GameObject("RTS Camera");
            var cam = cameraGo.AddComponent<Camera>();
            cam.tag = "MainCamera";
            cameraGo.AddComponent<AudioListener>();

            var rtsCam = cameraGo.AddComponent<RTSCameraController>();
            rtsCam.SetBounds(CameraBounds.FromCenterExtents(Vector3.zero, new Vector2(90f, 90f)));

            var inputHandler = cameraGo.AddComponent<PlayerInputHandler>();
            SetPrivateField(inputHandler, "controls", controls);
            SetPrivateField(inputHandler, "cameraController", rtsCam);
            SetPrivateField(inputHandler, "targetCamera", cam);

            cameraGo.transform.position = new Vector3(0f, 30f, -20f);
            return cameraGo;
        }

        private static void BuildSystems(GameObject cameraGo)
        {
            var systems = new GameObject("Systems");
            var gameManager = systems.AddComponent<GameManager>();

            var selectionGo = new GameObject("SelectionManager");
            selectionGo.transform.SetParent(systems.transform);
            var selectionManager = selectionGo.AddComponent<SelectionManager>();
            selectionGo.AddComponent<ControlGroupManager>();
            SetPrivateField(selectionGo.GetComponent<ControlGroupManager>(), "cameraController", cameraGo.GetComponent<RTSCameraController>());

            var inputHandler = cameraGo.GetComponent<PlayerInputHandler>();
            SetPrivateField(inputHandler, "selectionManager", selectionManager);

            var poolGo = new GameObject("ObjectPoolManager");
            poolGo.transform.SetParent(systems.transform);
            poolGo.AddComponent<ObjectPoolManager>();

            var fogGo = new GameObject("FogOfWarManager");
            fogGo.transform.SetParent(systems.transform);
            fogGo.AddComponent<FogOfWarManager>();

            var saveGo = new GameObject("SaveManager");
            saveGo.transform.SetParent(systems.transform);
            saveGo.AddComponent<Save.SaveManager>();
            saveGo.AddComponent<Save.StatisticsTracker>();
        }

        private static void BuildUI(GameObject cameraGo)
        {
            var canvasGo = new GameObject("HUD Canvas", typeof(Canvas), typeof(CanvasScaler), typeof(GraphicRaycaster));
            var canvas = canvasGo.GetComponent<Canvas>();
            canvas.renderMode = RenderMode.ScreenSpaceOverlay;

            new GameObject("EventSystem",
                typeof(UnityEngine.EventSystems.EventSystem),
                typeof(UnityEngine.InputSystem.UI.InputSystemUIInputModule));

            // Drag-select rectangle overlay.
            var boxGo = new GameObject("SelectionBox", typeof(RectTransform), typeof(Image));
            boxGo.transform.SetParent(canvasGo.transform, false);
            var boxImage = boxGo.GetComponent<Image>();
            boxImage.color = new Color(0.4f, 0.9f, 1f, 0.25f);
            var boxUi = boxGo.AddComponent<SelectionBoxUI>();
            SetPrivateField(boxUi, "selectionManager", Object.FindFirstObjectByType<SelectionManager>());
            SetPrivateField(boxUi, "image", boxImage);

            // Minimap panel (bottom-left), fed by a dedicated top-down camera.
            var minimapCamGo = new GameObject("Minimap Camera");
            var minimapCam = minimapCamGo.AddComponent<Camera>();
            minimapCam.orthographic = true;
            minimapCam.orthographicSize = 100f;
            minimapCam.transform.position = new Vector3(0f, 100f, 0f);
            minimapCam.transform.rotation = Quaternion.Euler(90f, 0f, 0f);
            var renderTexture = new RenderTexture(256, 256, 16);
            minimapCam.targetTexture = renderTexture;

            var minimapPanelGo = new GameObject("MinimapPanel", typeof(RectTransform), typeof(RawImage));
            minimapPanelGo.transform.SetParent(canvasGo.transform, false);
            var minimapRect = minimapPanelGo.GetComponent<RectTransform>();
            minimapRect.anchorMin = Vector2.zero;
            minimapRect.anchorMax = Vector2.zero;
            minimapRect.pivot = Vector2.zero;
            minimapRect.anchoredPosition = new Vector2(16f, 16f);
            minimapRect.sizeDelta = new Vector2(220f, 220f);
            minimapPanelGo.GetComponent<RawImage>().texture = renderTexture;

            var iconsContainerGo = new GameObject("Icons", typeof(RectTransform));
            iconsContainerGo.transform.SetParent(minimapPanelGo.transform, false);
            var iconsRect = iconsContainerGo.GetComponent<RectTransform>();
            iconsRect.anchorMin = Vector2.zero;
            iconsRect.anchorMax = Vector2.one;
            iconsRect.offsetMin = Vector2.zero;
            iconsRect.offsetMax = Vector2.zero;

            var minimapController = minimapPanelGo.AddComponent<MinimapController>();
            SetPrivateField(minimapController, "minimapCamera", minimapCam);
            SetPrivateField(minimapController, "minimapImage", minimapPanelGo.GetComponent<RawImage>());
            SetPrivateField(minimapController, "iconsContainer", iconsRect);
            SetPrivateField(minimapController, "mainCameraController", cameraGo.GetComponent<RTSCameraController>());
        }

        private struct FactionAssets
        {
            public UnitDataSO Harvester;
            public UnitDataSO Infantry;
            public BuildingDataSO CommandCenter;
            public GameObject HarvesterPrefab;
            public GameObject InfantryPrefab;
            public GameObject CommandCenterPrefab;
        }

        private static FactionAssets BuildFactionAssets(string factionKey, Color color)
        {
            var weapon = ScriptableObject.CreateInstance<WeaponDataSO>();
            weapon.weaponName = $"{factionKey} Rifle";
            weapon.damage = 8f;
            weapon.damageType = DamageType.Kinetic;
            weapon.rateOfFire = 1.5f;
            weapon.range = 10f;
            weapon.isHitscan = true;
            AssetDatabase.CreateAsset(weapon, $"{DataFolder}/{factionKey}_Weapon.asset");

            var harvesterData = ScriptableObject.CreateInstance<UnitDataSO>();
            harvesterData.unitName = $"{factionKey} Harvester";
            harvesterData.category = UnitCategory.Vehicle;
            harvesterData.isHarvester = true;
            harvesterData.maxHealth = 80f;
            harvesterData.moveSpeed = 4f;
            harvesterData.ferriteCost = 150;
            harvesterData.buildTimeSeconds = 10f;
            AssetDatabase.CreateAsset(harvesterData, $"{DataFolder}/{factionKey}_Harvester.asset");

            var infantryData = ScriptableObject.CreateInstance<UnitDataSO>();
            infantryData.unitName = $"{factionKey} Rifle Infantry";
            infantryData.category = UnitCategory.Infantry;
            infantryData.maxHealth = 60f;
            infantryData.moveSpeed = 3.5f;
            infantryData.ferriteCost = 100;
            infantryData.buildTimeSeconds = 6f;
            infantryData.weapon = weapon;
            AssetDatabase.CreateAsset(infantryData, $"{DataFolder}/{factionKey}_Infantry.asset");

            var commandCenter = ScriptableObject.CreateInstance<BuildingDataSO>();
            commandCenter.buildingName = $"{factionKey} Command Center";
            commandCenter.maxHealth = 1000f;
            commandCenter.ferriteCost = 0; // starting building, not purchasable
            commandCenter.buildTimeSeconds = 1f;
            commandCenter.isResourceDropoff = true;
            commandCenter.powerProduced = 50;
            commandCenter.producibleUnits = new[] { harvesterData, infantryData };
            AssetDatabase.CreateAsset(commandCenter, $"{DataFolder}/{factionKey}_CommandCenter.asset");

            var harvesterPrefab = PrefabGenerator.CreateUnitPrefab($"{factionKey}_Harvester", PrimitiveType.Capsule, color, harvesterData, isHarvester: true, addWeapon: false);
            var infantryPrefab = PrefabGenerator.CreateUnitPrefab($"{factionKey}_Infantry", PrimitiveType.Capsule, color, infantryData, isHarvester: false, addWeapon: true);
            var commandCenterPrefab = PrefabGenerator.CreateBuildingPrefab($"{factionKey}_CommandCenter", new Vector3(8f, 4f, 8f), color, commandCenter, isResourceDropoff: true, addProduction: true);

            harvesterData.prefab = harvesterPrefab;
            infantryData.prefab = infantryPrefab;
            commandCenter.prefab = commandCenterPrefab;
            EditorUtility.SetDirty(harvesterData);
            EditorUtility.SetDirty(infantryData);
            EditorUtility.SetDirty(commandCenter);

            return new FactionAssets
            {
                Harvester = harvesterData,
                Infantry = infantryData,
                CommandCenter = commandCenter,
                HarvesterPrefab = harvesterPrefab,
                InfantryPrefab = infantryPrefab,
                CommandCenterPrefab = commandCenterPrefab
            };
        }

        private static void PlacePlayerBase(FactionAssets assets, int ownerId, Vector3 origin, bool spawnAiController = false)
        {
            var economyGo = new GameObject($"PlayerEconomy_{ownerId}");
            var economy = economyGo.AddComponent<PlayerEconomy>();
            SetPrivateField(economy, "playerId", ownerId);

            var commandCenter = (GameObject)PrefabUtility.InstantiatePrefab(assets.CommandCenterPrefab);
            commandCenter.transform.position = origin;
            SetOwner(commandCenter, ownerId);

            for (int i = 0; i < 3; i++)
            {
                var harvester = (GameObject)PrefabUtility.InstantiatePrefab(assets.HarvesterPrefab);
                harvester.transform.position = origin + new Vector3(6f + i * 2f, 0f, -6f);
                SetOwner(harvester, ownerId);
            }

            for (int i = 0; i < 4; i++)
            {
                var infantry = (GameObject)PrefabUtility.InstantiatePrefab(assets.InfantryPrefab);
                infantry.transform.position = origin + new Vector3(-6f + i * 2f, 0f, 6f);
                SetOwner(infantry, ownerId);
            }

            if (spawnAiController)
            {
                EnsureFolder(DataFolder);
                var difficulty = ScriptableObject.CreateInstance<AIDifficultyDataSO>();
                difficulty.difficultyName = "Normal";
                AssetDatabase.CreateAsset(difficulty, $"{DataFolder}/Difficulty_Normal.asset");

                var aiGo = new GameObject($"AIController_{ownerId}");
                var ai = aiGo.AddComponent<AIController>();
                SetPrivateField(ai, "playerId", ownerId);
                SetPrivateField(ai, "difficulty", difficulty);
            }
        }

        private static void SetOwner(GameObject go, int ownerId)
        {
            if (go.TryGetComponent<Selectable>(out var selectable))
            {
                selectable.SetOwner(ownerId);
            }
        }

        private static void PlaceResourceNodes()
        {
            Vector3[] positions =
            {
                new Vector3(-15f, 0f, -15f),
                new Vector3(15f, 0f, 15f),
                new Vector3(0f, 0f, 0f), // contested strategic node between both bases
            };

            for (int i = 0; i < positions.Length; i++)
            {
                var nodeGo = GameObject.CreatePrimitive(PrimitiveType.Cylinder);
                nodeGo.name = i == positions.Length - 1 ? "CommandCells_Node" : "Ferrite_Node";
                nodeGo.transform.position = positions[i];
                nodeGo.transform.localScale = new Vector3(2f, 0.5f, 2f);

                var node = nodeGo.AddComponent<ResourceNode>();
                SetPrivateField(node, "resourceType", i == positions.Length - 1 ? ResourceType.CommandCells : ResourceType.Ferrite);
                SetPrivateField(node, "totalAmount", i == positions.Length - 1 ? 1000 : 5000);
            }
        }

        private static void BakeNavMesh()
        {
            foreach (var surface in Object.FindObjectsByType<NavMeshSurface>(FindObjectsSortMode.None))
            {
                surface.BuildNavMesh();
            }
        }

        private static void SetPrivateField(object target, string fieldName, object value)
        {
            var type = target.GetType();
            var field = type.GetField(fieldName, System.Reflection.BindingFlags.NonPublic | System.Reflection.BindingFlags.Public | System.Reflection.BindingFlags.Instance);
            if (field == null)
            {
                Debug.LogWarning($"SceneBootstrapper: field '{fieldName}' not found on {type.Name}.");
                return;
            }
            field.SetValue(target, value);
        }
    }
}
