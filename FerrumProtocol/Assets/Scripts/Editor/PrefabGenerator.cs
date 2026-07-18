using System.IO;
using FerrumProtocol.Buildings;
using FerrumProtocol.Combat;
using FerrumProtocol.Data;
using FerrumProtocol.FogOfWar;
using FerrumProtocol.Minimap;
using FerrumProtocol.Resources;
using FerrumProtocol.Selection;
using FerrumProtocol.Units;
using UnityEditor;
using UnityEngine;
using UnityEngine.AI;

namespace FerrumProtocol.EditorTools
{
    /// <summary>
    /// Generates primitive-mesh placeholder prefabs (per the project's "no final art yet"
    /// Phase 1 policy) fully wired with gameplay components, so the game is playable end to
    /// end before a single real model/texture exists. Used by <see cref="SceneBootstrapper"/>
    /// and callable standalone for any new unit/building data asset.
    /// </summary>
    public static class PrefabGenerator
    {
        private const string UnitsFolder = "Assets/Prefabs/Generated/Units";
        private const string BuildingsFolder = "Assets/Prefabs/Generated/Buildings";

        private static Material CreateColorMaterial(Color color)
        {
            Shader shader = Shader.Find("Universal Render Pipeline/Lit") ?? Shader.Find("Standard");
            var mat = new Material(shader);
            if (mat.HasProperty("_BaseColor"))
            {
                mat.SetColor("_BaseColor", color);
            }
            else if (mat.HasProperty("_Color"))
            {
                mat.SetColor("_Color", color);
            }
            return mat;
        }

        private static void EnsureFolder(string path)
        {
            if (!AssetDatabase.IsValidFolder(path))
            {
                Directory.CreateDirectory(path);
                AssetDatabase.ImportAsset(path);
            }
        }

        public static GameObject CreateUnitPrefab(string unitName, PrimitiveType shape, Color teamColor,
            UnitDataSO data, bool isHarvester, bool addWeapon)
        {
            EnsureFolder(UnitsFolder);

            var root = new GameObject(unitName);
            var visual = GameObject.CreatePrimitive(shape);
            visual.name = "Visual";
            visual.transform.SetParent(root.transform, false);
            visual.transform.localScale = shape == PrimitiveType.Capsule ? Vector3.one : new Vector3(1f, 0.5f, 1.5f);
            visual.transform.localPosition = shape == PrimitiveType.Capsule ? Vector3.zero : new Vector3(0f, 0.25f, 0f);
            Object.DestroyImmediate(visual.GetComponent<Collider>());
            visual.GetComponent<Renderer>().sharedMaterial = CreateColorMaterial(teamColor);

            var agent = root.AddComponent<NavMeshAgent>();
            agent.radius = 0.5f;
            agent.height = 2f;

            root.AddComponent<BoxCollider>().size = new Vector3(1f, 2f, 1f);

            var health = root.AddComponent<Health>();
            var selectable = root.AddComponent<Selectable>();
            root.AddComponent<UnitMotor>();
            root.AddComponent<StanceController>();
            var fogRevealer = root.AddComponent<FogRevealer>();
            var minimapIcon = root.AddComponent<MinimapIcon>();
            var unitController = root.AddComponent<UnitController>();
            AssignSerializedRef(fogRevealer, "selectable", selectable);
            AssignSerializedRef(minimapIcon, "selectable", selectable);

            WeaponSystem weapon = null;
            if (addWeapon)
            {
                weapon = root.AddComponent<WeaponSystem>();
            }

            if (isHarvester)
            {
                var harvester = root.AddComponent<HarvesterUnit>();
                var repair = root.AddComponent<Buildings.RepairSystem>();
                AssignSerializedRef(harvester, "selectable", selectable);
                AssignSerializedRef(repair, "selectable", selectable);
            }

            AssignSerializedRef(unitController, "unitData", data);
            AssignSerializedRef(unitController, "health", health);
            AssignSerializedRef(unitController, "selectable", selectable);
            if (weapon != null)
            {
                AssignSerializedRef(unitController, "weapon", weapon);
                AssignSerializedRef(root.GetComponent<StanceController>(), "weapon", weapon);
                AssignSerializedRef(root.GetComponent<StanceController>(), "selectable", selectable);
            }

            string path = $"{UnitsFolder}/{unitName}.prefab";
            var prefab = PrefabUtility.SaveAsPrefabAsset(root, path);
            Object.DestroyImmediate(root);
            return prefab;
        }

        public static GameObject CreateBuildingPrefab(string buildingName, Vector3 footprint, Color teamColor,
            BuildingDataSO data, bool isResourceDropoff, bool addProduction)
        {
            EnsureFolder(BuildingsFolder);

            var root = new GameObject(buildingName);
            root.AddComponent<BoxCollider>().size = footprint;

            var visual = GameObject.CreatePrimitive(PrimitiveType.Cube);
            visual.name = "Visual";
            visual.transform.SetParent(root.transform, false);
            visual.transform.localPosition = new Vector3(0f, footprint.y * 0.5f, 0f);
            visual.transform.localScale = footprint;
            Object.DestroyImmediate(visual.GetComponent<Collider>());
            visual.GetComponent<Renderer>().sharedMaterial = CreateColorMaterial(teamColor);

            var health = root.AddComponent<Health>();
            var selectable = root.AddComponent<Selectable>();
            var fogRevealer = root.AddComponent<FogRevealer>();
            var minimapIcon = root.AddComponent<MinimapIcon>();
            var buildingController = root.AddComponent<BuildingController>();
            AssignSerializedRef(fogRevealer, "selectable", selectable);
            AssignSerializedRef(minimapIcon, "selectable", selectable);

            if (addProduction)
            {
                var rally = new GameObject("RallyPoint").transform;
                rally.SetParent(root.transform, false);
                rally.localPosition = new Vector3(0f, 0f, footprint.z);

                var spawn = new GameObject("SpawnPoint").transform;
                spawn.SetParent(root.transform, false);
                spawn.localPosition = new Vector3(0f, 0f, footprint.z * 0.5f);

                var queue = root.AddComponent<ProductionQueue>();
                AssignSerializedRef(queue, "building", buildingController);
                AssignSerializedRef(queue, "rallyPoint", rally);
                AssignSerializedRef(queue, "spawnPoint", spawn);
            }

            AssignSerializedRef(buildingController, "buildingData", data);
            AssignSerializedRef(buildingController, "health", health);
            AssignSerializedRef(buildingController, "selectable", selectable);
            AssignSerializedRef(buildingController, "visualRoot", visual.transform);

            string path = $"{BuildingsFolder}/{buildingName}.prefab";
            var prefab = PrefabUtility.SaveAsPrefabAsset(root, path);
            Object.DestroyImmediate(root);
            return prefab;
        }

        /// <summary>Sets a private [SerializeField] via SerializedObject so generated prefabs come out fully wired without needing public setters everywhere.</summary>
        private static void AssignSerializedRef(Object target, string fieldName, Object value)
        {
            var so = new SerializedObject(target);
            var prop = so.FindProperty(fieldName);
            if (prop == null)
            {
                Debug.LogWarning($"PrefabGenerator: field '{fieldName}' not found on {target.GetType().Name}.");
                return;
            }
            prop.objectReferenceValue = value;
            so.ApplyModifiedPropertiesWithoutUndo();
        }
    }
}
