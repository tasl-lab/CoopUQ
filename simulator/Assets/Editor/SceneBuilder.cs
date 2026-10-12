using System.IO;
using Unity.AI.Navigation;
using UnityEditor;
using UnityEditor.SceneManagement;
using UnityEngine;
using UnityEngine.SceneManagement;

// Builds a scene from a SceneConfig: floor, boundary walls, obstacles, light, a top-down camera, the
// robot manager and the pedestrian spawner.
public static class SceneBuilder
{
    const int ObstacleLayer = 12;   // "Obstacle" in the tag manager; NPC is 3 and Robot is 6

    public static SceneConfig Load(string path)
    {
        return JsonUtility.FromJson<SceneConfig>(File.ReadAllText(path));
    }

    public static Scene Build(SceneConfig config, GameObject robotPrefab, GameObject pedestrianPrefab)
    {
        var scene = EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
        float sx = config.room_size[0], sz = config.room_size[1];

        var floor = GameObject.CreatePrimitive(PrimitiveType.Plane);
        floor.name = "Floor";
        floor.transform.localScale = new Vector3(sx / 10f, 1f, sz / 10f);
        floor.GetComponent<MeshRenderer>().sharedMaterial = SceneMaterials.Floor(sx, sz);
        var surface = floor.AddComponent<NavMeshSurface>();
        surface.collectObjects = CollectObjects.All;
        surface.useGeometry = UnityEngine.AI.NavMeshCollectGeometry.PhysicsColliders;

        const float t = 0.1f;   // wall thickness
        Wall("Wall_East", new Vector3(sx / 2, 0, 0), new Vector3(t, config.wall_height, sz));
        Wall("Wall_West", new Vector3(-sx / 2, 0, 0), new Vector3(t, config.wall_height, sz));
        Wall("Wall_North", new Vector3(0, 0, sz / 2), new Vector3(sx, config.wall_height, t));
        Wall("Wall_South", new Vector3(0, 0, -sz / 2), new Vector3(sx, config.wall_height, t));

        for (int i = 0; i < config.obstacles.Count; i++)
        {
            var o = config.obstacles[i];
            bool cylinder = o.type == "cylinder";
            var go = GameObject.CreatePrimitive(cylinder ? PrimitiveType.Cylinder : PrimitiveType.Cube);
            go.name = $"Obstacle_{i}";
            go.transform.position = new Vector3(o.position[0], o.height / 2, o.position[1]);
            go.transform.rotation = Quaternion.Euler(0, o.rotation, 0);
            go.transform.localScale = cylinder
                ? new Vector3(o.size[0], o.height / 2, o.size[0])     // Unity's cylinder is 2 units tall
                : new Vector3(o.size[0], o.height, o.size[1]);
            MarkObstacle(go, SceneMaterials.Obstacle());
        }

        var light = new GameObject("Light").AddComponent<Light>();
        light.type = LightType.Directional;
        light.intensity = 1.1f;
        light.shadows = LightShadows.Soft;
        light.shadowStrength = 0.6f;
        light.transform.rotation = Quaternion.Euler(55, -35, 0);
        RenderSettings.ambientMode = UnityEngine.Rendering.AmbientMode.Flat;
        RenderSettings.ambientLight = new Color(0.55f, 0.55f, 0.58f);

        // Only seen when the simulator is started with a window; evaluation runs without graphics.
        var camera = new GameObject("Camera").AddComponent<Camera>();
        camera.tag = "MainCamera";
        camera.orthographic = true;
        camera.orthographicSize = Mathf.Max(sx, sz) / 2 + 0.5f;
        camera.transform.SetPositionAndRotation(new Vector3(0, 20, 0), Quaternion.Euler(90, 0, 0));
        camera.clearFlags = CameraClearFlags.SolidColor;
        camera.backgroundColor = new Color(0.93f, 0.94f, 0.96f);

        var manager = new GameObject("RobotManager").AddComponent<RobotManager>();
        manager.robotPrefab = robotPrefab;
        manager.numberOfRobots = config.robots;
        manager.fixedSeed = config.fixed_seed;
        manager.continueAfterGoal = config.continue_after_goal;
        manager.pedestrianSpeedRange = new Vector2(config.pedestrian_speed[0], config.pedestrian_speed[1]);
        manager.spawnAreaSize = new Vector2(sx, sz);
        manager.minTargetDistance = config.goal_distance[0];
        manager.maxTargetDistance = config.goal_distance[1];
        manager.humanLayer = 1 << 3;
        manager.obstacleLayer = 1 << ObstacleLayer;
        manager.robotLayer = 1 << 6;

        var spawner = new GameObject("PedestrianSpawner").AddComponent<PedestrianSpawner>();
        spawner.pedestrianPrefab = pedestrianPrefab;
        spawner.numberOfPedestrians = config.pedestrians;
        spawner.spawnAreaSize = new Vector2(sx, sz);

        return scene;
    }

    static void Wall(string name, Vector3 position, Vector3 size)
    {
        var go = GameObject.CreatePrimitive(PrimitiveType.Cube);
        go.name = name;
        go.transform.position = position + new Vector3(0, size.y / 2, 0);
        go.transform.localScale = size;
        MarkObstacle(go, SceneMaterials.Wall());
    }

    static void MarkObstacle(GameObject go, Material material)
    {
        go.layer = ObstacleLayer;
        go.tag = "Obstacle";
        go.GetComponent<MeshRenderer>().sharedMaterial = material;
    }
}
