using System.Collections.Generic;
using Unity.AI.Navigation;
using UnityEngine;
using UnityEngine.AI;

// Owns the robots of one scene: builds the NavMesh, spawns the robots, hands out starts and goals,
// and keeps the global occupancy grid the robots crop their local map from.
public class RobotManager : MonoBehaviour
{
    public GameObject robotPrefab;
    public int numberOfRobots = 3;
    // Seed the random generator with a fixed value at startup, so that every launch replays the same
    // random sequence.
    public bool fixedSeed = false;
    public Vector2 pedestrianSpeedRange = new Vector2(0.5f, 1.5f);
    public static Vector2 PedestrianSpeedRange = new Vector2(0.5f, 1.5f);

    [Header("Starts and goals")]
    public Vector2 spawnAreaSize = new Vector2(12, 12);
    public Vector3 spawnAreaCenter = Vector3.zero;
    public float navMeshSampleRadius = 0.5f;
    public float minTargetDistance = 5f;
    public float maxTargetDistance = 10f;
    public float goalRadius = 1f;
    // false: every episode starts from a new random position. true: after reaching its goal a robot
    // keeps its position and only gets a new goal.
    public bool continueAfterGoal = false;

    [Header("Occupancy grid")]
    public float cellSize = 0.1f;
    public float pedestrianRadius = 0.3f;

    // The evaluation code tells the episode outcomes apart by these values.
    [Header("Rewards")]
    public float successReward = 10f;
    public float collisionPenaltyObstacle = -20f;
    public float collisionPenaltyPedestrian = -15f;
    public float collisionPenaltyRobot = -10f;
    public float distanceRewardFactor = 5f;
    public float constantPenalty = 0f;

    public LayerMask humanLayer;
    public LayerMask obstacleLayer;
    public LayerMask robotLayer;

    public List<RobotAgent> Robots { get; } = new List<RobotAgent>();
    GameObject[] pedestrians = new GameObject[0];

    Vector3 mapCorner;
    int mapResolutionX, mapResolutionZ;
    float[,] occupancyGrid;   // 0 free, 1 pedestrian, 2 static obstacle, 3 robot

    void Awake()
    {
        if (fixedSeed) Random.InitState(100);
        PedestrianSpeedRange = pedestrianSpeedRange;
        // Awake runs before any Start, so the NavMesh exists when pedestrians and robots are placed.
        foreach (var surface in FindObjectsOfType<NavMeshSurface>()) surface.BuildNavMesh();
    }

    void Start()
    {
        Time.captureFramerate = 60;
        // The spawner has already placed the pedestrians (it runs first), so the robot starts keep
        // clear of them and every observation holds all of them from the first step.
        pedestrians = GameObject.FindGameObjectsWithTag("NPC");
        SpawnRobots();
        BuildStaticOccupancy();
    }

    void Update()
    {
        UpdateDynamicOccupancy();
    }

    void SpawnRobots()
    {
        for (int i = 0; i < numberOfRobots; i++)
        {
            var robot = Instantiate(robotPrefab, AssignNewPosition(), Quaternion.identity);
            var agent = robot.GetComponent<RobotAgent>();
            agent.uid = i;
            agent.manager = this;
            agent.Target = GetRandomNavMeshPosition(spawnAreaCenter, new Vector2(1, 1), navMeshSampleRadius);
            Robots.Add(agent);
            Physics.SyncTransforms();   // the next robot's start has to keep clear of this one
        }
    }

    public List<(Vector3 position, Quaternion rotation, Vector3 target)> GetPedestriansInfo()
    {
        var info = new List<(Vector3, Quaternion, Vector3)>();
        foreach (var pedestrian in pedestrians)
        {
            var agent = pedestrian.GetComponent<NavMeshAgent>();
            info.Add((pedestrian.transform.position, pedestrian.transform.rotation,
                      agent != null && agent.enabled ? agent.destination : Vector3.zero));
        }
        return info;
    }

    void BuildStaticOccupancy()
    {
        mapCorner = spawnAreaCenter - new Vector3(spawnAreaSize.x, 0, spawnAreaSize.y) / 2;
        mapResolutionX = Mathf.RoundToInt(spawnAreaSize.x / cellSize);
        mapResolutionZ = Mathf.RoundToInt(spawnAreaSize.y / cellSize);
        occupancyGrid = new float[mapResolutionX, mapResolutionZ];
        var half = new Vector3(cellSize / 2, 0.5f, cellSize / 2);
        for (int x = 0; x < mapResolutionX; x++)
            for (int z = 0; z < mapResolutionZ; z++)
            {
                var centre = mapCorner + new Vector3((x + 0.5f) * cellSize, 1f, (z + 0.5f) * cellSize);
                if (Physics.CheckBox(centre, half, Quaternion.identity, obstacleLayer)) occupancyGrid[x, z] = 2f;
            }
    }

    // Marks a square around a pedestrian or robot. Static obstacles are never overwritten, and a
    // pedestrian cell is not overwritten by a robot.
    void Stamp(Vector3 position, int halfSide, float value)
    {
        Vector2Int index = WorldToMapIndex(position);
        for (int dx = -halfSide; dx <= halfSide; dx++)
            for (int dz = -halfSide; dz <= halfSide; dz++)
            {
                int x = index.x + dx, z = index.y + dz;
                if (x >= 0 && x < mapResolutionX && z >= 0 && z < mapResolutionZ && occupancyGrid[x, z] != 1f && occupancyGrid[x, z] != 2f)
                    occupancyGrid[x, z] = value;
            }
    }

    void UpdateDynamicOccupancy()
    {
        for (int x = 0; x < mapResolutionX; x++)
            for (int z = 0; z < mapResolutionZ; z++)
                if (occupancyGrid[x, z] == 1f || occupancyGrid[x, z] == 3f) occupancyGrid[x, z] = 0f;

        int halfSide = Mathf.CeilToInt(pedestrianRadius / cellSize);
        foreach (var pedestrian in pedestrians) Stamp(pedestrian.transform.position, halfSide, 1f);
        foreach (var robot in Robots) Stamp(robot.transform.position, 3, 3f);
    }

    // Cells outside the map are reported as 4.
    public float[,] GetRobotCentricOGM(Vector3 robotPosition, Quaternion robotRotation, int sizeX, int sizeZ)
    {
        float[,] local = new float[sizeX, sizeZ];
        int centreX = sizeX / 2, centreZ = sizeZ / 2;
        for (int lx = -centreX; lx < centreX; lx++)
            for (int lz = -centreZ; lz < centreZ; lz++)
            {
                Vector2Int index = WorldToMapIndex(robotPosition + robotRotation * new Vector3(lx * cellSize, 0, lz * cellSize));
                bool inside = index.x >= 0 && index.x < mapResolutionX && index.y >= 0 && index.y < mapResolutionZ;
                local[lx + centreX, lz + centreZ] = inside ? occupancyGrid[index.x, index.y] : 4f;
            }
        return local;
    }

    Vector2Int WorldToMapIndex(Vector3 world)
    {
        return new Vector2Int(Mathf.FloorToInt((world.x - mapCorner.x) / cellSize),
                              Mathf.FloorToInt((world.z - mapCorner.z) / cellSize));
    }

    public Vector3 AssignNewPosition()
    {
        return GetRandomNavMeshPosition(spawnAreaCenter, spawnAreaSize, navMeshSampleRadius);
    }

    // The robot has already been moved to its new start when this is called.
    public Vector3 AssignNewTarget(int uid)
    {
        Vector3 start = Robots[uid].transform.position;
        for (int attempt = 0; attempt < 10; attempt++)
        {
            if (!NavMesh.SamplePosition(RandomPoint(spawnAreaCenter, spawnAreaSize), out NavMeshHit hit, navMeshSampleRadius, NavMesh.AllAreas)) continue;
            float distance = (hit.position - start).magnitude;
            if (distance >= minTargetDistance && distance <= maxTargetDistance) return hit.position;
        }
        return spawnAreaCenter;
    }

    static Vector3 RandomPoint(Vector3 center, Vector2 size)
    {
        return new Vector3(Random.Range(-size.x / 2, size.x / 2), 0f, Random.Range(-size.y / 2, size.y / 2)) + center;
    }

    // A free NavMesh point with no pedestrian or robot within 1.5 m, lifted so that the robot drops onto the floor.
    Vector3 GetRandomNavMeshPosition(Vector3 center, Vector2 size, float radius)
    {
        var lift = new Vector3(0, 0.3f, 0);
        for (int attempt = 0; attempt < 50; attempt++)
        {
            if (!NavMesh.SamplePosition(RandomPoint(center, size), out NavMeshHit hit, radius, NavMesh.AllAreas)) continue;
            if (Physics.OverlapSphere(hit.position, 1.5f, humanLayer | robotLayer).Length == 0) return hit.position + lift;
        }
        return center + lift;
    }
}
