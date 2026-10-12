using UnityEngine;
using UnityEngine.AI;

// Places the pedestrians at random NavMesh points. Runs before the robots are spawned, so that
// robot starts keep clear of them.
[DefaultExecutionOrder(-10000)]
public class PedestrianSpawner : MonoBehaviour
{
    public GameObject pedestrianPrefab;
    public int numberOfPedestrians = 10;
    public Vector2 spawnAreaSize = new Vector2(12, 12);
    public Vector3 spawnAreaCenter = Vector3.zero;

    void Start()
    {
        for (int i = 0; i < numberOfPedestrians; i++) Spawn();
    }

    void Spawn()
    {
        for (int attempt = 0; attempt < 10; attempt++)
        {
            Vector3 candidate = new Vector3(
                Random.Range(-spawnAreaSize.x / 2, spawnAreaSize.x / 2), 0,
                Random.Range(-spawnAreaSize.y / 2, spawnAreaSize.y / 2)) + spawnAreaCenter;
            if (!NavMesh.SamplePosition(candidate, out NavMeshHit hit, 1.0f, NavMesh.AllAreas)) continue;

            GameObject pedestrian = Instantiate(pedestrianPrefab, hit.position + new Vector3(0, 0.9f, 0), Quaternion.identity);
            NavMeshAgent agent = pedestrian.GetComponent<NavMeshAgent>();
            if (agent == null) agent = pedestrian.AddComponent<NavMeshAgent>();
            agent.agentTypeID = NavMesh.GetSettingsByIndex(0).agentTypeID;
            agent.obstacleAvoidanceType = ObstacleAvoidanceType.HighQualityObstacleAvoidance;
            pedestrian.tag = "NPC";
            pedestrian.AddComponent<PedestrianWander>();
            return;
        }
        Debug.LogWarning("No free NavMesh point for a pedestrian after 10 attempts.");
    }
}
