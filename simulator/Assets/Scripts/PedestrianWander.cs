using UnityEngine;
using UnityEngine.AI;

// Random walk: the pedestrian keeps picking a new NavMesh destination near its current position.
public class PedestrianWander : MonoBehaviour
{
    public float wanderRadius = 3.0f;
    NavMeshAgent agent;

    void Start()
    {
        agent = GetComponent<NavMeshAgent>();
        agent.speed = Random.Range(RobotManager.PedestrianSpeedRange.x, RobotManager.PedestrianSpeedRange.y);
        agent.angularSpeed = 100f;
        agent.acceleration = 1.0f;
        agent.stoppingDistance = 0.5f;
    }

    void Update()
    {
        if (agent.remainingDistance <= agent.stoppingDistance) agent.SetDestination(RandomNavMeshPoint());
    }

    Vector3 RandomNavMeshPoint()
    {
        for (int attempt = 0; attempt < 50; attempt++)
        {
            Vector3 point = transform.position + Random.insideUnitSphere * (wanderRadius * 0.9f);
            point.y = 0f;
            if (NavMesh.SamplePosition(point, out NavMeshHit hit, wanderRadius * 0.1f, NavMesh.AllAreas)) return hit.position;
        }
        return transform.position;
    }
}
