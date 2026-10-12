using Unity.MLAgents;
using Unity.MLAgents.Actuators;
using Unity.MLAgents.Sensors;
using UnityEngine;

// One holonomic robot. The action is its planar velocity; the observation is
//   local occupancy crop (64 x 64) | LiDAR ranges (1080) | own position (3) | own goal (3)
//   | position and goal of every other robot (6 each) | position, rotation and goal of every pedestrian (9 each).
public class RobotAgent : Agent
{
    public Vector3 Target;
    public RobotManager manager;
    public float maxSpeed = 1.0f;
    public int uid;
    public Vector2Int observationCropSize = new Vector2Int(64, 64);

    public int lidarRayCount = 1080;
    public float lidarMaxRange = 7f;
    public LayerMask lidarLayerMask;

    Rigidbody rBody;
    float previousDistanceToTarget;
    bool hitPedestrian, hitObstacle, hitRobot, reachedGoal;

    public override void Initialize()
    {
        rBody = GetComponent<Rigidbody>();
    }

    float[] LidarScan()
    {
        float[] ranges = new float[lidarRayCount];
        float step = 360f / lidarRayCount;
        for (int i = 0; i < lidarRayCount; i++)
        {
            Vector3 direction = Quaternion.Euler(0, step * i, 0) * Vector3.forward;
            // a ray that hits nothing within range is reported as 0
            if (Physics.Raycast(transform.position, direction, out RaycastHit hit, lidarMaxRange, lidarLayerMask))
                ranges[i] = hit.distance;
        }
        return ranges;
    }

    public override void OnEpisodeBegin()
    {
        rBody.velocity = Vector3.zero;
        rBody.angularVelocity = Vector3.zero;
        transform.rotation = Quaternion.identity;
        // With continueAfterGoal a robot that reached its goal drives on from there to the next goal.
        if (!(manager.continueAfterGoal && reachedGoal)) transform.position = manager.AssignNewPosition();
        reachedGoal = false;
        Target = manager.AssignNewTarget(uid);
        previousDistanceToTarget = Vector3.Distance(transform.position, Target);
    }

    public override void CollectObservations(VectorSensor sensor)
    {
        float[,] crop = manager.GetRobotCentricOGM(transform.position, transform.rotation, observationCropSize.x, observationCropSize.y);
        for (int i = 0; i < observationCropSize.x; i++)
            for (int j = 0; j < observationCropSize.y; j++)
                sensor.AddObservation(crop[i, j]);

        foreach (float range in LidarScan()) sensor.AddObservation(range);

        sensor.AddObservation(transform.position);
        sensor.AddObservation(Target);

        foreach (var robot in manager.Robots)
        {
            if (robot.uid == uid) continue;
            sensor.AddObservation(robot.transform.position);
            sensor.AddObservation(robot.Target);
        }
        foreach (var pedestrian in manager.GetPedestriansInfo())
        {
            sensor.AddObservation(pedestrian.position);
            sensor.AddObservation(pedestrian.rotation.eulerAngles);
            sensor.AddObservation(pedestrian.target);
        }
    }

    (float reward, bool done) StepOutcome()
    {
        if (StepCount >= MaxStep) return (0f, true);
        if (Vector3.Distance(transform.position, Target) < manager.goalRadius) { reachedGoal = true; return (manager.successReward, true); }
        if (hitObstacle) { hitObstacle = false; return (manager.collisionPenaltyObstacle, true); }
        if (hitPedestrian) { hitPedestrian = false; return (manager.collisionPenaltyPedestrian, true); }
        if (hitRobot) { hitRobot = false; return (manager.collisionPenaltyRobot, true); }

        float distance = Vector3.Distance(transform.position, Target);
        float reward = manager.distanceRewardFactor * (previousDistanceToTarget - distance) + manager.constantPenalty;
        previousDistanceToTarget = distance;
        return (reward, false);
    }

    public override void OnActionReceived(ActionBuffers actionBuffers)
    {
        float vx = Mathf.Clamp(actionBuffers.ContinuousActions[0], -maxSpeed, maxSpeed);
        float vz = Mathf.Clamp(actionBuffers.ContinuousActions[1], -maxSpeed, maxSpeed);
        // action[0] moves along z and action[1] along x (the axis order the policy was trained with)
        transform.position += new Vector3(vz, 0, vx) * Time.deltaTime;

        var (reward, done) = StepOutcome();
        SetReward(reward);
        if (done) EndEpisode();
    }

    void OnCollisionEnter(Collision collision)
    {
        if (collision.collider.CompareTag("Obstacle")) hitObstacle = true;
        else if (collision.collider.CompareTag("NPC_legs")) hitPedestrian = true;
        else if (collision.collider.CompareTag("Robot")) hitRobot = true;
    }

    public override void Heuristic(in ActionBuffers actionsOut)
    {
        var actions = actionsOut.ContinuousActions;
        actions[0] = Input.GetAxis("Vertical");
        actions[1] = Input.GetAxis("Horizontal");
    }
}
