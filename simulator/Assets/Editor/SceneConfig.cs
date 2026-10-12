using System;
using System.Collections.Generic;

// A scene description, read from a JSON file in configs/. Lengths are in metres, angles in degrees;
// the room is centred at the origin with x to the right and z up when seen from above.
[Serializable]
public class SceneConfig
{
    public float[] room_size = { 12, 12 };          // extent along x and z
    public float wall_height = 2f;
    public List<Obstacle> obstacles = new List<Obstacle>();

    public int robots = 3;
    public int pedestrians = 10;
    public float[] pedestrian_speed = { 0.5f, 1.5f };   // each pedestrian draws its speed from this range (m/s)
    public float[] goal_distance = { 5, 10 };           // distance from a robot's start to its goal
    public int max_steps = 200;                         // episode length; one step is 0.2 s
    public bool fixed_seed = false;                     // same random sequence at every launch
    public bool continue_after_goal = false;            // a robot that reached its goal drives on from there

    [Serializable]
    public class Obstacle
    {
        public string type = "box";                 // "box" or "cylinder"
        public float[] position = { 0, 0 };         // x, z of the centre
        public float[] size = { 1, 1 };             // box: extent along x and z; cylinder: diameter
        public float rotation = 0f;                 // box: rotation about the vertical axis
        public float height = 2f;
    }
}
