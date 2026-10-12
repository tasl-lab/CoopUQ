# Simulator

A Unity project that generates simulation scenes from a text description, so that the layout and
the numbers of robots and pedestrians can be changed without opening the editor.

## Requirements

- [Unity **2023.2.20f1**](https://unity.com/releases/editor/whats-new/2023.2.20f1) with
  [Linux Build Support](https://docs.unity.com/en-us/hub/add-modules), installed through
  [Unity Hub](https://docs.unity.com/en-us/hub/install-hub) (the Hub has to be
  signed in, otherwise the command-line build stops with a license error).
- The first build downloads the packages in `Packages/manifest.json` and takes a few minutes.

## Build a scene

```bash
cd simulator
./build.sh configs/id_1.json                              # -> Builds/id_1/id_1.x86_64
./build.sh configs/id_1.json id_1_n5 -robots 5 -pedestrians 12
```

`UNITY=/path/to/Editor/Unity ./build.sh ...` selects another editor binary. The options override
the values of the config file:

| Option | Meaning |
|---|---|
| `-robots N` | number of robots |
| `-pedestrians N` | number of pedestrians |
| `-pedestrianSpeed MIN,MAX` | each pedestrian draws its speed from this range (m/s) |
| `-goalDistance MIN,MAX` | distance from a robot's start to its goal (m) |
| `-maxSteps N` | episode length in control steps of 0.2 s |
| `-fixedSeed` | same random sequence at every launch |
| `-continueAfterGoal` | a robot that reaches its goal drives on from there to a new goal |

`configs/` includes three training layouts (`id_1`-`id_3`), three unseen layouts (`ood_1`-`ood_3`),
and `rooms`, a layout with four furnished rooms off a central corridor (15 m x 15 m,
doors 1.5-2.0 m).

## Run the policy in a build

```bash
cd ..
ENV=simulator/Builds/id_1/id_1.x86_64 bash onpolicy/scripts/eval.sh               # success and collision rates
bash onpolicy/scripts/visualize.sh simulator/Builds/id_1/id_1.x86_64 id_1.mp4     # top-down video
```

`build.sh` writes the scene description it built next to the executable (`Builds/id_1/id_1.json`);
`visualize.sh` draws the obstacles from it. Add `NUM_ROBOTS=5` for a build with five robots.

## Scene description

```json
{
  "room_size": [12, 12],
  "wall_height": 2.0,
  "obstacles": [
    {"type": "box", "position": [-2.35, 3.72], "size": [2.0, 0.3], "rotation": 0, "height": 2.0},
    {"type": "cylinder", "position": [0, 2.3], "size": [0.6]}
  ],
  "robots": 3,
  "pedestrians": 10,
  "pedestrian_speed": [0.5, 1.5],
  "goal_distance": [5, 10],
  "max_steps": 200,
  "fixed_seed": false,
  "continue_after_goal": false
}
```

The room is centred at the origin; `position` is the (x, z) centre of an obstacle and `size` its
extent along x and z (the diameter for a cylinder). Robots and pedestrians start at random free
positions; pedestrians walk to random nearby destinations. Obstacles and walls should be at least
1.2 m high to be seen by the LiDAR, which is mounted 1 m above the floor. With
`continue_after_goal` a robot that reaches its goal drives on from there to a new goal instead of
being moved to a new start.

## Layout

```
Assets/Scripts/     RobotAgent, RobotManager, PedestrianSpawner, PedestrianWander   simulation
Assets/Editor/      SceneConfig, SceneBuilder, SceneMaterials, BuildCLI             scene generation and build
Assets/Prefabs/     Robot, Pedestrian                                               simulated bodies
configs/            scene descriptions
```
