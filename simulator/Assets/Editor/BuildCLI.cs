using System;
using System.IO;
using System.Linq;
using UnityEditor;
using UnityEditor.Build.Reporting;
using UnityEditor.SceneManagement;
using UnityEngine;

// Headless player build of one generated scene, driven by build.sh:
//   Unity -batchmode -quit -projectPath . -executeMethod BuildCLI.Build
//         -config configs/id_1.json -out Builds/id_1/id_1.x86_64
//         [-robots 5] [-pedestrians 12] [-pedestrianSpeed 0.8,2] [-goalDistance 5,8]
//         [-maxSteps 200] [-fixedSeed] [-continueAfterGoal]
// The options override the values of the config file. The scene description actually built is
// written next to the executable (<name>.json), for onpolicy/scripts/render_2d.py.
public static class BuildCLI
{
    public const string TmpDir = "Assets/_BuildTmp";
    const string RobotPrefab = "Assets/Prefabs/Robot.prefab";
    const string PedestrianPrefab = "Assets/Prefabs/Pedestrian.prefab";

    static string Arg(string name)
    {
        var a = Environment.GetCommandLineArgs();
        int i = Array.IndexOf(a, name);
        return (i >= 0 && i + 1 < a.Length) ? a[i + 1] : null;
    }

    static bool Flag(string name) => Environment.GetCommandLineArgs().Contains(name);

    static float[] Pair(string s) => s.Split(',').Select(float.Parse).ToArray();

    public static void Build()
    {
        try
        {
            Run();
        }
        catch (Exception e)
        {
            Debug.LogError("[BuildCLI] FAILED: " + e);
            EditorApplication.Exit(1);
        }
    }

    static void Run()
    {
        string configPath = Arg("-config") ?? throw new ArgumentException("-config is required");
        string outPath = Arg("-out") ?? throw new ArgumentException("-out is required");

        var config = SceneBuilder.Load(configPath);
        if (Arg("-robots") != null) config.robots = int.Parse(Arg("-robots"));
        if (Arg("-pedestrians") != null) config.pedestrians = int.Parse(Arg("-pedestrians"));
        if (Arg("-pedestrianSpeed") != null) config.pedestrian_speed = Pair(Arg("-pedestrianSpeed"));
        if (Arg("-goalDistance") != null) config.goal_distance = Pair(Arg("-goalDistance"));
        if (Arg("-maxSteps") != null) config.max_steps = int.Parse(Arg("-maxSteps"));
        if (Flag("-fixedSeed")) config.fixed_seed = true;
        if (Flag("-continueAfterGoal")) config.continue_after_goal = true;

        if (AssetDatabase.IsValidFolder(TmpDir)) AssetDatabase.DeleteAsset(TmpDir);
        AssetDatabase.CreateFolder("Assets", "_BuildTmp");
        try
        {
            // The observation width depends on the numbers of robots and pedestrians; it is set on a
            // copy of the robot prefab so that the prefab in the repository stays untouched.
            string robotCopy = $"{TmpDir}/Robot.prefab";
            AssetDatabase.CopyAsset(RobotPrefab, robotCopy);
            var robot = AssetDatabase.LoadAssetAtPath<GameObject>(robotCopy);
            var agent = robot.GetComponent<RobotAgent>();
            int obsSize = agent.observationCropSize.x * agent.observationCropSize.y + agent.lidarRayCount
                          + 6 + (config.robots - 1) * 6 + config.pedestrians * 9;
            agent.MaxStep = config.max_steps;
            var behavior = new SerializedObject(robot.GetComponent<Unity.MLAgents.Policies.BehaviorParameters>());
            behavior.FindProperty("m_BrainParameters.VectorObservationSize").intValue = obsSize;
            behavior.ApplyModifiedPropertiesWithoutUndo();
            EditorUtility.SetDirty(robot);
            PrefabUtility.SavePrefabAsset(robot);

            var pedestrian = AssetDatabase.LoadAssetAtPath<GameObject>(PedestrianPrefab);
            var scene = SceneBuilder.Build(config, robot, pedestrian);

            string scenePath = $"{TmpDir}/{Path.GetFileNameWithoutExtension(outPath)}.unity";
            if (!EditorSceneManager.SaveScene(scene, scenePath)) throw new Exception("Could not save " + scenePath);
            AssetDatabase.SaveAssets();
            AssetDatabase.Refresh();
            Debug.Log($"[BuildCLI] room {config.room_size[0]} x {config.room_size[1]} m, {config.obstacles.Count} obstacles, "
                      + $"{config.robots} robots, {config.pedestrians} pedestrians, observation size {obsSize}");

            Directory.CreateDirectory(Path.GetDirectoryName(Path.GetFullPath(outPath)));
            File.WriteAllText(Path.ChangeExtension(Path.GetFullPath(outPath), ".json"), JsonUtility.ToJson(config, true));
            var report = BuildPipeline.BuildPlayer(new BuildPlayerOptions
            {
                scenes = new[] { scenePath },
                locationPathName = outPath,
                target = BuildTarget.StandaloneLinux64,
                options = BuildOptions.None,
            });
            Debug.Log($"[BuildCLI] {report.summary.result}: {outPath} ({report.summary.totalSize / 1e6:F0} MB, {report.summary.totalErrors} errors)");
            if (report.summary.result != BuildResult.Succeeded) throw new Exception("Build result: " + report.summary.result);
        }
        finally
        {
            EditorSceneManager.NewScene(NewSceneSetup.EmptyScene, NewSceneMode.Single);
            AssetDatabase.DeleteAsset(TmpDir);
        }
    }
}
