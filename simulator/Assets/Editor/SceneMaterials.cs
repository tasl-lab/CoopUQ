using UnityEditor;
using UnityEngine;

// Materials for the generated scenes, created as assets in the temporary build folder.
public static class SceneMaterials
{
    static Material Make(string name, Color color, float smoothness = 0.2f)
    {
        string path = $"{BuildCLI.TmpDir}/{name}.mat";
        var existing = AssetDatabase.LoadAssetAtPath<Material>(path);
        if (existing != null) return existing;
        var material = new Material(Shader.Find("Universal Render Pipeline/Lit"));
        material.SetColor("_BaseColor", color);
        material.SetFloat("_Smoothness", smoothness);
        AssetDatabase.CreateAsset(material, path);
        return material;
    }

    public static Material Wall() => Make("Wall", new Color(0.86f, 0.84f, 0.80f));
    public static Material Obstacle() => Make("Obstacle", new Color(0.55f, 0.40f, 0.27f));

    // Light floor with a 1 m grid.
    public static Material Floor(float sizeX, float sizeZ)
    {
        var material = Make("Floor", Color.white, 0.1f);
        string path = $"{BuildCLI.TmpDir}/FloorTile.asset";
        var texture = AssetDatabase.LoadAssetAtPath<Texture2D>(path);
        if (texture == null)
        {
            const int n = 128;
            texture = new Texture2D(n, n, TextureFormat.RGB24, true);
            var tile = new Color(0.78f, 0.79f, 0.80f);
            var line = new Color(0.62f, 0.63f, 0.65f);
            for (int x = 0; x < n; x++)
                for (int y = 0; y < n; y++)
                    texture.SetPixel(x, y, x < 2 || y < 2 ? line : tile);
            texture.wrapMode = TextureWrapMode.Repeat;
            texture.anisoLevel = 8;
            texture.Apply(true);
            AssetDatabase.CreateAsset(texture, path);
        }
        material.SetTexture("_BaseMap", texture);
        material.SetTextureScale("_BaseMap", new Vector2(sizeX, sizeZ));
        EditorUtility.SetDirty(material);
        return material;
    }
}
