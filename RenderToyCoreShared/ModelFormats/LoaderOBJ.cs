using RenderToy.Meshes;
using RenderToy.Primitives;
using RenderToy.Utility;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace RenderToy.ModelFormat
{
    static class LoaderOBJ
    {
        public static IEnumerable<int> GenerateFaces(int vertexcount)
        {
            if ((vertexcount % 3) != 0) throw new FileLoadException("Bad vertex count.");
            for (int i = 0; i < vertexcount; ++i)
            {
                yield return i;
            }
        }
        public static IEnumerable<IPrimitive> LoadFromPath(string path)
        {
            var vertices = LoadFromPath2(path).ToArray();
            Mesh mesh = new Mesh(vertices, GenerateFaces(vertices.Length));
            yield return mesh;
        }
        public static IEnumerable<Vector3D> LoadFromPath2(string path)
        {
            var vertices = new List<Vector3D>();
            var normals = new List<Vector3D>();
            var texcoords = new List<Vector2D>();
            string groupname = null;
            string materialname = null;
            int smoothinggroup = -1;
            using (var stream = File.OpenText(path))
            {
                var line = stream.ReadLine();
                while ((line = stream.ReadLine()) != null)
                {
                    if (line.StartsWith("#")) continue;
                    if (line.Length == 0) continue;
                    if (line.StartsWith("mtllib ")) continue;
                    if (line.StartsWith("v "))
                    {
                        var parts = line.Split(new char[] { ' ' }, System.StringSplitOptions.RemoveEmptyEntries);
                        if (parts.Length != 4) throw new FileLoadException("Malformed vertex '" + line + "'.");
                        var v = new Vector3D();
                        v.X = double.Parse(parts[1]);
                        v.Y = double.Parse(parts[2]);
                        v.Z = double.Parse(parts[3]);
                        vertices.Add(v);
                        continue;
                    }
                    if (line.StartsWith("vn "))
                    {
                        var parts = line.Split(new char[] { ' ' }, System.StringSplitOptions.RemoveEmptyEntries);
                        if (parts.Length != 4) throw new FileLoadException("Malformed normal '" + line + "'.");
                        var v = new Vector3D();
                        v.X = double.Parse(parts[1]);
                        v.Y = double.Parse(parts[2]);
                        v.Z = double.Parse(parts[3]);
                        normals.Add(v);
                        continue;
                    }
                    if (line.StartsWith("vt "))
                    {
                        var parts = line.Split(new char[] { ' ' }, System.StringSplitOptions.RemoveEmptyEntries);
                        if (parts.Length != 4) throw new FileLoadException("Malformed texture coordinate '" + line + "'.");
                        var v = new Vector2D();
                        v.X = double.Parse(parts[1]);
                        v.Y = double.Parse(parts[2]);
                        texcoords.Add(v);
                        continue;
                    }
                    if (line.StartsWith("g "))
                    {
                        var parts = line.Split(new char[] { ' ' }, System.StringSplitOptions.RemoveEmptyEntries);
                        if (parts.Length != 2) throw new FileLoadException("Malformed group '" + line + "'.");
                        groupname = parts[1];
                        continue;
                    }
                    if (line.StartsWith("s "))
                    {
                        var parts = line.Split(new char[] { ' ' }, System.StringSplitOptions.RemoveEmptyEntries);
                        if (parts.Length != 2) throw new FileLoadException("Malformed smoothing group '" + line + "'.");
                        if (parts[1] == "off")
                        {
                            smoothinggroup = -1;
                            continue;
                        }
                        smoothinggroup = int.Parse(parts[1]);
                        continue;
                    }
                    if (line.StartsWith("usemtl "))
                    {
                        var parts = line.Split(new char[] { ' ' }, System.StringSplitOptions.RemoveEmptyEntries);
                        if (parts.Length != 2) throw new FileLoadException("Malformed usemtl '" + line + "'.");
                        materialname = parts[1];
                        continue;
                    }
                    if (line.StartsWith("f "))
                    {
                        var parts = line.Split(new char[] { ' ' }, System.StringSplitOptions.RemoveEmptyEntries);
                        if (parts.Length < 4) throw new FileLoadException("Insufficient indices '" + line + "'.");
                        if (parts.Length > 5) throw new FileLoadException("Too many indices '" + line + "'.");
                        var f = parts.Skip(1).Select(i => i.Split(new char[] { '/' })).ToArray();
                        if (!f.All(i => i.Length == f[0].Length)) throw new FileLoadException("Inconsistent face setups '" + line + "'.");
                        if (f[0].Length < 1 || f[0].Length > 3) throw new FileLoadException("Bad face component count '" + line + "'.");
                        int[] idxv = f.Select(i => int.Parse(i[0]) - 1).ToArray();
                        int[] idxt = f.Select(i => int.Parse(i[1]) - 1).ToArray();
                        int[] idxn = f.Select(i => int.Parse(i[2]) - 1).ToArray();
                        if (parts.Length == 4)
                        {
                            yield return vertices[idxv[0]];
                            yield return vertices[idxv[1]];
                            yield return vertices[idxv[2]];
                        }
                        if (parts.Length == 5)
                        {
                            yield return vertices[idxv[0]];
                            yield return vertices[idxv[1]];
                            yield return vertices[idxv[2]];
                            yield return vertices[idxv[2]];
                            yield return vertices[idxv[3]];
                            yield return vertices[idxv[0]];
                        }
                        continue;
                    }
                    throw new FileLoadException("Unknown tag '" + line + "'.");
                }
            }
        }
    }
}