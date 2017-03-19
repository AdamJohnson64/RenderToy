////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace RenderToy
{
    /// <summary>
    /// Triangle-only mesh.
    /// </summary>
    public class Mesh : IPrimitive
    {
        public Mesh(IEnumerable<Vector3D> vertices, IEnumerable<TriIndex> triangles)
        {
            Vertices = vertices.ToArray();
            Triangles = triangles.ToArray();
        }
        public readonly Vector3D[] Vertices;
        public readonly TriIndex[] Triangles;
    }
    #region - Section : Bounding Volume Hierarchy Node -
    [DebuggerDisplay("[{Bound.Min.X}, {Bound.Min.Y}, {Bound.Min.Z}] -> [{Bound.Max.X}, {Bound.Max.Y}, {Bound.Max.Z}], {Triangles.Length} Triangles")]
    public class MeshBVH : IPrimitive
    {
        public MeshBVH(Bound3D bound, Triangle3D[] triangles, MeshBVH[] children)
        {
            Bound = bound;
            Triangles = triangles;
            Children = children;
        }
        public static MeshBVH Create(Triangle3D[] triangles)
        {
            Stopwatch perf = new Stopwatch();
            perf.Restart();
            MeshBVH root = BVH.CreateLooseOctree2(triangles.ToArray());
            perf.Stop();
            Performance.LogEvent("MeshBVH build took " + perf.ElapsedMilliseconds + "ms.");
            return root;
        }
        public static IEnumerable<MeshBVH> EnumerateNodes(MeshBVH from)
        {
            yield return from;
            if (from.Children != null)
            {
                foreach (var child in from.Children)
                {
                    foreach (var childnode in EnumerateNodes(child))
                    {
                        yield return childnode;
                    }
                }
            }
        }
        public readonly Bound3D Bound;
        public readonly Triangle3D[] Triangles;
        public readonly MeshBVH[] Children;
    }
    #endregion
    public static class MeshHelp
    {
        public static Mesh CreateMesh(IParametricUV shape, int usteps, int vsteps)
        {
            var vertices = new List<Vector3D>();
            for (int v = 0; v <= vsteps; ++v)
            {
                for (int u = 0; u <= usteps; ++u)
                {
                    vertices.Add(shape.GetPointUV((double)u / usteps, (double)v / vsteps));
                }
            }
            var indices = new List<TriIndex>();
            for (int v = 0; v < vsteps; ++v)
            {
                for (int u = 0; u < usteps; ++u)
                {
                    indices.Add(new TriIndex((u + 0) + (v + 0) * (usteps + 1), (u + 1) + (v + 0) * (usteps + 1), (u + 1) + (v + 1) * (usteps + 1)));
                    indices.Add(new TriIndex((u + 1) + (v + 1) * (usteps + 1), (u + 0) + (v + 1) * (usteps + 1), (u + 0) + (v + 0) * (usteps + 1)));
                }
            }
            return new Mesh(vertices, indices);
        }
        public static IEnumerable<Triangle3D> CollapseIndices(IReadOnlyList<Vector3D> vertices, IEnumerable<TriIndex> triangles)
        {
            return triangles.Select(t => new Triangle3D(vertices[t.Index0], vertices[t.Index1], vertices[t.Index2]));
        }
    }
}