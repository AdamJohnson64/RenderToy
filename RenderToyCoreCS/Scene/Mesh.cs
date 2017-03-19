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
    public class MeshBVH : IPrimitive
    {
        public MeshBVH(IEnumerable<Triangle3D> triangles)
        {
            Stopwatch perf = new Stopwatch();
            perf.Restart();
            Root = BVH.CreateLooseOctree(triangles.ToArray());
            perf.Stop();
            Performance.LogEvent("MeshBVH build took " + perf.ElapsedMilliseconds + "ms.");
            var allnodes = EnumerateNodes(Root);
            int count_triangles_initial = triangles.Count();
            int count_triangles_final = EnumerateNodes(Root).Where(x => x.Triangles != null).SelectMany(x => x.Triangles).Count();
        }
        public static IEnumerable<Node> EnumerateNodes(Node from)
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
        public readonly Node Root;
        #region - Section : Bounding Volume Hierarchy Node -
        [DebuggerDisplay("[{Bound.Min.X}, {Bound.Min.Y}, {Bound.Min.Z}] -> [{Bound.Max.X}, {Bound.Max.Y}, {Bound.Max.Z}], {Triangles.Length} Triangles")]
        public class Node
        {
            public Node(Bound3D bound, Triangle3D[] triangles, Node[] children)
            {
                Bound = bound;
                Triangles = triangles;
                Children = children;
            }
            public readonly Bound3D Bound;
            public readonly Triangle3D[] Triangles;
            public readonly Node[] Children;
        }
        #endregion
    }
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