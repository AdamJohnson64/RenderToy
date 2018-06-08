////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Diagnostics;
using RenderToy.Math;
using RenderToy.Primitives;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace RenderToy.Meshes
{
    [DebuggerDisplay("[{Bound.Min.X}, {Bound.Min.Y}, {Bound.Min.Z}] -> [{Bound.Max.X}, {Bound.Max.Y}, {Bound.Max.Z}], {Triangles.Length} Triangles")]
    public class MeshBVH : IPrimitive
    {
        public MeshBVH(Bound3D bound, Triangle3D[] triangles, MeshBVH[] children)
        {
            Bound = bound;
            Triangles = triangles;
            Children = children;
        }
        public static MeshBVH Create(Mesh mesh)
        {
            return Create(Triangle3D.ExtractTriangles((IReadOnlyList<Vector3D>)mesh.Vertices.GetVertices(), (IEnumerable<int>)mesh.Vertices.GetIndices()).ToArray());
        }
        public static MeshBVH Create(Triangle3D[] triangles)
        {
            Stopwatch perf = new Stopwatch();
            perf.Restart();
            MeshBVH root = BoundingVolumeHierarchy.KDTree.Create(triangles.ToArray());
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
}