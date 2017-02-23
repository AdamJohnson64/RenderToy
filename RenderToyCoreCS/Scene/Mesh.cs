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
            Root = BVH.CreateLooseOctree(triangles.ToArray(), 6);
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
        [DebuggerDisplay("[{Min.X}, {Min.Y}, {Min.Z}] -> [{Max.X}, {Max.Y}, {Max.Z}]")]
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
}