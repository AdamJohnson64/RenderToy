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
    public class Mesh
    {
        public Mesh(IEnumerable<Vector3D> vertices, IEnumerable<TriIndex> triangles)
        {
            Vertices = vertices.ToArray();
            Triangles = triangles.ToArray();
        }
        public readonly Vector3D[] Vertices;
        public readonly TriIndex[] Triangles;
    }
    public class MeshBVH
    {
        public MeshBVH(IEnumerable<Triangle3D> triangles)
        {
            Root = CreateLooseOctree(triangles.ToArray(), 6);
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
        #region - Section : Hierarchy Construction -
        public static Node CreateLooseOctree(Triangle3D[] triangles, int level)
        {
            Bound3D bound = ComputeBounds(triangles);
            // Stop at 8 levels.
            if (level <= 0) goto EMITUNMODIFIED;
            // Stop at 4 triangles
            if (triangles.Length < 4) goto EMITUNMODIFIED;
            // Slice this region into 8 subcubes (roughly octree).
            List<Node> children = new List<Node>();
            foreach (var subbox in EnumerateSplit222(bound))
            {
                // Partition the triangles.
                var contained_triangles = triangles
                    .Where(t => ShapeIntersects(subbox, new Triangle3D(t.P0, t.P1, t.P2)))
                    .ToArray();
                // If there are no triangles in this child node then skip it entirely.
                if (contained_triangles.Length == 0) continue;
                // If all the triangles are still in the child then stop splitting this node.
                // This might mean we have a rats nest of triangles with no potential split planes.
                if (contained_triangles.Length == triangles.Length) goto EMITUNMODIFIED;
                // Generate the new child node.
                // Also, recompute the extents of this bounding volume.
                // It's possible if the mesh has large amounts of space crossing the clip plane such that the bounds are now too big.
                var newnode = CreateLooseOctree(contained_triangles, level - 1);
                children.Add(newnode);
            }
            return new Node(bound, null, children.ToArray());
        EMITUNMODIFIED:
            return new Node(bound, triangles, null);
        }
        #endregion
        #region - Section : Intersection Test (Separating Axis Theorem) -
        static Bound3D ComputeBounds(IEnumerable<Triangle3D> triangles)
        {
            var vertices = triangles.SelectMany(t => EnumeratePoints(t));
            return new Bound3D(
                new Vector3D(vertices.Min(p => p.X), vertices.Min(p => p.Y), vertices.Min(p => p.Z)),
                new Vector3D(vertices.Max(p => p.X), vertices.Max(p => p.Y), vertices.Max(p => p.Z)));
        }
        static Bound3D ComputeBounds(IEnumerable<Vector3D> vertices)
        {
            return new Bound3D(
                new Vector3D(vertices.Min(p => p.X), vertices.Min(p => p.Y), vertices.Min(p => p.Z)),
                new Vector3D(vertices.Max(p => p.X), vertices.Max(p => p.Y), vertices.Max(p => p.Z)));
        }
        static Bound3D ComputeBounds(IReadOnlyList<Vector3D> vertices, IEnumerable<TriIndex> indices)
        {
            return ComputeBounds(
                indices
                .SelectMany(i => new[] { i.Index0, i.Index1, i.Index2 })
                .Select(i => vertices[i])
                .ToArray());
        }
        static IEnumerable<Vector3D> EnumeratePoints(Bound3D box)
        {
            for (int z = 0; z < 2; ++z)
            {
                for (int y = 0; y < 2; ++y)
                {
                    for (int x = 0; x < 2; ++x)
                    {
                        yield return new Vector3D(
                            x == 0 ? box.Min.X : box.Max.X,
                            y == 0 ? box.Min.Y : box.Max.Y,
                            z == 0 ? box.Min.Z : box.Max.Z);
                    }
                }
            }
        }
        static IEnumerable<Vector3D> EnumeratePoints(Triangle3D triangle)
        {
            yield return triangle.P0;
            yield return triangle.P1;
            yield return triangle.P2;
        }
        static IEnumerable<Bound3D> EnumerateSplit222(Bound3D box)
        {
            double midx = (box.Min.X + box.Max.X) / 2;
            double midy = (box.Min.Y + box.Max.Y) / 2;
            double midz = (box.Min.Z + box.Max.Z) / 2;
            for (int z = 0; z < 2; ++z)
            {
                for (int y = 0; y < 2; ++y)
                {
                    for (int x = 0; x < 2; ++x)
                    {
                        // Compute the expected subcube extents.
                        Vector3D min = new Vector3D(x == 0 ? box.Min.X : midx, y == 0 ? box.Min.Y : midy, z == 0 ? box.Min.Z : midz);
                        Vector3D max = new Vector3D(x == 0 ? midx : box.Max.X, y == 0 ? midy : box.Max.Y, z == 0 ? midz : box.Max.Z);
                        yield return new Bound3D(min, max);
                    }
                }
            }
        }
        static bool ShapeContains(Bound3D box, Vector3D point)
        {
            return
                point.X >= box.Min.X && point.X <= box.Max.X &&
                point.Y >= box.Min.Y && point.Y <= box.Max.Y &&
                point.Z >= box.Min.Z && point.Z <= box.Max.Z;
        }
        static bool ShapeContains(Bound3D box, Triangle3D triangle)
        {
            return EnumeratePoints(triangle).All(p => ShapeContains(box, p));
        }
        static bool ShapeIntersects(Bound1D lhs, Bound1D rhs) { return !(lhs.Max < rhs.Min || lhs.Min > rhs.Max); }
        static bool ShapeIntersects(Bound3D box, Triangle3D triangle)
        {
            Vector3D aabb_x = new Vector3D(1, 0, 0);
            Vector3D aabb_y = new Vector3D(0, 1, 0);
            Vector3D aabb_z = new Vector3D(0, 0, 1);
            Vector3D tnorml = MathHelp.Cross(triangle.P1 - triangle.P0, triangle.P2 - triangle.P0);
            Vector3D tedg_0 = triangle.P1 - triangle.P0;
            Vector3D tedg_1 = triangle.P2 - triangle.P1;
            Vector3D tedg_2 = triangle.P0 - triangle.P2;
            Vector3D[] axis_to_test =
            {
                aabb_x, aabb_y, aabb_z, tnorml,
                MathHelp.Cross(aabb_x, tedg_0), MathHelp.Cross(aabb_x, tedg_1), MathHelp.Cross(aabb_x, tedg_2),
                MathHelp.Cross(aabb_y, tedg_0), MathHelp.Cross(aabb_y, tedg_1), MathHelp.Cross(aabb_y, tedg_2),
                MathHelp.Cross(aabb_z, tedg_0), MathHelp.Cross(aabb_z, tedg_1), MathHelp.Cross(aabb_z, tedg_2),
            };
            return !axis_to_test
                .Any(axis => !ShapeIntersects(ProjectPoints(EnumeratePoints(box), axis), ProjectPoints(EnumeratePoints(triangle), axis)));
        }
        static Bound1D ProjectPoints(IEnumerable<Vector3D> vertices, Vector3D project)
        {
            return new Bound1D(vertices.Select(x => MathHelp.Dot(x, project)).Min(), vertices.Select(x => MathHelp.Dot(x, project)).Max());
        }
        #endregion
    }
    public struct Bound1D
    {
        public Bound1D(double min, double max) { Min = min; Max = max; }
        public readonly double Min, Max;
    }
    public struct Bound3D
    {
        public Bound3D(Vector3D min, Vector3D max) { Min = min; Max = max; }
        public readonly Vector3D Min, Max;
    }
    public struct Triangle3D
    {
        public Triangle3D(Vector3D p0, Vector3D p1, Vector3D p2) { P0 = p0; P1 = p1; P2 = p2; }
        public readonly Vector3D P0, P1, P2;
    }
    public struct Triangle4D
    {
        public Triangle4D(Vector4D p0, Vector4D p1, Vector4D p2) { P0 = p0; P1 = p1; P2 = p2; }
        public readonly Vector4D P0, P1, P2;
    }
    public struct TriIndex
    {
        public TriIndex(int i0, int i1, int i2) { Index0 = i0; Index1 = i1; Index2 = i2; }
        public readonly int Index0, Index1, Index2;
    }
}