////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System;
using System.Collections.Generic;
using System.Diagnostics;
using System.Linq;

namespace RenderToy
{
    public interface IPoints
    {
        IReadOnlyList<Point3D> GetPoints();
    }
    public interface IMesh
    {
        IReadOnlyList<Point3D> GetMeshVertices();
        IReadOnlyList<TriIndex> GetMeshTriangles();
    }
    public interface IParametricUV
    {
        /// <summary>
        /// Get a 3D point on this parametric surface.
        /// Parametric surfaces are only meaningfully defined in the range [0,1] in both U and V.
        /// </summary>
        /// <param name="u">The U location on the surface.</param>
        /// <param name="v">The V location on the surface.</param>
        /// <returns>A 3D point in object local space.</returns>
        Point3D GetPointUV(double u, double v);
    }
    public interface IParametricUVW
    {
        /// <summary>
        /// Get a 3D point within a parametric volume.
        /// Parametric volumes are only meaningfully defined in the range [0,1] in U, V and W.
        /// </summary>
        /// <param name="u">The U location in the volume.</param>
        /// <param name="v">The V location in the volume.</param>
        /// <param name="w">The W location in the volume.</param>
        /// <returns>A 3D point in object local space.</returns>
        Point3D GetPointUVW(double u, double v, double w);
    }
    public class BezierPatch : IParametricUV
    {
        public BezierPatch()
        {
            // Define the hull for the patch.
            const double h = 0.5;
            hull = new Point3D[16]
            {
                new Point3D(-1, 0, -1), new Point3D(-h, 0, -1), new Point3D(+h, 0, -1), new Point3D(+1, 0, -1),
                new Point3D(-1, 0, -h), new Point3D(-h, 4, -h), new Point3D(+h, -4, -h), new Point3D(+1, 0, -h),
                new Point3D(-1, 0, +h), new Point3D(-h, -4, +h), new Point3D(+h, 4, +h), new Point3D(+1, 0, +h),
                new Point3D(-1, 0, +1), new Point3D(-h, 0, +1), new Point3D(+h, 0, +1), new Point3D(+1, 0, +1),
            };
        }
        public BezierPatch(Point3D[] hull)
        {
            this.hull = hull;
        }
        public Point3D GetPointUV(double u, double v)
        {
            // The Bernstein polynomial factors.
            double nu = 1 - u;
            double[] bu = new double[4] { nu * nu * nu, 3 * u * nu * nu, 3 * u * u * nu, u * u * u };
            double nv = 1 - v;
            double[] bv = new double[4] { nv * nv * nv, 3 * v * nv * nv, 3 * v * v * nv, v * v * v };
            // Compute the UV point.
            Point3D acc = new Point3D(0, 0, 0);
            for (int j = 0; j < 4; ++j)
            {
                for (int i = 0; i < 4; ++i)
                {
                    acc = MathHelp.Add(acc, MathHelp.Multiply(hull[i + j * 4], bu[i] * bv[j]));
                }
            }
            return acc;
        }
        Point3D[] hull = null;
    }
    public class Cube : IParametricUVW
    {
        public Point3D GetPointUVW(double u, double v, double w)
        {
            return new Point3D(-1 + u * 2, -1 + v * 2, -1 + w * 2);
        }
    }
    public class Cylinder : IParametricUV
    {
        public Point3D GetPointUV(double u, double v)
        {
            // The central axis of the sphere points through world Y.
            // The U direction defines latitude and sweeps a full circle for 0 <= u <= 1.
            // The V direction defines linear distance along Y.
            double ucos = Math.Cos(u * Math.PI * 2);
            double usin = Math.Sin(u * Math.PI * 2);
            return new Point3D(-usin, -1 + v * 2, ucos);
        }
    }
    /// <summary>
    /// Plane in XZ.
    /// Note that for the purposes of parametric definitions this plane is bounded [-1,+1] in X and Z.
    /// The raytracer definition of this plane is infinite in the XZ plane.
    /// </summary>
    public class Plane : IParametricUV
    {
        public Point3D GetPointUV(double u, double v)
        {
            return new Point3D(-1 + u * 2, 0, -1 + v * 2);
        }
    }
    /// <summary>
    /// Sphere of unit radius.
    /// The parametric definition of this sphere is oriented with the poles in Y.
    /// The "seam" of the sphere is deliberately behind the sphere in +Z.
    /// </summary>
    public class Sphere : IParametricUV
    {
        public Point3D GetPointUV(double u, double v)
        {
            // The central axis of the sphere points through world Y.
            // The U direction defines latitude and sweeps a full circle for 0 <= u <= 1.
            // The V direction defines longitude ans sweeps a half circle for 0 <= v <= 1.
            double ucos = Math.Cos(u * Math.PI * 2);
            double usin = Math.Sin(u * Math.PI * 2);
            double vcos = Math.Cos(v * Math.PI);
            double vsin = Math.Sin(v * Math.PI);
            return new Point3D(-usin * vsin, vcos, ucos * vsin);
        }
    }
    /// <summary>
    /// Single triangle [0,0,0], [0,1,0], [1,0,0].
    /// </summary>
    public class Triangle
    {
    }
    /// <summary>
    /// Triangle-only mesh.
    /// </summary>
    public class Mesh : IMesh
    {
        public Mesh(IEnumerable<Point3D> vertices, IEnumerable<TriIndex> triangles)
        {
            Vertices = vertices.ToArray();
            Triangles = triangles.ToArray();
        }
        public IReadOnlyList<Point3D> GetMeshVertices() { return Vertices; }
        public IReadOnlyList<TriIndex> GetMeshTriangles() { return Triangles; }
        public Point3D[] Vertices;
        public TriIndex[] Triangles;
    }
    public class MeshBVH : IPoints
    {
        public MeshBVH(IEnumerable<Point3D> vertices, IEnumerable<TriIndex> triangles)
        {
            Vertices = vertices.ToArray();
            var node = new Node(ComputeBounds(Vertices, triangles), triangles.ToArray(), null);
            node.Subdivide(Vertices, 0);
            Root = node;
            var allnodes = EnumNodes(Root);
            int count_triangles_initial = triangles.Count();
            int count_triangles_final = EnumNodes(Root).Where(x => x.Triangles != null).SelectMany(x => x.Triangles).Count();
            int test = 0;
        }
        static IEnumerable<Node> EnumNodes(Node from)
        {
            yield return from;
            if (from.Children != null)
            {
                foreach (var child in from.Children)
                {
                    foreach (var childnode in EnumNodes(child))
                    {
                        yield return childnode;
                    }
                }
            }
        }
        public IReadOnlyList<Point3D> GetPoints() { return Vertices; }
        public Point3D[] Vertices;
        public Node Root;
        #region - Section : Bounding Volume Hierarchy Node -
        [DebuggerDisplay("[{Min.X}, {Min.Y}, {Min.Z}] -> [{Max.X}, {Max.Y}, {Max.Z}]")]
        public class Node
        {
            public Node(Bound3D bound, IEnumerable<TriIndex> triangles, IEnumerable<Node> children)
            {
                Bound = bound;
                Triangles = triangles.ToArray();
                Children = children == null ? null : children.ToArray();
            }
            public void Subdivide(IReadOnlyList<Point3D> vertices, int level)
            {
                // Stop at 8 levels.
                if (level == 8) return;
                // Stop at 4 triangles
                if (Triangles.Length < 4) return;
                // Slice this region into 8 subcubes (roughly octree).
                List<Node> children = new List<Node>();
                foreach (var subbox in EnumerateSplit222(Bound))
                {
                    // Partition the triangles.
                    var contained_triangles = Triangles
                        .Where(t => ShapeIntersects(subbox, new Triangle3D(vertices[t.Index0], vertices[t.Index1], vertices[t.Index2])))
                        .ToArray();
                    // If there are no triangles in this child node then skip it entirely.
                    if (contained_triangles.Length == 0) continue;
                    // If all the triangles are still in the child then stop splitting this node.
                    // This might mean we have a rats nest of triangles with no potential split planes.
                    if (contained_triangles.Length == Triangles.Length) return;
                    // Generate the new child node.
                    // Also, recompute the extents of this bounding volume.
                    // It's possible if the mesh has large amounts of space crossing the clip plane such that the bounds are now too big.
                    var newnode = new Node(ComputeBounds(vertices, contained_triangles), contained_triangles, null);
                    newnode.Subdivide(vertices, level + 1);
                    children.Add(newnode);
                }
                Triangles = null;
                Children = children.ToArray();
            }
            public Bound3D Bound;
            public TriIndex[] Triangles;
            public Node[] Children;
        }
        #endregion
        #region - Section : Intersection Test (Separating Axis Theorem) -
        public static Bound3D ComputeBounds(IEnumerable<Point3D> vertices)
        {
            return new Bound3D(
                new Point3D(vertices.Min(p => p.X), vertices.Min(p => p.Y), vertices.Min(p => p.Z)),
                new Point3D(vertices.Max(p => p.X), vertices.Max(p => p.Y), vertices.Max(p => p.Z)));
        }
        static Bound3D ComputeBounds(IReadOnlyList<Point3D> vertices, IEnumerable<TriIndex> indices)
        {
            return ComputeBounds(
                indices
                .SelectMany(i => new[] { i.Index0, i.Index1, i.Index2 })
                .Select(i => vertices[i])
                .ToArray());
        }
        static IEnumerable<Point3D> EnumeratePoints(Bound3D box)
        {
            for (int z = 0; z < 2; ++z)
            {
                for (int y = 0; y < 2; ++y)
                {
                    for (int x = 0; x < 2; ++x)
                    {
                        yield return new Point3D(
                            x == 0 ? box.Min.X : box.Max.X,
                            y == 0 ? box.Min.Y : box.Max.Y,
                            z == 0 ? box.Min.Z : box.Max.Z);
                    }
                }
            }
        }
        static IEnumerable<Point3D> EnumeratePoints(Triangle3D triangle)
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
                        Point3D min = new Point3D(x == 0 ? box.Min.X : midx, y == 0 ? box.Min.Y : midy, z == 0 ? box.Min.Z : midz);
                        Point3D max = new Point3D(x == 0 ? midx : box.Max.X, y == 0 ? midy : box.Max.Y, z == 0 ? midz : box.Max.Z);
                        yield return new Bound3D(min, max);
                    }
                }
            }
        }
        static bool ShapeContains(Bound3D box, Point3D point)
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
                .Any(axis => !Bound1D.Intersect(ProjectPoints(EnumeratePoints(box), axis), ProjectPoints(EnumeratePoints(triangle), axis)));
        }
        static Bound1D ProjectPoints(IEnumerable<Point3D> vertices, Vector3D project)
        {
            return new Bound1D(vertices.Select(x => MathHelp.Dot(x, project)).Min(), vertices.Select(x => MathHelp.Dot(x, project)).Max());
        }
        #endregion
    }
    public struct Bound1D
    {
        public Bound1D(double min, double max) { Min = min; Max = max; }
        public static bool Intersect(Bound1D lhs, Bound1D rhs) { return !(lhs.Max < rhs.Min || lhs.Min > rhs.Max); }
        double Min, Max;
    }
    public struct Bound3D
    {
        public Bound3D(Point3D min, Point3D max) { Min = min; Max = max; }
        public readonly Point3D Min, Max;
    }
    public struct Triangle3D
    {
        public Triangle3D(Point3D p0, Point3D p1, Point3D p2) { P0 = p0; P1 = p1; P2 = p2; }
        public readonly Point3D P0, P1, P2;
    }
    public struct TriIndex
    {
        public TriIndex(int i0, int i1, int i2)
        {
            Index0 = i0; Index1 = i1; Index2 = i2;
        }
        public readonly int Index0, Index1, Index2;
    }
}