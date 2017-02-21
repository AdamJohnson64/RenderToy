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
    public struct TriIndex
    {
        public TriIndex(int i0, int i1, int i2)
        {
            Index0 = i0; Index1 = i1; Index2 = i2;
        }
        public readonly int Index0, Index1, Index2;
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
            var referenced_verts = triangles
                .SelectMany(t => new[] { t.Index0, t.Index1, t.Index2 })
                .Select(v => Vertices[v])
                .ToArray();
            var node = new BVHNode(
                new Point3D(referenced_verts.Min(v => v.X), referenced_verts.Min(v => v.Y), referenced_verts.Min(v => v.Z)),
                new Point3D(referenced_verts.Max(v => v.X), referenced_verts.Max(v => v.Y), referenced_verts.Max(v => v.Z)),
                triangles.ToArray(), null);
            node.Subdivide(Vertices);
        }
        public IReadOnlyList<Point3D> GetPoints() { return Vertices; }
        public Point3D[] Vertices;
        #region - Section : Bounding Volume Hierarchy Node -
        [DebuggerDisplay("[{Min.X}, {Min.Y}, {Min.Z}] -> [{Max.X}, {Max.Y}, {Max.Z}]")]
        class BVHNode
        {
            public BVHNode(Point3D min, Point3D max, IEnumerable<TriIndex> triangles, IEnumerable<BVHNode> children)
            {
                Min = min;
                Max = max;
                Triangles = triangles.ToArray();
                Children = children == null ? null : children.ToArray();
            }
            public void Subdivide(IReadOnlyList<Point3D> vertices)
            {
                // Stop at 16 triangles
                if (Triangles.Length < 32) return;
                // Slice this region into 8 subcubes (roughly octree).
                double midx = (Min.X + Max.X) / 2;
                double midy = (Min.Y + Max.Y) / 2;
                double midz = (Min.Z + Max.Z) / 2;
                List<BVHNode> children = new List<BVHNode>();
                for (int z = 0; z < 2; ++z)
                {
                    for (int y = 0; y < 2; ++y)
                    {
                        for (int x = 0; x < 2; ++x)
                        {
                            // Compute the expected subcube extents.
                            Point3D min = new Point3D(x == 0 ? Min.X : midx, y == 0 ? Min.Y : midy, z == 0 ? Min.Z : midz);
                            Point3D max = new Point3D(x == 0 ? midx : Max.X, y == 0 ? midy : Max.Y, z == 0 ? midz : Max.Z);
                            // Partition the triangles.
                            var contained_triangles = Triangles
                                .Where(t => TriangleInBounds(min, max, vertices[t.Index0], vertices[t.Index1], vertices[t.Index2]))
                                .ToArray();
                            // If there are no triangles in this child node then skip it entirely.
                            if (contained_triangles.Length == 0) continue;
                            // If all the triangles are still in the child then stop splitting this node.
                            // This might mean we have a rats nest of triangles with no potential split planes.
                            if (contained_triangles.Length == Triangles.Length) return;
                            // Recompute the extents of this bounding volume.
                            // It's possible if the mesh has large amounts of space crossing the clip plane that the bounds are now too big.
                            var points = contained_triangles
                                .SelectMany(t => new[] { t.Index0, t.Index1, t.Index2 })
                                .Select(v => vertices[v]);
                            min = new Point3D(points.Min(v => v.X), points.Min(v => v.Y), points.Min(v => v.Z));
                            max = new Point3D(points.Max(v => v.X), points.Max(v => v.Y), points.Max(v => v.Z));
                            // Generate the new child node.
                            var newnode = new BVHNode(min, max, contained_triangles, null);
                            newnode.Subdivide(vertices);
                            children.Add(newnode);
                        }
                    }
                }
                Triangles = null;
                Children = children.ToArray();
            }
            Point3D Min;
            Point3D Max;
            TriIndex[] Triangles;
            BVHNode[] Children;
        }
        #endregion
        #region - Section : Intersection Test (Separating Axis Theorem) -
        static IEnumerable<Point3D> EnumPoints(Point3D min, Point3D max)
        {
            for (int z = 0; z < 2; ++z)
                for (int y = 0; y < 2; ++y)
                    for (int x = 0; x < 2; ++x)
                        yield return new Point3D(x == 0 ? min.X : max.X, y == 0 ? min.Y : max.Y, z == 0 ? min.Z : max.Z);
        }
        static IEnumerable<Point3D> EnumPoints(Point3D p0, Point3D p1, Point3D p2)
        {
            yield return p0;
            yield return p1;
            yield return p2;
        }
        static Bound ProjectPoints(IEnumerable<Point3D> v, Vector3D project)
        {
            return new Bound(v.Select(x => MathHelp.Dot(x, project)).Min(), v.Select(x => MathHelp.Dot(x, project)).Max());
        }
        static bool TriangleInBounds(Point3D min, Point3D max, Point3D p0, Point3D p1, Point3D p2)
        {
            Vector3D axis_x = new Vector3D(1, 0, 0);
            Vector3D axis_y = new Vector3D(0, 1, 0);
            Vector3D axis_z = new Vector3D(0, 0, 1);
            Vector3D tri_n = MathHelp.Cross(p1 - p0, p2 - p0);
            Vector3D edge0 = p1 - p0;
            Vector3D edge1 = p2 - p1;
            Vector3D edge2 = p0 - p2;
            Vector3D[] axis_to_test =
            {
                axis_x, axis_y, axis_z, tri_n,
                MathHelp.Cross(axis_x, edge0), MathHelp.Cross(axis_x, edge1), MathHelp.Cross(axis_x, edge2),
                MathHelp.Cross(axis_y, edge0), MathHelp.Cross(axis_y, edge1), MathHelp.Cross(axis_y, edge2),
                MathHelp.Cross(axis_z, edge0), MathHelp.Cross(axis_z, edge1), MathHelp.Cross(axis_z, edge2),
            };
            return !axis_to_test
                .Any(axis => !Bound.Intersect(ProjectPoints(EnumPoints(min, max), axis), ProjectPoints(EnumPoints(p0, p1, p2), axis)));
        }
        struct Bound
        {
            public Bound(double min, double max)
            {
                Min = min;
                Max = max;
            }
            public static bool Intersect(Bound lhs, Bound rhs)
            {
                return !(lhs.Max < rhs.Min || lhs.Min > rhs.Max);
            }
            double Min, Max;
        }
        #endregion
    }
}