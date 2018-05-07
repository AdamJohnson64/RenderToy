////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System;
using System.Collections.Generic;
using System.Linq;

namespace RenderToy.BoundingVolumeHierarchy
{
    public static partial class CommonBVH
    {
        #region - Section : Bounds -
        public static Bound3D ComputeBounds(IEnumerable<Triangle3D> triangles)
        {
            return ComputeBounds(triangles.SelectMany(t => EnumeratePoints(t)));
        }
        public static Bound3D ComputeBounds(IEnumerable<Vector3D> vertices)
        {
            Vector3D min = new Vector3D(double.PositiveInfinity, double.PositiveInfinity, double.PositiveInfinity);
            Vector3D max = new Vector3D(double.NegativeInfinity, double.NegativeInfinity, double.NegativeInfinity);
            foreach (var v in vertices)
            {
                min.X = Math.Min(min.X, v.X);
                min.Y = Math.Min(min.Y, v.Y);
                min.Z = Math.Min(min.Z, v.Z);
                max.X = Math.Max(max.X, v.X);
                max.Y = Math.Max(max.Y, v.Y);
                max.Z = Math.Max(max.Z, v.Z);
            }
            return new Bound3D(min, max);
        }
        public static Bound3D ComputeBounds(IReadOnlyList<Vector3D> vertices, IEnumerable<TriIndex> indices)
        {
            return ComputeBounds(
                indices
                .SelectMany(i => new[] { vertices[i.Index0], vertices[i.Index1], vertices[i.Index2] }));
        }
        #endregion
        #region - Section : Intersection Test (Separating Axis Theorem) -
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
        public static bool ShapeContains(Bound3D box, Vector3D point)
        {
            return
                point.X >= box.Min.X && point.X <= box.Max.X &&
                point.Y >= box.Min.Y && point.Y <= box.Max.Y &&
                point.Z >= box.Min.Z && point.Z <= box.Max.Z;
        }
        /// <summary>
        /// Determine if one range completely contains another.
        /// </summary>
        /// <param name="lhs">The containing range.</param>
        /// <param name="rhs">The range to test for containment.</param>
        /// <returns>True if rhs is completely contained by (and DOES NOT extend outside) lhs.</returns>
        public static bool ShapeContains(Bound1D lhs, Bound1D rhs)
        {
            return rhs.Min >= lhs.Min && rhs.Max <= lhs.Max;
        }
        /// <summary>
        /// Determine if one 3D AABB completely contains another.
        /// </summary>
        /// <param name="lhs">The containing AABB.</param>
        /// <param name="rhs">The AABB to test for containment.</param>
        /// <returns>True if rhs is completely contained by (and DOES NOT extend outside) lhs.</returns>
        public static bool ShapeContains(Bound3D lhs, Bound3D rhs)
        {
            return
                ShapeContains(new Bound1D(lhs.Min.X, lhs.Max.X), new Bound1D(rhs.Min.X, rhs.Max.X)) &&
                ShapeContains(new Bound1D(lhs.Min.Y, lhs.Max.Y), new Bound1D(rhs.Min.Y, rhs.Max.Y)) &&
                ShapeContains(new Bound1D(lhs.Min.Z, lhs.Max.Z), new Bound1D(rhs.Min.Z, rhs.Max.Z));
        }
        public static bool ShapeContains(Bound3D box, Triangle3D triangle)
        {
            return EnumeratePoints(triangle).All(p => ShapeContains(box, p));
        }
        public static bool ShapeIntersects(Bound1D lhs, Bound1D rhs) { return !(lhs.Max < rhs.Min || lhs.Min > rhs.Max); }
        public static bool ShapeIntersects(Bound3D box, Triangle3D triangle)
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
        /// <summary>
        /// Unfortunately we have to clamp the BVH depth quite aggressively.
        /// In CUDA this is quite conservative.
        /// In C++AMP setting this too high kills the compiler.
        /// </summary>
        public const int MAXIMUM_BVH_DEPTH = 8;
    }
}

namespace RenderToy
{
    #region - Section : Data Types -
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
    public struct TriIndex
    {
        public TriIndex(int i0, int i1, int i2) { Index0 = i0; Index1 = i1; Index2 = i2; }
        public readonly int Index0, Index1, Index2;
    }
    #endregion
}