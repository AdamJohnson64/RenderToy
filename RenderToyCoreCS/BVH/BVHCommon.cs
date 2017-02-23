////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System.Collections.Generic;
using System.Linq;

namespace RenderToy
{
    public static partial class BVH
    {
        #region - Section : Bounds -
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
    #endregion
}