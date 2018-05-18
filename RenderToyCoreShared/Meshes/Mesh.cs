////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Primitives;
using RenderToy.Utility;
using System;
using System.Collections.Generic;
using System.Linq;

namespace RenderToy.Meshes
{
    /// <summary>
    /// Triangle-only mesh.
    /// </summary>
    public class Mesh : IPrimitive
    {
        public Mesh(IEnumerable<Vector3D> vertices, IEnumerable<int> triangles)
        {
            Vertices = vertices.ToArray();
            Triangles = triangles.ToArray();
        }
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
            var indices = new List<int>();
            for (int v = 0; v < vsteps; ++v)
            {
                for (int u = 0; u < usteps; ++u)
                {
                    indices.Add((u + 0) + (v + 0) * (usteps + 1));
                    indices.Add((u + 1) + (v + 0) * (usteps + 1));
                    indices.Add((u + 1) + (v + 1) * (usteps + 1));
                    indices.Add((u + 1) + (v + 1) * (usteps + 1));
                    indices.Add((u + 0) + (v + 1) * (usteps + 1));
                    indices.Add((u + 0) + (v + 0) * (usteps + 1));
                }
            }
            return new Mesh(vertices, indices);
        }
        public static IEnumerable<Triangle3D> FlattenIndices(IReadOnlyList<Vector3D> vertices, IEnumerable<int> indices)
        {
            return Mesh.ExtractTriangles(indices).Select(t => new Triangle3D(vertices[t.Index0], vertices[t.Index1], vertices[t.Index2]));
        }
        public readonly Vector3D[] Vertices;
        public readonly int[] Triangles;
        internal static IEnumerable<Triangle3D> ExtractTriangles(IEnumerable<Vector3D> triangles)
        {
            var iter = triangles.GetEnumerator();
            while (iter.MoveNext())
            {
                var p0 = iter.Current;
                if (!iter.MoveNext()) throw new Exception();
                var p1 = iter.Current;
                if (!iter.MoveNext()) throw new Exception();
                var p2 = iter.Current;
                yield return new Triangle3D(p0, p1, p2);
            }
        }
        internal static IEnumerable<TriIndex> ExtractTriangles(IEnumerable<int> indices)
        {
            var iter = indices.GetEnumerator();
            while (iter.MoveNext())
            {
                var i0 = iter.Current;
                if (!iter.MoveNext()) throw new Exception();
                var i1 = iter.Current;
                if (!iter.MoveNext()) throw new Exception();
                var i2 = iter.Current;
                yield return new TriIndex(i0, i1, i2);
            }
        }
    }
}