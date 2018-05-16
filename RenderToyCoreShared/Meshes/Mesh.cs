////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Primitives;
using RenderToy.Utility;
using System.Collections.Generic;
using System.Linq;

namespace RenderToy.Meshes
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
        public static IEnumerable<Triangle3D> FlattenIndices(IReadOnlyList<Vector3D> vertices, IEnumerable<TriIndex> triangles)
        {
            return triangles.Select(t => new Triangle3D(vertices[t.Index0], vertices[t.Index1], vertices[t.Index2]));
        }
        public readonly Vector3D[] Vertices;
        public readonly TriIndex[] Triangles;
    }
}