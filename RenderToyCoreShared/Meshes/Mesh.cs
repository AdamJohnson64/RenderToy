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
        public readonly Vector3D[] Vertices;
        public readonly int[] Triangles;
    }
}