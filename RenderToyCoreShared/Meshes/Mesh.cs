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
        public Mesh(IEnumerable<int> triangles, IEnumerable<Vector3D> vertices)
        {
            Triangles = triangles.ToArray();
            Vertices = vertices.ToArray();
        }
        public Mesh(IEnumerable<int> triangles, IEnumerable<Vector3D> vertices, IEnumerable<Vector3D> normals, IEnumerable<Vector2D> texcoords, IEnumerable<Vector3D> tangents, IEnumerable<Vector3D> bitangents)
        {
            Triangles = triangles.ToArray();
            Vertices = vertices.ToArray();
            Normals = normals.ToArray();
            TexCoords = texcoords.ToArray();
            Tangents = tangents.ToArray();
            Bitangents = bitangents.ToArray();
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
            return new Mesh(indices, vertices);
        }
        public readonly int[] Triangles = null;
        public readonly Vector3D[] Vertices = null;
        public readonly Vector3D[] Normals = null;
        public readonly Vector2D[] TexCoords = null;
        public readonly Vector3D[] Tangents = null;
        public readonly Vector3D[] Bitangents = null;
    }
}