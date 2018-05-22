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
    public class MeshChannel<DATATYPE>
    {
        public MeshChannel(IReadOnlyList<DATATYPE> vertices, IReadOnlyList<int> indices)
        {
            Vertices = vertices;
            Indices = indices;
        }
        public IReadOnlyList<DATATYPE> GetVertices()
        {
            return Vertices;
        }
        public IReadOnlyList<int> GetIndices()
        {
            return Indices;
        }
        readonly IReadOnlyList<DATATYPE> Vertices;
        readonly IReadOnlyList<int> Indices;
    }
    /// <summary>
    /// Triangle-only mesh.
    /// </summary>
    public class Mesh : IPrimitive
    {
        public Mesh(IEnumerable<int> triangles, IEnumerable<Vector3D> vertices)
        {
            Vertices = new MeshChannel<Vector3D>(vertices.ToArray(), triangles.ToArray());
        }
        public Mesh(IEnumerable<int> triangles, IEnumerable<Vector3D> vertices, IEnumerable<Vector3D> normals, IEnumerable<Vector2D> texcoords, IEnumerable<Vector3D> tangents, IEnumerable<Vector3D> bitangents)
        {
            var trianglesarray = triangles.ToArray();
            Vertices = new MeshChannel<Vector3D>(vertices.ToArray(), trianglesarray);
            Normals = new MeshChannel<Vector3D>(normals.ToArray(), trianglesarray);
            TexCoords = new MeshChannel<Vector2D>(texcoords.ToArray(), trianglesarray);
            Tangents = new MeshChannel<Vector3D>(tangents.ToArray(), trianglesarray);
            Bitangents = new MeshChannel<Vector3D>(bitangents.ToArray(), trianglesarray);
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
        public readonly MeshChannel<Vector3D> Vertices = null;
        public readonly MeshChannel<Vector3D> Normals = null;
        public readonly MeshChannel<Vector2D> TexCoords = null;
        public readonly MeshChannel<Vector3D> Tangents = null;
        public readonly MeshChannel<Vector3D> Bitangents = null;
    }
}