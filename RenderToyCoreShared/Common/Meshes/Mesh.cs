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
        public Mesh()
        {
        }
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
        public void GenerateTangentSpace()
        {
            var vpos = Vertices.GetVertices();
            var ipos = Vertices.GetIndices();
            var vtex = TexCoords.GetVertices();
            var itex = TexCoords.GetIndices();
            var indexcount = ipos.Count;
            var doublelookup =
                SequenceHelp.GenerateIntegerSequence(indexcount)
                .Select(i => new { VertexIndex = ipos[i], TexCoordIndex = itex[i] });
            var triangles = SequenceHelp.Split3(doublelookup);
            var tangents = new List<Vector3D>();
            var bitangents = new List<Vector3D>();
            var collectedtangentfaces = new List<int>();
            var collectedbitangentfaces = new List<int>();
            foreach (var t in triangles)
            {
                // Compute tangent and bitangent.
                var P0 = vpos[t.Item1.VertexIndex];
                var P1 = vpos[t.Item2.VertexIndex];
                var P2 = vpos[t.Item3.VertexIndex];
                var T0 = vtex[t.Item1.TexCoordIndex];
                var T1 = vtex[t.Item2.TexCoordIndex];
                var T2 = vtex[t.Item3.TexCoordIndex];
                var m = MathHelp.Invert(new Matrix2D(T1.X - T0.X, T2.X - T0.X, T1.Y - T0.Y, T2.Y - T0.Y));
                var solve00 = MathHelp.Transform(m, new Vector2D(0 - T0.X, 0 - T0.Y));
                var solve10 = MathHelp.Transform(m, new Vector2D(1 - T0.X, 0 - T0.Y));
                var solve01 = MathHelp.Transform(m, new Vector2D(0 - T0.X, 1 - T0.Y));
                var udir = MathHelp.Normalized(solve10.X * (P1 - P0) + solve10.Y * (P2 - P0));
                var vdir = MathHelp.Normalized(solve01.X * (P1 - P0) + solve01.Y * (P2 - P0));
                int ti0 = tangents.Count;
                tangents.Add(udir);
                collectedtangentfaces.Add(ti0);
                collectedtangentfaces.Add(ti0);
                collectedtangentfaces.Add(ti0);
                int bi0 = bitangents.Count;
                bitangents.Add(vdir);
                collectedbitangentfaces.Add(bi0);
                collectedbitangentfaces.Add(bi0);
                collectedbitangentfaces.Add(bi0);
            }
            Tangents = new MeshChannel<Vector3D>(tangents, collectedtangentfaces);
            Bitangents = new MeshChannel<Vector3D>(bitangents, collectedbitangentfaces);
        }
        public MeshChannel<Vector3D> Vertices = null;
        public MeshChannel<Vector3D> Normals = null;
        public MeshChannel<Vector2D> TexCoords = null;
        public MeshChannel<Vector3D> Tangents = null;
        public MeshChannel<Vector3D> Bitangents = null;
    }
}