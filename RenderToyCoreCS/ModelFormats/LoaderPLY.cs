﻿////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Meshes;
using RenderToy.Primitives;
using RenderToy.Utility;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace RenderToy.ModelFormat
{
    public static partial class LoaderPLY
    {
        public static IPrimitive LoadFromPath(string path)
        {
            return LoadFromPath(path, (v,i) => new Mesh(v, i));
        }
        public static IPrimitive LoadBVHFromPath(string path)
        {
            return LoadFromPath(path, (v,i) => MeshBVH.Create(Mesh.FlattenIndices(v.ToArray(), i.ToArray()).ToArray()));
        }
        delegate IPrimitive ConditionMesh(IReadOnlyList<Vector3D> vertices, IReadOnlyList<TriIndex> triangles);
        static IPrimitive LoadFromPath(string path, ConditionMesh conditioner)
        {
            using (StreamReader streamreader = File.OpenText(path))
            {
                return LoadFromStream(streamreader, conditioner);
            }
        }
        static IPrimitive LoadFromStream(StreamReader streamreader, ConditionMesh conditioner)
        {
            string line;
            // HACK: This isn't remotely generic but it's absolutely deliberate.
            // This format will read the Stanford Bunny (bunny_zipper_res4.ply) in the online scan library.
            if (streamreader.ReadLine() != "ply") throw new Exception("Expected 'ply'.");
            if (streamreader.ReadLine() != "format ascii 1.0") throw new Exception("Expected 'format ascii 1.0'.");
            {
                line = streamreader.ReadLine();
                if (!line.StartsWith("comment ")) throw new Exception("Expected 'comment <txt>'.");
            }
            int numvertex;
            {
                line = streamreader.ReadLine();
                if (!line.StartsWith("element vertex ")) throw new Exception("Expected 'element vertex <n>'.");
                numvertex = int.Parse(line.Substring(15));
            }
            int indexx = -1, indexy = -1, indexz = -1;
            {
                line = streamreader.ReadLine();
                int index = -1;
                while (line.StartsWith("property "))
                {
                    ++index;
                    if (line == "property float x") indexx = index;
                    if (line == "property float y") indexy = index;
                    if (line == "property float z") indexz = index;
                    line = streamreader.ReadLine();
                }
            }
            int numface;
            {
                if (!line.StartsWith("element face ")) throw new Exception("Expected 'element face <n>'.");
                numface = int.Parse(line.Substring(13));
            }
            if (streamreader.ReadLine() != "property list uchar int vertex_indices") throw new Exception("Expected 'property list uchar int vertex_indices'.");
            if (streamreader.ReadLine() != "end_header") throw new Exception("Expected 'end_header'.");
            var vertices = new List<Vector3D>();
            var indices = new List<TriIndex>();
            for (int v = 0; v < numvertex; ++v)
            {
                line = streamreader.ReadLine();
                string[] parts = line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                vertices.Add(new Vector3D(double.Parse(parts[indexx]), double.Parse(parts[indexy]), double.Parse(parts[indexz])));
            }
            for (int t = 0; t < numface; ++t)
            {
                line = streamreader.ReadLine();
                string[] parts = line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                if (parts[0] != "3") throw new Exception("Expected '3'.");
                indices.Add(new TriIndex(int.Parse(parts[1]), int.Parse(parts[2]), int.Parse(parts[3])));
            }
            return conditioner(vertices, indices);
        }
    }
}