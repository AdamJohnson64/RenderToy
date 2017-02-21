////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System;
using System.Collections.Generic;
using System.IO;

namespace RenderToy
{
    public class MeshPLY
    {
        public static Mesh LoadFromPath(string path)
        {
            using (StreamReader streamreader = File.OpenText(path))
            {
                return LoadFromStream(streamreader);
            }
        }
        static Mesh LoadFromStream(StreamReader streamreader)
        {
            // HACK: This isn't remotely generic but it's absolutely deliberate.
            // This format will read the Stanford Bunny (bunny_zipper_res4.ply) in the online scan library.
            if (streamreader.ReadLine() != "ply") throw new Exception("Expected 'ply'.");
            if (streamreader.ReadLine() != "format ascii 1.0") throw new Exception("Expected 'format ascii 1.0'.");
            if (streamreader.ReadLine() != "comment zipper output") throw new Exception("Expected 'comment zipper output'.");
            int numvertex;
            {
                string line = streamreader.ReadLine();
                if (!line.StartsWith("element vertex ")) throw new Exception("Expected 'element vertex <n>'.");
                numvertex = int.Parse(line.Substring(15));
            }
            if (streamreader.ReadLine() != "property float x") throw new Exception("Expected 'property float x'.");
            if (streamreader.ReadLine() != "property float y") throw new Exception("Expected 'property float y'.");
            if (streamreader.ReadLine() != "property float z") throw new Exception("Expected 'property float z'.");
            if (streamreader.ReadLine() != "property float confidence") throw new Exception("Expected 'property float confidence'.");
            if (streamreader.ReadLine() != "property float intensity") throw new Exception("Expected 'property float intensity'.");
            int numface;
            {
                string line = streamreader.ReadLine();
                if (!line.StartsWith("element face ")) throw new Exception("Expected 'element face <n>'.");
                numface = int.Parse(line.Substring(13));
            }
            if (streamreader.ReadLine() != "property list uchar int vertex_indices") throw new Exception("Expected 'property list uchar int vertex_indices'.");
            if (streamreader.ReadLine() != "end_header") throw new Exception("Expected 'end_header'.");
            var vertices = new List<Point3D>();
            var triangles = new List<TriIndex>();
            for (int v = 0; v < numvertex; ++v)
            {
                string line = streamreader.ReadLine();
                string[] parts = line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                vertices.Add(new Point3D(double.Parse(parts[0]), double.Parse(parts[1]), double.Parse(parts[2])));
            }
            for (int t = 0; t < numface; ++t)
            {
                string line = streamreader.ReadLine();
                string[] parts = line.Split(new[] { ' ' }, StringSplitOptions.RemoveEmptyEntries);
                if (parts[0] != "3") throw new Exception("Expected '3'.");
                triangles.Add(new TriIndex(int.Parse(parts[1]), int.Parse(parts[2]), int.Parse(parts[3])));
            }
            //return new MeshBVH(vertices.ToArray(), triangles.ToArray());
            return new Mesh(vertices.ToArray(), triangles.ToArray());
        }
    }
}