////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Materials;
using RenderToy.Meshes;
using RenderToy.SceneGraph;
using RenderToy.Textures;
using RenderToy.TextureFormats;
using RenderToy.Transforms;
using RenderToy.Utility;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace RenderToy.ModelFormat
{
    static class LoaderOBJ
    {
        public static IEnumerable<INode> LoadFromPath(string path)
        {
            var vertices = new List<Vector3D>();
            var normals = new List<Vector3D>();
            var texcoords = new List<Vector2D>();
            var collectedfaces = new List<int>();
            var collectedtexcoordfaces = new List<int>();
            string groupname = null;
            string materialname = null;
            int smoothinggroup = -1;
            var materials = new Dictionary<string, IMaterial>();
            using (var stream = File.OpenText(path))
            {
                string line;
                while ((line = stream.ReadLine()) != null)
                {
                    line = line.Trim();
                    if (line.StartsWith("#")) continue;
                    if (line.Length == 0) continue;
                    if (line.StartsWith("mtllib "))
                    {
                        int splitat = line.IndexOf(' ');
                        string materialfile = line.Substring(splitat + 1);
                        materials = LoadMaterialLibrary(path, materialfile);
                        continue;
                    }
                    if (line.StartsWith("v "))
                    {
                        var parts = line.Split(new char[] { ' ' }, System.StringSplitOptions.RemoveEmptyEntries);
                        if (parts.Length != 4) throw new FileLoadException("Malformed vertex '" + line + "'.");
                        var v = new Vector3D();
                        v.X = double.Parse(parts[1]);
                        v.Y = double.Parse(parts[2]);
                        v.Z = double.Parse(parts[3]);
                        vertices.Add(v);
                        continue;
                    }
                    if (line.StartsWith("vn "))
                    {
                        var parts = line.Split(new char[] { ' ' }, System.StringSplitOptions.RemoveEmptyEntries);
                        if (parts.Length != 4) throw new FileLoadException("Malformed normal '" + line + "'.");
                        var v = new Vector3D();
                        v.X = double.Parse(parts[1]);
                        v.Y = double.Parse(parts[2]);
                        v.Z = double.Parse(parts[3]);
                        normals.Add(v);
                        continue;
                    }
                    if (line.StartsWith("vt "))
                    {
                        var parts = line.Split(new char[] { ' ' }, System.StringSplitOptions.RemoveEmptyEntries);
                        if (parts.Length != 4) throw new FileLoadException("Malformed texture coordinate '" + line + "'.");
                        var v = new Vector2D();
                        v.X = double.Parse(parts[1]);
                        // OpenGL texture coordinates :(
                        v.Y = 1 - double.Parse(parts[2]);
                        texcoords.Add(v);
                        continue;
                    }
                    if (line.StartsWith("g "))
                    {
                        var parts = line.Split(new char[] { ' ' }, System.StringSplitOptions.RemoveEmptyEntries);
                        if (parts.Length != 2) throw new FileLoadException("Malformed group '" + line + "'.");
                        groupname = parts[1];
                        continue;
                    }
                    if (line.StartsWith("s "))
                    {
                        var parts = line.Split(new char[] { ' ' }, System.StringSplitOptions.RemoveEmptyEntries);
                        if (parts.Length != 2) throw new FileLoadException("Malformed smoothing group '" + line + "'.");
                        if (parts[1] == "off")
                        {
                            smoothinggroup = -1;
                            continue;
                        }
                        smoothinggroup = int.Parse(parts[1]);
                        continue;
                    }
                    if (line.StartsWith("usemtl "))
                    {
                        if (materialname != null)
                        {
                            // Flush this mesh to the caller.
                            var flatvertices = collectedfaces.Select(i => vertices[i]);
                            var flattexcoords = collectedtexcoordfaces.Select(i => texcoords[i]);
                            var flatindices = GenerateIntegerSequence(collectedfaces.Count);
                            var primitive = new Mesh(flatvertices, flatindices, flattexcoords);
                            yield return new Node(materialname, new TransformMatrix(Matrix3D.Identity), primitive, StockMaterials.White, materials[materialname]);
                            // Reset our state.
                            materialname = null;
                            collectedfaces.Clear();
                            collectedtexcoordfaces.Clear();
                        }
                        var parts = line.Split(new char[] { ' ' }, System.StringSplitOptions.RemoveEmptyEntries);
                        if (parts.Length != 2) throw new FileLoadException("Malformed usemtl '" + line + "'.");
                        materialname = parts[1];
                        continue;
                    }
                    if (line.StartsWith("f "))
                    {
                        var parts = line.Split(new char[] { ' ' }, System.StringSplitOptions.RemoveEmptyEntries);
                        if (parts.Length < 4) throw new FileLoadException("Insufficient indices '" + line + "'.");
                        if (parts.Length > 5) throw new FileLoadException("Too many indices '" + line + "'.");
                        var f = parts.Skip(1).Select(i => i.Split(new char[] { '/' })).ToArray();
                        if (!f.All(i => i.Length == f[0].Length)) throw new FileLoadException("Inconsistent face setups '" + line + "'.");
                        if (f[0].Length < 1 || f[0].Length > 3) throw new FileLoadException("Bad face component count '" + line + "'.");
                        int[] idxv = f.Select(i => int.Parse(i[0]) - 1).ToArray();
                        int[] idxt = f.Select(i => int.Parse(i[1]) - 1).ToArray();
                        int[] idxn = f.Select(i => int.Parse(i[2]) - 1).ToArray();
                        if (parts.Length == 4)
                        {
                            collectedfaces.Add(idxv[0]);
                            collectedfaces.Add(idxv[1]);
                            collectedfaces.Add(idxv[2]);
                            collectedtexcoordfaces.Add(idxt[0]);
                            collectedtexcoordfaces.Add(idxt[1]);
                            collectedtexcoordfaces.Add(idxt[2]);
                        }
                        if (parts.Length == 5)
                        {
                            collectedfaces.Add(idxv[0]);
                            collectedfaces.Add(idxv[1]);
                            collectedfaces.Add(idxv[2]);
                            collectedfaces.Add(idxv[2]);
                            collectedfaces.Add(idxv[3]);
                            collectedfaces.Add(idxv[0]);
                            collectedtexcoordfaces.Add(idxt[0]);
                            collectedtexcoordfaces.Add(idxt[1]);
                            collectedtexcoordfaces.Add(idxt[2]);
                            collectedtexcoordfaces.Add(idxt[2]);
                            collectedtexcoordfaces.Add(idxt[3]);
                            collectedtexcoordfaces.Add(idxt[0]);
                        }
                        continue;
                    }
                    throw new FileLoadException("Unknown tag '" + line + "'.");
                }
            }
            if (materialname != null)
            {
                // Flush this mesh to the caller.
                var primitive = new Mesh(vertices, collectedfaces);
                yield return new Node(materialname, new TransformMatrix(Matrix3D.Identity), primitive, StockMaterials.White, materials[materialname]);
                // Reset our state.
                materialname = null;
                collectedfaces.Clear();
                collectedtexcoordfaces.Clear();
            }
        }
        static Dictionary<string, IMaterial> LoadMaterialLibrary(string objpath, string mtlrelative)
        {
            var result = new Dictionary<string, IMaterial>();
            var objdir = Path.GetDirectoryName(objpath);
            var mtlfile = Path.Combine(objdir, mtlrelative);
            IMaterial map_Ka = null;
            using (var streamreader = File.OpenText(mtlfile))
            {
                string line;
                string materialname = null;
                Action FLUSHMATERIAL = () =>
                {
                    if (materialname == null) return;
                    result.Add(materialname, map_Ka);
                    materialname = null;
                    map_Ka = null;
                };
                while ((line = streamreader.ReadLine()) != null)
                {
                    line = line.Trim();
                    if (line.StartsWith("#")) continue;
                    if (line.Length == 0) continue;
                    if (line.StartsWith("newmtl "))
                    {
                        FLUSHMATERIAL();
                        int find = line.IndexOf(' ');
                        materialname = line.Substring(find + 1);
                        continue;
                    }
                    if (line.StartsWith("Ns "))
                    {
                        continue;
                    }
                    if (line.StartsWith("Ni "))
                    {
                        continue;
                    }
                    if (line.StartsWith("d "))
                    {
                        continue;
                    }
                    if (line.StartsWith("Tr "))
                    {
                        continue;
                    }
                    if (line.StartsWith("Tf "))
                    {
                        continue;
                    }
                    if (line.StartsWith("illum "))
                    {
                        continue;
                    }
                    if (line.StartsWith("illum "))
                    {
                        continue;
                    }
                    if (line.StartsWith("Ka "))
                    {
                        continue;
                    }
                    if (line.StartsWith("Kd "))
                    {
                        continue;
                    }
                    if (line.StartsWith("Ks "))
                    {
                        continue;
                    }
                    if (line.StartsWith("Ke "))
                    {
                        continue;
                    }
                    if (line.StartsWith("map_Ka "))
                    {
                        int find = line.IndexOf(' ');
                        map_Ka = LoadTexture(mtlfile, line.Substring(find + 1));
                        continue;
                    }
                    if (line.StartsWith("map_Kd "))
                    {
                        continue;
                    }
                    if (line.StartsWith("map_d "))
                    {
                        continue;
                    }
                    if (line.StartsWith("map_bump "))
                    {
                        continue;
                    }
                    if (line.StartsWith("bump "))
                    {
                        continue;
                    }
                    throw new FileLoadException("Unknown tag '" + line + "'.");
                }
                FLUSHMATERIAL();
            }
            return result;
        }
        static IMaterial LoadTexture(string mtlpath, string texrelative)
        {
            var mtldir = Path.GetDirectoryName(mtlpath);
            var texfile = Path.Combine(mtldir, texrelative);
            if (!File.Exists(texfile)) return null;
            return new Texture(texrelative, LoaderTGA.LoadFromPath(texfile), true);
        }
        static IEnumerable<int> GenerateIntegerSequence(int count)
        {
            for (int i = 0; i < count; ++i)
            {
                yield return i;
            }
        }
    }
}