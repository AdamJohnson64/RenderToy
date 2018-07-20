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
using RenderToy.Math;

namespace RenderToy.ModelFormat
{
    public static class LoaderOBJ
    {
        public static IEnumerable<INode> LoadFromPath(string path)
        {
            var vertices = new List<Vector3D>();
            var normals = new List<Vector3D>();
            var texcoords = new List<Vector2D>();
            var tangents = new List<Vector3D>();
            var bitangents = new List<Vector3D>();
            var collectedvertexfaces = new List<int>();
            var collectednormalfaces = new List<int>();
            var collectedtexcoordfaces = new List<int>();
            string groupname = null;
            string materialname = null;
            int smoothinggroup = -1;
            var materials = new Dictionary<string, IMaterial>();
            Func<Mesh> FlushMesh = () =>
            {
                // Flush this mesh to the caller.
                var primitive = new Mesh();
                primitive.Vertices = new MeshChannel<Vector3D>(vertices, collectedvertexfaces);
                primitive.Normals = new MeshChannel<Vector3D>(normals, collectednormalfaces);
                primitive.TexCoords = new MeshChannel<Vector2D>(texcoords, collectedtexcoordfaces);
                primitive.GenerateTangentSpace();
                // Reset our state.
                collectedvertexfaces = new List<int>();
                collectednormalfaces = new List<int>();
                collectedtexcoordfaces = new List<int>();
                return primitive;
            };
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
                        if (parts.Length < 3) throw new FileLoadException("Malformed texture coordinate '" + line + "'.");
                        var v = new Vector2D();
                        v.X = double.Parse(parts[1]);
                        // OpenGL texture coordinates :(
                        v.Y = 1 - double.Parse(parts[2]);
                        texcoords.Add(v);
                        continue;
                    }
                    if (line.StartsWith("g "))
                    {
                        groupname = line.Substring(2);
                        continue;
                    }
                    if (line == "g")
                    {
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
                            yield return new Node(materialname, new TransformMatrix(Matrix3D.Identity), FlushMesh(), StockMaterials.White, materials[materialname]);
                            materialname = null;
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
                        var f = parts.Skip(1).Select(i => i.Split(new char[] { '/' })).ToArray();
                        if (!f.All(i => i.Length == f[0].Length)) throw new FileLoadException("Inconsistent face setups '" + line + "'.");
                        if (f[0].Length < 1 || f[0].Length > 3) throw new FileLoadException("Bad face component count '" + line + "'.");
                        int[] idxv = f.Select(i => int.Parse(i[0]) - 1).ToArray();
                        int[] idxn = f.Select(i => int.Parse(i[2]) - 1).ToArray();
                        int[] idxt = f.Select(i => int.Parse(i[1]) - 1).ToArray();
                        Action<int, int, int> FlushFace = (int i0, int i1, int i2) =>
                        {
                            collectedvertexfaces.Add(idxv[i0]);
                            collectedvertexfaces.Add(idxv[i1]);
                            collectedvertexfaces.Add(idxv[i2]);
                            collectednormalfaces.Add(idxn[i0]);
                            collectednormalfaces.Add(idxn[i1]);
                            collectednormalfaces.Add(idxn[i2]);
                            collectedtexcoordfaces.Add(idxt[i0]);
                            collectedtexcoordfaces.Add(idxt[i1]);
                            collectedtexcoordfaces.Add(idxt[i2]);
                        };
                        for (int fan = 0; fan < parts.Length - 3; ++fan)
                        {
                            FlushFace(0, fan + 1, fan + 2);
                        }
                        continue;
                    }
                    throw new FileLoadException("Unknown tag '" + line + "'.");
                }
            }
            if (materialname != null)
            {
                yield return new Node(materialname, new TransformMatrix(Matrix3D.Identity), FlushMesh(), StockMaterials.White, materials[materialname]);
                materialname = null;
            }
            else
            {
                yield return new Node("NoMaterial", new TransformMatrix(Matrix3D.Identity), FlushMesh(), StockMaterials.White, StockMaterials.PlasticWhite);
                materialname = null;
            }
        }
        static Dictionary<string, IMaterial> LoadMaterialLibrary(string objpath, string mtlrelative)
        {
            var materialbyname = new Dictionary<string, IMaterial>();
            var texturebyname = new Dictionary<string, IMaterial>();
            var objdir = Path.GetDirectoryName(objpath);
            var mtlfile = Path.Combine(objdir, mtlrelative);
            IMaterial map_Ka = null;
            IMaterial map_Kd = null;
            IMaterial map_Ks = null;
            IMaterial map_d = null;
            IMaterial map_bump = null;
            IMaterial bump = null;
            Func<string, IMaterial> LoadUniqueTexture = (string name) =>
            {
                IMaterial found = null;
                if (texturebyname.TryGetValue(name, out found)) return found;
                var texfile = Path.Combine(objdir, name);
                var image = LoaderImage.LoadFromPath(texfile);
                return texturebyname[name] = image == null ? null : new Texture(name, image, true);
            };
            using (var streamreader = File.OpenText(mtlfile))
            {
                string line;
                string materialname = null;
                Action FLUSHMATERIAL = () =>
                {
                    if (materialname == null) return;
                    materialbyname.Add(materialname, new OBJMaterial { Name = materialname, map_Ka = map_Ka, map_Kd = map_Kd, map_d = map_d, map_bump = map_bump, bump = bump });
                    materialname = null;
                    map_Ka = null;
                    map_Kd = null;
                    map_Ks = null;
                    map_d = null;
                    map_bump = null;
                    bump = null;
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
                        map_Ka = LoadUniqueTexture(line.Substring(find + 1));
                        continue;
                    }
                    if (line.StartsWith("map_Kd "))
                    {
                        int find = line.IndexOf(' ');
                        map_Kd = LoadUniqueTexture(line.Substring(find + 1));
                        continue;
                    }
                    if (line.StartsWith("map_Ks "))
                    {
                        int find = line.IndexOf(' ');
                        map_Ks = LoadUniqueTexture(line.Substring(find + 1));
                        continue;
                    }
                    if (line.StartsWith("map_d "))
                    {
                        int find = line.IndexOf(' ');
                        map_d = LoadUniqueTexture(line.Substring(find + 1));
                        continue;
                    }
                    if (line.StartsWith("map_bump "))
                    {
                        int find = line.IndexOf(' ');
                        map_bump = LoadUniqueTexture(line.Substring(find + 1));
                        continue;
                    }
                    if (line.StartsWith("bump "))
                    {
                        int find = line.IndexOf(' ');
                        bump = LoadUniqueTexture(line.Substring(find + 1));
                        continue;
                    }
                    throw new FileLoadException("Unknown tag '" + line + "'.");
                }
                FLUSHMATERIAL();
            }
            return materialbyname;
        }
        public class OBJMaterial : ITexture, INamed
        {
            public string Name
            {
                get { return _name; }
                set { _name = value; }
            }
            public IMaterial map_Ka
            {
                get { return _map_Ka; }
                set { _map_Ka = value; }
            }
            public IMaterial map_Kd
            {
                get { return _map_Kd; }
                set { _map_Kd = value; }
            }
            public IMaterial map_d
            {
                get { return _map_d; }
                set { _map_d = value; }
            }
            public IMaterial map_bump
            {
                get { return _map_bump; }
                set { _map_bump = value; }
            }
            public IMaterial bump
            {
                get { return _bump; }
                set { _bump = value; }
            }
            public IMaterial displacement
            {
                get { return _displacement; }
                set { _displacement = value; }
            }
            public string GetName()
            {
                return _name;
            }
            public bool IsConstant()
            {
                return
                    (_map_Ka == null ? true : _map_Ka.IsConstant()) &&
                    (_map_Kd == null ? true : _map_Kd.IsConstant()) &&
                    (_map_d == null ? true : _map_d.IsConstant()) &&
                    (_map_bump == null ? true : _map_bump.IsConstant()) &&
                    (_bump == null ? true : _bump.IsConstant());
            }
            public int GetTextureLevelCount()
            {
                var kd = _map_Kd as ITexture;
                return kd == null ? 0 : kd.GetTextureLevelCount();
            }
            public IImageBgra32 GetTextureLevel(int level)
            {
                var kd = _map_Kd as ITexture;
                return kd == null ? null : kd.GetTextureLevel(level);
            }
            string _name;
            IMaterial _map_Ka;
            IMaterial _map_Kd;
            IMaterial _map_d;
            IMaterial _map_bump;
            IMaterial _bump;
            IMaterial _displacement;
        }
    }
}