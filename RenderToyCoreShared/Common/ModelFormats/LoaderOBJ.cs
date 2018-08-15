////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Materials;
using RenderToy.Meshes;
using RenderToy.SceneGraph;
using RenderToy.Textures;
using RenderToy.Transforms;
using RenderToy.Utility;
using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using RenderToy.Math;
using System.Threading.Tasks;
using System.Collections.Concurrent;

namespace RenderToy.ModelFormat
{
    public static class LoaderOBJ
    {
        public static async Task<INode> LoadFromPathAsync(string path)
        {
            var loadednodes = new List<INode>();
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
            IDictionary<string, Task<IMaterial>> materials = new ConcurrentDictionary<string, Task<IMaterial>>();
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
                        materials = await LoadMaterialLibrary(path, materialfile);
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
                            loadednodes.Add(new Node(materialname, new TransformMatrix(Matrix3D.Identity), FlushMesh(), StockMaterials.White, await materials[materialname]));
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
                loadednodes.Add(new Node(materialname, new TransformMatrix(Matrix3D.Identity), FlushMesh(), StockMaterials.White, await materials[materialname]));
                materialname = null;
            }
            else
            {
                loadednodes.Add(new Node("NoMaterial", new TransformMatrix(Matrix3D.Identity), FlushMesh(), StockMaterials.White, StockMaterials.PlasticWhite));
                materialname = null;
            }
            var root = new Node(path, new TransformMatrix(Matrix3D.Identity), null, StockMaterials.Black, null);
            root.children.AddRange(loadednodes);
            return root;
        }
        class TextureLoader
        {
            public TextureLoader(string root)
            {
                Root = root;
            }
            public async Task<IMaterial> LoadTexture(string name)
            {
                Task<IMaterial> found = null;
                if (texturebyname.TryGetValue(name, out found)) return await found;
                var loadit = LoadImage(Path.Combine(Root, name));
                texturebyname[name] = loadit;
                return await loadit;
            }
            public async Task<IMaterial> LoadImage(string name)
            {
                return Texture.Create(name, await LoaderImage.LoadFromPathAsync(name), true);
            }
            string Root;
            Dictionary<string, Task<IMaterial>> texturebyname = new Dictionary<string, Task<IMaterial>>();
        }
        class MaterialLoader
        {
            public string materialname = "None";
            public Task<IMaterial> map_Ka = nullreturn;
            public Task<IMaterial> map_Kd = nullreturn;
            public Task<IMaterial> map_Ks = nullreturn;
            public Task<IMaterial> map_d = nullreturn;
            public Task<IMaterial> map_bump = nullreturn;
            public Task<IMaterial> bump = nullreturn;
            static Task<IMaterial> nullreturn = Task.FromResult<IMaterial>(null);
        }
        static async Task<IDictionary<string, Task<IMaterial>>> LoadMaterialLibrary(string objpath, string mtlrelative)
        {
            var materialbyname = new ConcurrentDictionary<string, Task<IMaterial>>();
            var objdir = Path.GetDirectoryName(objpath);
            var mtlfile = Path.Combine(objdir, mtlrelative);
            var textureloader = new TextureLoader(Path.GetDirectoryName(objpath));
            MaterialLoader buildmaterial = null;
            using (var streamreader = File.OpenText(mtlfile))
            {
                string line;
                while ((line = streamreader.ReadLine()) != null)
                {
                    line = line.Trim();
                    if (line.StartsWith("#")) continue;
                    if (line.Length == 0) continue;
                    if (line.StartsWith("newmtl "))
                    {
                        if (buildmaterial != null)
                        {
                            var buildthis = buildmaterial;
                            var material = Task.Run(async () => (IMaterial)new OBJMaterial { Name = buildthis.materialname, map_Ka = await buildthis.map_Ka, map_Kd = await buildthis.map_Kd, map_d = await buildthis.map_d, map_bump = await buildthis.map_bump, bump = await buildthis.bump });
                            materialbyname.TryAdd(buildmaterial.materialname, material);
                        }
                        buildmaterial = new MaterialLoader();
                        int find = line.IndexOf(' ');
                        buildmaterial.materialname = line.Substring(find + 1);
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
                        if (buildmaterial == null) throw new InvalidDataException();
                        int find = line.IndexOf(' ');
                        buildmaterial.map_Ka = textureloader.LoadTexture(line.Substring(find + 1));
                        continue;
                    }
                    if (line.StartsWith("map_Kd "))
                    {
                        if (buildmaterial == null) throw new InvalidDataException();
                        int find = line.IndexOf(' ');
                        buildmaterial.map_Kd = textureloader.LoadTexture(line.Substring(find + 1));
                        continue;
                    }
                    if (line.StartsWith("map_Ks "))
                    {
                        if (buildmaterial == null) throw new InvalidDataException();
                        int find = line.IndexOf(' ');
                        buildmaterial.map_Ks = textureloader.LoadTexture(line.Substring(find + 1));
                        continue;
                    }
                    if (line.StartsWith("map_d "))
                    {
                        if (buildmaterial == null) throw new InvalidDataException();
                        int find = line.IndexOf(' ');
                        buildmaterial.map_d = textureloader.LoadTexture(line.Substring(find + 1));
                        continue;
                    }
                    if (line.StartsWith("map_bump "))
                    {
                        if (buildmaterial == null) throw new InvalidDataException();
                        int find = line.IndexOf(' ');
                        buildmaterial.map_bump = textureloader.LoadTexture(line.Substring(find + 1));
                        continue;
                    }
                    if (line.StartsWith("bump "))
                    {
                        if (buildmaterial == null) throw new InvalidDataException();
                        int find = line.IndexOf(' ');
                        buildmaterial.bump = textureloader.LoadTexture(line.Substring(find + 1));
                        continue;
                    }
                    throw new FileLoadException("Unknown tag '" + line + "'.");
                }
                if (buildmaterial != null)
                {
                    var buildthis = buildmaterial;
                    var material = Task.Run(async () => (IMaterial)new OBJMaterial { Name = buildthis.materialname, map_Ka = await buildthis.map_Ka, map_Kd = await buildthis.map_Kd, map_d = await buildthis.map_d, map_bump = await buildthis.map_bump, bump = await buildthis.bump });
                    materialbyname.TryAdd(buildmaterial.materialname, material);
                    buildmaterial = null;
                }
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
            public int GetTextureArrayCount()
            {
                return 1;
            }
            public int GetTextureLevelCount()
            {
                var kd = _map_Kd as ITexture;
                return kd == null ? 0 : kd.GetTextureLevelCount();
            }
            public IImageBgra32 GetSurface(int array, int level)
            {
                var kd = _map_Kd as ITexture;
                return kd == null ? null : kd.GetSurface(array, level);
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