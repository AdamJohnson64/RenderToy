////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;

namespace RenderToy
{
    public class SceneFormatter
    {
        public static byte[] CreateFlatMemoryF32(Scene scene)
        {
            return new SceneFormatter(scene, false).m.ToArray();
        }
        public static byte[] CreateFlatMemoryF64(Scene scene)
        {
            return new SceneFormatter(scene, true).m.ToArray();
        }
        public static byte[] CreateFlatMemoryF32(Matrix3D obj)
        {
            var memory = new MemoryStream();
            using (var stream = new BinaryWriter(memory))
            {
                Serialize(obj, v => stream.Write((float)v));
            }
            return memory.ToArray();
        }
        public static byte[] CreateFlatMemoryF64(Matrix3D obj)
        {
            var memory = new MemoryStream();
            using (var stream = new BinaryWriter(memory))
            {
                Serialize(obj, v => stream.Write((double)v));
            }
            return memory.ToArray();
        }
        SceneFormatter(Scene scene, bool use_f64)
        {
            UseF64 = use_f64;
            // Prepare the scene definition for the renderer.
            using (binarywriter = new BinaryWriter(m))
            {
                var transformedobjects = TransformedObject.Enumerate(scene).ToList();
                // Write the file size and object count.
                binarywriter.Write((int)0);
                binarywriter.Write((int)transformedobjects.Count);
                binarywriter.Write((int)0);
                binarywriter.Write((int)0);
                // Write all the objects.
                foreach (var obj in transformedobjects)
                {
                    Serialize(obj);
                }
                // Write all the outstanding queued object references and patch their reference sources.
                foreach (var writeremaining in WriteObjects)
                {
                    // Always pad additional objects to 16 bytes for safety.
                    // We don't want to inherit alignment from any previous data.
                    while (m.Position % 16 != 0)
                    {
                        binarywriter.Write((byte)0xCC);
                    }
                    if (writeremaining.Value.Target is Mesh)
                    {
                        // Record our location and write this object.
                        int offset_object = (int)m.Position;
                        writeremaining.Value.Offset = offset_object;
                        Serialize((Mesh)writeremaining.Value.Target);
                        // Record the EOF offset then go and patch the references.
                        int offset_eof = (int)m.Position;
                        foreach (var reference in writeremaining.Value.References)
                        {
                            binarywriter.Seek(reference, SeekOrigin.Begin);
                            binarywriter.Write((int)offset_object);
                        }
                        // Go back to the end of the file.
                        binarywriter.Seek(offset_eof, SeekOrigin.Begin);
                    }
                    if (writeremaining.Value.Target is MaterialCommon)
                    {
                        // Record our location and write this object.
                        int offset_object = (int)m.Position;
                        writeremaining.Value.Offset = offset_object;
                        Serialize((MaterialCommon)writeremaining.Value.Target);
                        // Record the EOF offset then go and patch the references.
                        int offset_eof = (int)m.Position;
                        foreach (var reference in writeremaining.Value.References)
                        {
                            binarywriter.Seek(reference, SeekOrigin.Begin);
                            binarywriter.Write((int)offset_object);
                        }
                        // Go back to the end of the file.
                        binarywriter.Seek(offset_eof, SeekOrigin.Begin);
                    }
                }
                // Go back and update the total file size.
                int length = (int)m.Position;
                binarywriter.Seek(0, SeekOrigin.Begin);
                binarywriter.Write((int)length);
            }
        }
        void Serialize(TransformedObject obj)
        {
            // Write the transform.
            Serialize(obj.Transform, Serialize);
            // Write the inverse transform.
            Serialize(MathHelp.Invert(obj.Transform), Serialize);
            // Write the object type.
            if (obj.Node.primitive is Plane)
            {
                binarywriter.Write((int)Geometry.GEOMETRY_PLANE);
                binarywriter.Write((int)0);
            }
            else if (obj.Node.primitive is Sphere)
            {
                binarywriter.Write((int)Geometry.GEOMETRY_SPHERE);
                binarywriter.Write((int)0);
            }
            else if (obj.Node.primitive is Cube)
            {
                binarywriter.Write((int)Geometry.GEOMETRY_CUBE);
                binarywriter.Write((int)0);
            }
            else if (obj.Node.primitive is Triangle)
            {
                binarywriter.Write((int)Geometry.GEOMETRY_TRIANGLE);
                binarywriter.Write((int)0);
            }
            else if (obj.Node.primitive is Mesh)
            {
                binarywriter.Write((int)Geometry.GEOMETRY_TRIANGLELIST);
                EmitAndQueue(obj.Node.primitive);
            }
            else
            {
                binarywriter.Write((int)Geometry.GEOMETRY_NONE);
                binarywriter.Write((int)0);
            }
            // Write the material type.
            if (obj.Node.material is MaterialCommon)
            {
                binarywriter.Write((int)Material.MATERIAL_COMMON);
                EmitAndQueue(obj.Node.material);
            }
            else if (obj.Node.material is CheckerboardMaterial)
            {
                binarywriter.Write((int)Material.MATERIAL_CHECKERBOARD_XZ);
                binarywriter.Write((int)0);
            }
            else
            {
                binarywriter.Write((int)Material.MATERIAL_NONE);
                binarywriter.Write((int)0);
            }
        }
        void Serialize(MaterialCommon obj)
        {
            Serialize(obj.Ambient, Serialize);
            Serialize(obj.Diffuse, Serialize);
            Serialize(obj.Specular, Serialize);
            Serialize(obj.Reflect, Serialize);
            Serialize(obj.Refract, Serialize);
            Serialize((double)obj.Ior);
        }
        void Serialize(Mesh obj)
        {
            // Write the header.
            binarywriter.Write((int)obj.Triangles.Length);
            binarywriter.Write((int)0);
            binarywriter.Write((int)0);
            binarywriter.Write((int)0);
            // Write the bounds.
            Point3D min = new Point3D(obj.Vertices.Min(x => x.X), obj.Vertices.Min(x => x.Y), obj.Vertices.Min(x => x.Z));
            Point3D max = new Point3D(obj.Vertices.Max(x => x.X), obj.Vertices.Max(x => x.Y), obj.Vertices.Max(x => x.Z));
            Serialize(min, Serialize);
            Serialize(max, Serialize);
            // Write all the triangles.
            foreach (var vtx in obj.Triangles
                .SelectMany( t => new[] { t.Index0, t.Index1, t.Index2 })
                .Select(v => obj.Vertices[v]))
            {
                Serialize(vtx, Serialize);
            }
        }
        void Serialize(double obj)
        {
            if (UseF64)
            {
                binarywriter.Write((double)obj);
            }
            else
            {
                binarywriter.Write((float)obj);
            }
        }
        private bool UseF64 = false;
        private MemoryStream m = new MemoryStream();
        private BinaryWriter binarywriter;
        #region - Section : Pointer Patchup -
        void EmitAndQueue(object obj)
        {
            // Try to find a record of this object.
            if (!WriteObjects.ContainsKey(obj))
            {
                WriteObjects[obj] = new PointerRecord { Target = obj };
            }
            // Flush the writer and make a note of this reference offset.
            WriteObjects[obj].References.Add((int)m.Position);
            // Write a placeholder pointer.
            binarywriter.Write((int)0);
        }
        class PointerRecord
        {
            public object Target = null;
            public int Offset = 0;
            public List<int> References = new List<int>();
        }
        Dictionary<object, PointerRecord> WriteObjects = new Dictionary<object, PointerRecord>();
        #endregion
        #region - Section : Serialization Primitives -
        enum Geometry
        {
            GEOMETRY_NONE = 0,
            GEOMETRY_PLANE = 0x6e616c50,        // FOURCC "Plan"
            GEOMETRY_SPHERE = 0x72687053,       // FOURCC "Sphr"
            GEOMETRY_CUBE = 0x65627543,         // FOURCC "Cube"
            GEOMETRY_TRIANGLE = 0x61697254,     // FOURCC "Tria"
            GEOMETRY_TRIANGLELIST = 0x4c697254, // FOURCC "TriL"
        }
        enum Material
        {
            MATERIAL_NONE = 0,
            MATERIAL_COMMON = 0x6c74614d,           // FOURCC "Matl"
            MATERIAL_CHECKERBOARD_XZ = 0x5a586843,  // FOURCC "ChXZ"
        }
        static void Serialize(Matrix3D obj, Action<double> write)
        {
            write(obj.M11); write(obj.M12); write(obj.M13); write(obj.M14);
            write(obj.M21); write(obj.M22); write(obj.M23); write(obj.M24);
            write(obj.M31); write(obj.M32); write(obj.M33); write(obj.M34);
            write(obj.M41); write(obj.M42); write(obj.M43); write(obj.M44);
        }
        static void Serialize(Point3D obj, Action<double> write)
        {
            write(obj.X); write(obj.Y); write(obj.Z); write(0);
        }
        static void Serialize(Point4D obj, Action<double> write)
        {
            write(obj.X); write(obj.Y); write(obj.Z); write(obj.W);
        }
        static void Serialize(Vector3D obj, Action<double> write)
        {
            write(obj.X); write(obj.Y); write(obj.Z); write(0);
        }
        #endregion
    }
}