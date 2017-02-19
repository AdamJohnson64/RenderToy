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
        static void Serialize(Matrix3D obj, Action<double> write)
        {
            write(obj.M11); write(obj.M12); write(obj.M13); write(obj.M14);
            write(obj.M21); write(obj.M22); write(obj.M23); write(obj.M24);
            write(obj.M31); write(obj.M32); write(obj.M33); write(obj.M34);
            write(obj.M41); write(obj.M42); write(obj.M43); write(obj.M44);
        }
        static void Serialize(Vector3D obj, Action<double> write)
        {
            write(obj.X);
            write(obj.Y);
            write(obj.Z);
            write(0);
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
                    if (writeremaining.Value.Target is MaterialCommon)
                    {
                        // Pad to 16 bytes for safety (there may be double4s in here).
                        while (m.Position % 16 != 0)
                        {
                            binarywriter.Write((byte)0xCC);
                        }
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
            Serialize(obj.Transform);
            // Write the inverse transform.
            Serialize(MathHelp.Invert(obj.Transform));
            // Write the object type.
            if (obj.Node.primitive is Plane) { binarywriter.Write((int)Geometry.GEOMETRY_PLANE); }
            else if (obj.Node.primitive is Sphere) { binarywriter.Write((int)Geometry.GEOMETRY_SPHERE); }
            else if (obj.Node.primitive is Cube) { binarywriter.Write((int)Geometry.GEOMETRY_CUBE); }
            else if (obj.Node.primitive is Triangle) { binarywriter.Write((int)Geometry.GEOMETRY_TRIANGLE); }
            else { binarywriter.Write((int)Geometry.GEOMETRY_NONE); }
            // Write the offset to the object (or zero).
            binarywriter.Write((int)0);
            // Write the material type.
            if (obj.Node.material is MaterialCommon) { binarywriter.Write((int)1); }
            else if (obj.Node.material is CheckerboardMaterial) { binarywriter.Write((int)2); }
            else { binarywriter.Write((int)0); }
            // Write the offset to the material (or zero).
            if (obj.Node.material is MaterialCommon) { EmitAndQueue(obj.Node.material); }
            else { binarywriter.Write((int)0); }
        }
        void Serialize(Point4D obj)
        {
            if (m.Position % 16 != 0)
            {
                throw new Exception("Data will not be 16-byte aligned.");
            }
            Serialize((double)obj.X);
            Serialize((double)obj.Y);
            Serialize((double)obj.Z);
            Serialize((double)obj.W);
        }
        void Serialize(MaterialCommon obj)
        {
            Serialize(obj.Ambient);
            Serialize(obj.Diffuse);
            Serialize(obj.Specular);
            Serialize(obj.Reflect);
            Serialize(obj.Refract);
            Serialize((double)obj.Ior);
        }
        void Serialize(Matrix3D obj)
        {
            Serialize((double)obj.M11); Serialize((double)obj.M12); Serialize((double)obj.M13); Serialize((double)obj.M14);
            Serialize((double)obj.M21); Serialize((double)obj.M22); Serialize((double)obj.M23); Serialize((double)obj.M24);
            Serialize((double)obj.M31); Serialize((double)obj.M32); Serialize((double)obj.M33); Serialize((double)obj.M34);
            Serialize((double)obj.M41); Serialize((double)obj.M42); Serialize((double)obj.M43); Serialize((double)obj.M44);
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
        private bool UseF64 = false;
        private MemoryStream m = new MemoryStream();
        private BinaryWriter binarywriter;
        class PointerRecord
        {
            public object Target = null;
            public int Offset = 0;
            public List<int> References = new List<int>();
        }
        Dictionary<object, PointerRecord> WriteObjects = new Dictionary<object, PointerRecord>();
        enum Geometry
        {
            GEOMETRY_NONE = 0,
            GEOMETRY_PLANE = 1,
            GEOMETRY_SPHERE = 2,
            GEOMETRY_CUBE = 3,
            GEOMETRY_TRIANGLE = 4,
        }
    }
}