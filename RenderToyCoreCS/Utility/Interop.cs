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
                // Write all the objects.
                foreach (var obj in transformedobjects)
                {
                    Serialize(obj);
                }
                // Write all the outstanding queued object references and patch their reference sources.
                while (WriteNext.Count > 0)
                {
                    var writeremaining = WriteObjects[WriteNext.Dequeue()];
                    // Always pad additional objects to 8 bytes in double mode.
                    // We don't want to inherit alignment from any previous data.
                    while (UseF64 && m.Position % 8 != 0)
                    {
                        binarywriter.Write((byte)0xCC);
                    }
                    // Record our location and write this object.
                    writeremaining.Offset = (int)m.Position;
                    if (writeremaining.Target is Mesh)
                    {
                        Serialize((Mesh)writeremaining.Target);
                    }
                    else if (writeremaining.Target is MeshBVH)
                    {
                        Serialize((MeshBVH)writeremaining.Target);
                    }
                    else if (writeremaining.Target is MeshBVH.Node)
                    {
                        Serialize((MeshBVH.Node)writeremaining.Target);
                    }
                    else if (writeremaining.Target is IReadOnlyList<MeshBVH.Node>)
                    {
                        Serialize((IReadOnlyList<MeshBVH.Node>)writeremaining.Target);
                    }
                    else if (writeremaining.Target is IReadOnlyList<TriIndex>)
                    {
                        Serialize((IReadOnlyList<TriIndex>)writeremaining.Target);
                    }
                    else if (writeremaining.Target is MaterialCommon)
                    {
                        Serialize((MaterialCommon)writeremaining.Target);
                    }
                    else if (writeremaining.Target is IReadOnlyList<Point3D>)
                    {
                        Serialize((IReadOnlyList<Point3D>)writeremaining.Target);
                    }
                    else
                    {
                        throw new InvalidCastException("Cannot serialize '" + writeremaining.Target.GetType().Name + "'.");
                    }
                        
                }
                // Go through all appended objects and patch up their references.
                foreach (var writeremaining in WriteObjects)
                {
                    // Record the EOF offset then go and patch the references.
                    foreach (var reference in writeremaining.Value.References)
                    {
                        binarywriter.Seek(reference, SeekOrigin.Begin);
                        binarywriter.Write((int)writeremaining.Value.Offset);
                    }
                    // Go back to the end of the file.
                    binarywriter.Seek(0, SeekOrigin.End);
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
            else if (obj.Node.primitive is MeshBVH)
            {
                binarywriter.Write((int)Geometry.GEOMETRY_MESHBVH);
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
        void Serialize(MeshBVH obj)
        {
            EmitAndQueue(obj.Vertices);
            EmitAndQueue(obj.Root);
        }
        void Serialize(MeshBVH.Node obj)
        {
            Serialize(obj.Bound.Min, Serialize);
            Serialize(obj.Bound.Max, Serialize);
            EmitAndQueue(obj.Children);
            EmitAndQueue(obj.Triangles);
        }
        void Serialize(IReadOnlyList<MeshBVH.Node> obj)
        {
            binarywriter.Write((int)obj.Count);
            binarywriter.Write((int)0);
            foreach (var item in obj)
            {
                Serialize(item);
            }
        }
        void Serialize(IReadOnlyList<Point3D> obj)
        {
            binarywriter.Write((int)obj.Count);
            binarywriter.Write((int)0);
            foreach (var item in obj)
            {
                Serialize(item, Serialize);
            }
        }
        void Serialize(IReadOnlyList<TriIndex> obj)
        {
            binarywriter.Write((int)obj.Count);
            foreach (var item in obj)
            {
                binarywriter.Write(item.Index0);
                binarywriter.Write(item.Index1);
                binarywriter.Write(item.Index2);
            }
        }
        void Serialize(double obj)
        {
            if (UseF64)
            {
                if (binarywriter.BaseStream.Position % 8 != 0)
                {
                    throw new DataMisalignedException("Misaligned double (suboptimal on CPU and illegal on GPU).");
                }
                binarywriter.Write((double)obj);
            }
            else
            {
                if (binarywriter.BaseStream.Position % 4 != 0)
                {
                    throw new DataMisalignedException("Misaligned float (suboptimal on CPU and illegal on GPU).");
                }
                binarywriter.Write((float)obj);
            }
        }
        private bool UseF64 = false;
        private MemoryStream m = new MemoryStream();
        private BinaryWriter binarywriter;
        #region - Section : Pointer Patchup -
        void EmitAndQueue(object obj)
        {
            int offset = (int)m.Position;
            // Write a placeholder pointer.
            binarywriter.Write((int)0);
            // If this pointer is null then it can't be written; leave it as zero.
            if (obj == null) return;
            // Try to find a record of this object.
            if (!WriteObjects.ContainsKey(obj))
            {
                WriteObjects[obj] = new PointerRecord { Target = obj };
                WriteNext.Enqueue(obj);
            }
            // Flush the writer and make a note of this reference offset.
            WriteObjects[obj].References.Add((int)offset);
        }
        class PointerRecord
        {
            public object Target = null;
            public int Offset = 0;
            public List<int> References = new List<int>();
        }
        Dictionary<object, PointerRecord> WriteObjects = new Dictionary<object, PointerRecord>();
        Queue<object> WriteNext = new Queue<object>();
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
            GEOMETRY_MESHBVH = 0x4268734d,      // FOURCC "MshB"
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
            write(obj.X); write(obj.Y); write(obj.Z);
        }
        static void Serialize(Point4D obj, Action<double> write)
        {
            write(obj.X); write(obj.Y); write(obj.Z); write(obj.W);
        }
        static void Serialize(Vector3D obj, Action<double> write)
        {
            write(obj.X); write(obj.Y); write(obj.Z);
        }
        #endregion
    }
}