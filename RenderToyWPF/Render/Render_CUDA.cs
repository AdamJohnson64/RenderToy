////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System;
using System.Collections.Generic;
using System.IO;
using System.Linq;
using System.Windows.Media;
using System.Windows.Media.Media3D;

namespace RenderToy
{
    public static partial class Render
    {
        #region - Section : Phase 4 - Raytrace Rendering (CUDA) -
        public static bool CUDAAvailable()
        {
            return RenderToy.CUDASupport.Available();
        }
        public static void RaycastCUDA(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            RenderToy.RaycastCUDA.Fill(SceneFormatter.CreateFlatMemory(scene), MatrixToDoubles(MathHelp.Invert(mvp)), bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
        }
        public static void RaytraceCUDA(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            RenderToy.RaytraceCUDA.Fill(SceneFormatter.CreateFlatMemory(scene), MatrixToDoubles(MathHelp.Invert(mvp)), bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
        }
        static double[] MatrixToDoubles(Matrix3D mvp)
        {
            return new double[16] {
                mvp.M11, mvp.M12, mvp.M13, mvp.M14,
                mvp.M21, mvp.M22, mvp.M23, mvp.M24,
                mvp.M31, mvp.M32, mvp.M33, mvp.M34,
                mvp.OffsetX, mvp.OffsetY, mvp.OffsetZ, mvp.M44,
            };
        }
        class SceneFormatter
        {
            public static byte[] CreateFlatMemory(Scene scene)
            {
                return new SceneFormatter(scene).m.ToArray();
            }
            SceneFormatter(Scene scene)
            {
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
                if (obj.Node.primitive is Plane) { binarywriter.Write((int)1); }
                else if (obj.Node.primitive is Sphere) { binarywriter.Write((int)2); }
                else if (obj.Node.primitive is Cube) { binarywriter.Write((int)3); }
                else { binarywriter.Write((int)0); }
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
            void Serialize(Color obj)
            {
                binarywriter.Write((double)obj.R / 255.0);
                binarywriter.Write((double)obj.G / 255.0);
                binarywriter.Write((double)obj.B / 255.0);
                binarywriter.Write((double)obj.A / 255.0);
            }
            void Serialize(MaterialCommon obj)
            {
                Serialize(obj.Ambient);
                Serialize(obj.Diffuse);
                Serialize(obj.Specular);
                Serialize(obj.Reflect);
                Serialize(obj.Refract);
                binarywriter.Write((double)obj.Ior);
            }
            void Serialize(Matrix3D obj)
            {
                binarywriter.Write((double)obj.M11); binarywriter.Write((double)obj.M12); binarywriter.Write((double)obj.M13); binarywriter.Write((double)obj.M14);
                binarywriter.Write((double)obj.M21); binarywriter.Write((double)obj.M22); binarywriter.Write((double)obj.M23); binarywriter.Write((double)obj.M24);
                binarywriter.Write((double)obj.M31); binarywriter.Write((double)obj.M32); binarywriter.Write((double)obj.M33); binarywriter.Write((double)obj.M34);
                binarywriter.Write((double)obj.OffsetX); binarywriter.Write((double)obj.OffsetY); binarywriter.Write((double)obj.OffsetZ); binarywriter.Write((double)obj.M44);
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
            private MemoryStream m = new MemoryStream();
            private BinaryWriter binarywriter;
            class PointerRecord
            {
                public object Target = null;
                public int Offset = 0;
                public List<int> References = new List<int>();
            }
            Dictionary<object, PointerRecord> WriteObjects = new Dictionary<object, PointerRecord>();
        }
        #endregion
    }
}