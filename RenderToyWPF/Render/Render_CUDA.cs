////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System;
using System.IO;
using System.Linq;
using System.Windows.Media.Media3D;

namespace RenderToy
{
    public static partial class Render
    {
        #region - Section : Phase 4 - Raytrace Rendering (CUDA) -
        public static bool CUDAAvailable()
        {
            return RenderToy.RaytraceCUDA.Available();
        }
        public static void RaytraceCUDA(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            // Prepare the scene definition for the renderer.
            MemoryStream mem = new MemoryStream();
            using (BinaryWriter wr = new BinaryWriter(mem))
            {
                var writeobjects = TransformedObject.Enumerate(scene).ToList();
                // Write the object count.
                wr.Write((int)writeobjects.Count);
                wr.Write((int)0);
                foreach (var o in writeobjects)
                {
                    // Write the transform.
                    Serialize(wr, o.Transform);
                    // Write the inverse transform.
                    Serialize(wr, MathHelp.Invert(o.Transform));
                    // Write the object type.
                    if (o.Node.primitive is Plane) { wr.Write((int)1); }
                    else if (o.Node.primitive is Sphere) { wr.Write((int)2); }
                    else { wr.Write((int)0); }
                    // Write the material type.
                    if (o.Node.material is CheckerboardMaterial) { wr.Write((int)1); }
                    else if (o.Node.material is ConstantColorMaterial) { wr.Write((int)2); }
                    else if (o.Node.material is GlassMaterial) { wr.Write((int)5); }
                    else { wr.Write((int)0); }
                }
            }
            // Prepare the camera inverse MVP for the renderer.
            mvp.Invert();
            double[] inverse_mvp = new double[16] {
                mvp.M11, mvp.M12, mvp.M13, mvp.M14,
                mvp.M21, mvp.M22, mvp.M23, mvp.M24,
                mvp.M31, mvp.M32, mvp.M33, mvp.M34,
                mvp.OffsetX, mvp.OffsetY, mvp.OffsetZ, mvp.M44,
            };
            // Render the scene using the CUDA raytracer.
            RenderToy.RaytraceCUDA.Fill(mem.ToArray(), inverse_mvp, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
        }
        static void Serialize(BinaryWriter w, Matrix3D m)
        {
            w.Write((double)m.M11); w.Write((double)m.M12); w.Write((double)m.M13); w.Write((double)m.M14);
            w.Write((double)m.M21); w.Write((double)m.M22); w.Write((double)m.M23); w.Write((double)m.M24);
            w.Write((double)m.M31); w.Write((double)m.M32); w.Write((double)m.M33); w.Write((double)m.M34);
            w.Write((double)m.OffsetX); w.Write((double)m.OffsetY); w.Write((double)m.OffsetZ); w.Write((double)m.M44);
        }
        #endregion
    }
}