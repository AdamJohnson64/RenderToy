////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System;
using System.Windows.Media.Media3D;

namespace RenderToy
{
    public static partial class Render
    {
        #region - Section : Phase 4 - Raytrace Rendering (CUDA) -
        public static void RaytraceCUDA(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            mvp.Invert();
            double[] MVP = new double[16] {
                mvp.M11, mvp.M12, mvp.M13, mvp.M14,
                mvp.M21, mvp.M22, mvp.M23, mvp.M24,
                mvp.M31, mvp.M32, mvp.M33, mvp.M34,
                mvp.OffsetX, mvp.OffsetY, mvp.OffsetZ, mvp.M44,
            };
            RenderToy.RaytraceCUDA.Fill(MVP, bitmap_ptr, bitmap_width, bitmap_height, bitmap_stride);
        }
        #endregion
    }
}