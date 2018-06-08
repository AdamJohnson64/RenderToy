////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.PipelineModel;
using RenderToy.SceneGraph;
using RenderToy.Utility;
using System;

namespace RenderToy.RenderMode
{
    public static class RenderModeCS
    {
        public static void PointCPUF64(IScene scene, Matrix3D mvp, IntPtr bitmap_ptr, int render_width, int render_height, int bitmap_stride)
        {
            var pixels = Rasterization.RasterizePoint(scene, mvp, render_width, render_height);
            Rasterization.FillBitmap(pixels, bitmap_ptr, render_width, render_height, bitmap_stride);
        }
        public static void WireframeCPUF64(IScene scene, Matrix3D mvp, IntPtr bitmap_ptr, int render_width, int render_height, int bitmap_stride)
        {
            var pixels = Rasterization.RasterizeLine(scene, mvp, render_width, render_height);
            Rasterization.FillBitmap(pixels, bitmap_ptr, render_width, render_height, bitmap_stride);
        }
        public static void RasterCPUF64(IScene scene, Matrix3D mvp, IntPtr bitmap_ptr, int render_width, int render_height, int bitmap_stride)
        {
            var pixels = Rasterization.RasterizeTriangle(scene, mvp, render_width, render_height);
            Rasterization.FillBitmap(pixels, bitmap_ptr, render_width, render_height, bitmap_stride);
        }
        public static void RasterHomogeneousCPUF64(IScene scene, Matrix3D mvp, IntPtr bitmap_ptr, int render_width, int render_height, int bitmap_stride)
        {
            var pixels = Rasterization.RasterizeHomogeneous(scene, mvp, (ushort)render_width, (ushort)render_height);
            Rasterization.FillBitmap(pixels, bitmap_ptr, (ushort)render_width, (ushort)render_height, bitmap_stride);
        }
    }
}