////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Math;
using RenderToy.PipelineModel;
using RenderToy.SceneGraph;
using System;
using System.Linq;

namespace RenderToy.RenderMode
{
    public static class RenderModeCS
    {
        public static void PointCPUF64(IScene scene, Matrix3D mvp, IntPtr bitmap_ptr, int render_width, int render_height, int bitmap_stride)
        {
            var pixels = TransformedObject.Enumerate(scene).SelectMany(transformednode => {
                var triangles = PrimitiveAssembly.CreatePoints(transformednode.Node.Primitive);
                var v3tov4 = Transformation.Vector3ToVector4(triangles);
                var clipspace = Transformation.Transform(v3tov4, transformednode.Transform * mvp);
                var clipped = Clipping.ClipPoint(clipspace);
                var hdiv = Transformation.HomogeneousDivide(clipped);
                var screenspace = Transformation.TransformToScreen(hdiv, render_width, render_height);
                return Rasterization.RasterizePoint(screenspace, Rasterization.ColorToUInt32(transformednode.Node.WireColor));
            });
            Rasterization.FillBitmap(pixels, bitmap_ptr, render_width, render_height, bitmap_stride);
        }
        public static void WireframeCPUF64(IScene scene, Matrix3D mvp, IntPtr bitmap_ptr, int render_width, int render_height, int bitmap_stride)
        {
            var pixels = TransformedObject.Enumerate(scene).SelectMany(transformednode => {
                var triangles = PrimitiveAssembly.CreateLines(transformednode.Node.Primitive);
                var v3tov4 = Transformation.Vector3ToVector4(triangles);
                var clipspace = Transformation.Transform(v3tov4, transformednode.Transform * mvp);
                var clipped = Clipping.ClipLine(clipspace);
                var hdiv = Transformation.HomogeneousDivide(clipped);
                var screenspace = Transformation.TransformToScreen(hdiv, render_width, render_height);
                return Rasterization.RasterizeLine(screenspace, Rasterization.ColorToUInt32(transformednode.Node.WireColor));
            });
            Rasterization.FillBitmap(pixels, bitmap_ptr, render_width, render_height, bitmap_stride);
        }
        public static void RasterCPUF64(IScene scene, Matrix3D mvp, IntPtr bitmap_ptr, int render_width, int render_height, int bitmap_stride)
        {
            var pixels = TransformedObject.Enumerate(scene).SelectMany(transformednode => {
                var triangles = PrimitiveAssembly.CreateTriangles(transformednode.Node.Primitive);
                var v3tov4 = Transformation.Vector3ToVector4(triangles);
                var clipspace = Transformation.Transform(v3tov4, transformednode.Transform * mvp);
                var clipped = Clipping.ClipTriangle(clipspace);
                var hdiv = Transformation.HomogeneousDivide(clipped);
                var screenspace = Transformation.TransformToScreen(hdiv, render_width, render_height);
                return Rasterization.RasterizeTriangle(screenspace, Rasterization.ColorToUInt32(transformednode.Node.WireColor));
            });
            Rasterization.FillBitmap(pixels, bitmap_ptr, render_width, render_height, bitmap_stride);
        }
        public static void RasterHomogeneousCPUF64(IScene scene, Matrix3D mvp, IntPtr bitmap_ptr, int render_width, int render_height, int bitmap_stride)
        {
            var pixels = TransformedObject.Enumerate(scene).SelectMany(transformednode => {
                var triangles = PrimitiveAssembly.CreateTriangles(transformednode.Node.Primitive);
                var v3tov4 = Transformation.Vector3ToVector4(triangles);
                var clipspace = Transformation.Transform(v3tov4, transformednode.Transform * mvp);
                return Rasterization.RasterizeHomogeneous(clipspace, (ushort)render_width, (ushort)render_height);
            });
            Rasterization.FillBitmap(pixels, bitmap_ptr, (ushort)render_width, (ushort)render_height, bitmap_stride);
        }
    }
}