////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using RenderToy.PipelineModel;
using RenderToy.SceneGraph;
using RenderToy.Utility;
using System;
using System.Linq;
using System.Runtime.InteropServices;

namespace RenderToy.WPF
{
    public struct D3DMatrix
    {
        public static float[] Convert(Matrix3D matrix)
        {
            return new float[16]
            {
                (float)matrix.M11, (float)matrix.M12, (float)matrix.M13, (float)matrix.M14,
                (float)matrix.M21, (float)matrix.M22, (float)matrix.M23, (float)matrix.M24,
                (float)matrix.M31, (float)matrix.M32, (float)matrix.M33, (float)matrix.M34,
                (float)matrix.M41, (float)matrix.M42, (float)matrix.M43, (float)matrix.M44,
            };
        }
    }
    public static class RenderD3D
    {
        #region - Section : Phase 3 - Rasterized Rendering (Direct3D 9) -
        struct XYZDiffuse
        {
            public float X, Y, Z;
            public uint Color;
        };
        public static void RasterXYZD3D9(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int render_width, int render_height, int bitmap_stride)
        {
            D3D9Surface d3dsurface = new D3D9Surface(render_width, render_height);
            d3dsurface.BeginScene();
            d3dsurface.SetFVF(D3DFvf.XYZ | D3DFvf.Diffuse);
            d3dsurface.SetRenderState(D3DRenderState.CullMode, (uint)D3DCullMode.None);
            d3dsurface.SetRenderState(D3DRenderState.Lighting, 0);
            foreach (var transformedobject in TransformedObject.Enumerate(scene))
            {
                var A = PrimitiveAssembly.CreateTriangles(transformedobject.Node.Primitive);
                var B = A.Select(i => new XYZDiffuse { X = (float)i.X, Y = (float)i.Y, Z = (float)i.Z, Color = Rasterization.ColorToUInt32(transformedobject.Node.WireColor) });
                var vertexbuffer = B.ToArray();
                d3dsurface.SetTransform(D3DTransformState.Projection, Marshal.UnsafeAddrOfPinnedArrayElement(D3DMatrix.Convert(transformedobject.Transform * mvp), 0));
                d3dsurface.DrawPrimitiveUP(RenderToy.D3DPrimitiveType.TriangleList, (uint)(vertexbuffer.Length / 3), Marshal.UnsafeAddrOfPinnedArrayElement(vertexbuffer, 0), (uint)Marshal.SizeOf(typeof(XYZDiffuse)));
            }
            d3dsurface.EndScene();
            d3dsurface.CopyTo(bitmap_ptr, render_width, render_height, bitmap_stride);
        }
        struct XYZWDiffuse
        {
            public float X, Y, Z, W;
            public uint Color;
        };
        public static void RasterXYZWD3D9(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int render_width, int render_height, int bitmap_stride)
        {
            D3D9Surface d3dsurface = new D3D9Surface(render_width, render_height);
            d3dsurface.BeginScene();
            d3dsurface.SetFVF(D3DFvf.XYZW | D3DFvf.Diffuse);
            d3dsurface.SetRenderState(D3DRenderState.CullMode, (uint)D3DCullMode.None);
            d3dsurface.SetRenderState(D3DRenderState.Lighting, 0);
            foreach (var transformedobject in TransformedObject.Enumerate(scene))
            {
                var A = PrimitiveAssembly.CreateTriangles(transformedobject.Node.Primitive);
                var B = Transformation.Vector3ToVector4(A);
                var C = Transformation.Transform(B, transformedobject.Transform * mvp);
                var D = Clipping.ClipTriangle(C);
                var E = Transformation.HomogeneousDivide(D);
                var F = E.Select(i => new XYZWDiffuse { X = (float)i.X, Y = (float)i.Y, Z = (float)i.Z, W = (float)i.W, Color = Rasterization.ColorToUInt32(transformedobject.Node.WireColor) });
                var vertexbuffer = F.ToArray();
                d3dsurface.DrawPrimitiveUP(RenderToy.D3DPrimitiveType.TriangleList, (uint)(vertexbuffer.Length / 3), Marshal.UnsafeAddrOfPinnedArrayElement(vertexbuffer, 0), (uint)Marshal.SizeOf(typeof(XYZWDiffuse)));
            }
            d3dsurface.EndScene();
            d3dsurface.CopyTo(bitmap_ptr, render_width, render_height, bitmap_stride);
        }
        #endregion
    }
}
