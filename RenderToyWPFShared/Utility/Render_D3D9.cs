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
        /// <summary>
        /// Rasterize the scene using XYZ 3D triangles and transforming via the hardware pipeline.
        /// </summary>
        /// <param name="scene">The scene to render.</param>
        /// <param name="mvp">The model-view-projection matrix.</param>
        /// <param name="bitmap_ptr">A pointer to an output bitmap.</param>
        /// <param name="render_width">The width of the output bitmap in pixels.</param>
        /// <param name="render_height">The height of the output bitmap in pixels.</param>
        /// <param name="bitmap_stride">The byte count between rasters of the output bitmap.</param>
        public static void RasterXYZD3D9(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int render_width, int render_height, int bitmap_stride)
        {
            var device = direct3d.CreateDevice();
            var rendertarget = device.CreateRenderTarget((uint)render_width, (uint)render_height, D3DFormat.A8R8G8B8, D3DMultisample.None, 0, 1);
            var depthstencil = device.CreateDepthStencilSurface((uint)render_width, (uint)render_height, D3DFormat.D24X8, D3DMultisample.None, 0, 1);
            device.SetRenderTarget(0, rendertarget);
            device.SetDepthStencilSurface(depthstencil);
            device.BeginScene();
            device.Clear(D3DClear.Target | D3DClear.ZBuffer, 0x00000000, 1.0f, 0);
            device.SetFVF(D3DFvf.XYZ | D3DFvf.Diffuse);
            device.SetRenderState(D3DRenderState.ZEnable, 1U);
            device.SetRenderState(D3DRenderState.CullMode, (uint)D3DCullMode.None);
            device.SetRenderState(D3DRenderState.Lighting, 0);
            foreach (var transformedobject in TransformedObject.Enumerate(scene))
            {
                var A = PrimitiveAssembly.CreateTriangles(transformedobject.Node.GetPrimitive());
                var B = A.Select(i => new XYZDiffuse { X = (float)i.X, Y = (float)i.Y, Z = (float)i.Z, Color = Rasterization.ColorToUInt32(transformedobject.Node.GetWireColor()) });
                var vertexbuffer = B.ToArray();
                device.SetTransform(D3DTransformState.Projection, Marshal.UnsafeAddrOfPinnedArrayElement(D3DMatrix.Convert(transformedobject.Transform * mvp), 0));
                device.DrawPrimitiveUP(RenderToy.D3DPrimitiveType.TriangleList, (uint)(vertexbuffer.Length / 3), Marshal.UnsafeAddrOfPinnedArrayElement(vertexbuffer, 0), (uint)Marshal.SizeOf(typeof(XYZDiffuse)));
            }
            device.EndScene();
            rendertarget.CopyTo(bitmap_ptr, render_width, render_height, bitmap_stride);
        }
        struct XYZWDiffuse
        {
            public float X, Y, Z, W;
            public uint Color;
        };
        /// <summary>
        /// Rasterize the scene using XYZW homogeneous triangles and transforming via the our software pipeline.
        /// </summary>
        /// <param name="scene">The scene to render.</param>
        /// <param name="mvp">The model-view-projection matrix.</param>
        /// <param name="bitmap_ptr">A pointer to an output bitmap.</param>
        /// <param name="render_width">The width of the output bitmap in pixels.</param>
        /// <param name="render_height">The height of the output bitmap in pixels.</param>
        /// <param name="bitmap_stride">The byte count between rasters of the output bitmap.</param>
        public static void RasterXYZWD3D9(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int render_width, int render_height, int bitmap_stride)
        {
            var device = direct3d.CreateDevice();
            var rendertarget = device.CreateRenderTarget((uint)render_width, (uint)render_height, D3DFormat.A8R8G8B8, D3DMultisample.None, 0, 1);
            var depthstencil = device.CreateDepthStencilSurface((uint)render_width, (uint)render_height, D3DFormat.D24X8, D3DMultisample.None, 0, 1);
            device.SetRenderTarget(0, rendertarget);
            device.SetDepthStencilSurface(depthstencil);
            device.BeginScene();
            device.Clear(D3DClear.Target | D3DClear.ZBuffer, 0x00000000, 1.0f, 0);
            device.SetFVF(D3DFvf.XYZW | D3DFvf.Diffuse);
            device.SetRenderState(D3DRenderState.ZEnable, 1U);
            device.SetRenderState(D3DRenderState.CullMode, (uint)D3DCullMode.None);
            device.SetRenderState(D3DRenderState.Lighting, 0);
            foreach (var transformedobject in TransformedObject.Enumerate(scene))
            {
                var A = PrimitiveAssembly.CreateTriangles(transformedobject.Node.GetPrimitive());
                var B = Transformation.Vector3ToVector4(A);
                var C = Transformation.Transform(B, transformedobject.Transform * mvp);
                var D = Clipping.ClipTriangle(C);
                var E = Transformation.HomogeneousDivide(D);
                var F = E.Select(i => new XYZWDiffuse { X = (float)i.X, Y = (float)i.Y, Z = (float)i.Z, W = (float)i.W, Color = Rasterization.ColorToUInt32(transformedobject.Node.GetWireColor()) });
                var vertexbuffer = F.ToArray();
                device.DrawPrimitiveUP(RenderToy.D3DPrimitiveType.TriangleList, (uint)(vertexbuffer.Length / 3), Marshal.UnsafeAddrOfPinnedArrayElement(vertexbuffer, 0), (uint)Marshal.SizeOf(typeof(XYZWDiffuse)));
            }
            device.EndScene();
            rendertarget.CopyTo(bitmap_ptr, render_width, render_height, bitmap_stride);
        }
        static Direct3D9 direct3d = new Direct3D9();
        #endregion
    }
}
