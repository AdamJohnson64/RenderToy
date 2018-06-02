﻿////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Materials;
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
    public static class D3DHelper
    {
        public static void CopySurface(Direct3DSurface9 surface, IntPtr bitmap_ptr, int render_width, int render_height, int bitmap_stride)
        {
            var locked = surface.LockRect();
            unsafe
            {
                for (int y = 0; y < render_height; ++y)
                {
                    void* rasterin = ((byte*)locked.Bits + locked.Pitch * y);
                    void* rasterout = ((byte*)bitmap_ptr + bitmap_stride * y);
                    Buffer.MemoryCopy(rasterin, rasterout, sizeof(uint) * render_width, sizeof(uint) * render_width);
                }
            }
            surface.UnlockRect();
        }
    }
    public static class RenderD3D
    {
        #region - Section : Phase 3 - Rasterized Rendering (Direct3D 9) -
        public struct XYZWDiffuse
        {
            public float X, Y, Z, W;
            public uint Diffuse;
        };
        /// <summary>
        /// Rasterize the scene using XYZW homogeneous triangles and transforming via our software pipeline.
        /// </summary>
        /// <param name="scene">The scene to render.</param>
        /// <param name="mvp">The model-view-projection matrix.</param>
        /// <param name="bitmap_ptr">A pointer to an output bitmap.</param>
        /// <param name="render_width">The width of the output bitmap in pixels.</param>
        /// <param name="render_height">The height of the output bitmap in pixels.</param>
        /// <param name="bitmap_stride">The byte count between rasters of the output bitmap.</param>
        public static void RasterD3D9(IScene scene, Matrix3D mvp, IntPtr bitmap_ptr, int render_width, int render_height, int bitmap_stride)
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
                var A = PrimitiveAssembly.CreateTriangles(transformedobject.Node.Primitive);
                var B = Transformation.Vector3ToVector4(A);
                var C = Transformation.Transform(B, transformedobject.Transform * mvp);
                var D = Clipping.ClipTriangle(C);
                var E = Transformation.HomogeneousDivide(D);
                var F = E.Select(i => new XYZWDiffuse { X = (float)i.X, Y = (float)i.Y, Z = (float)i.Z, W = (float)i.W, Diffuse = Rasterization.ColorToUInt32(transformedobject.Node.WireColor) });
                var vertexbuffer = F.ToArray();
                device.DrawPrimitiveUP(RenderToy.D3DPrimitiveType.TriangleList, (uint)(vertexbuffer.Length / 3), Marshal.UnsafeAddrOfPinnedArrayElement(vertexbuffer, 0), (uint)Marshal.SizeOf(typeof(XYZWDiffuse)));
            }
            device.EndScene();
            D3DHelper.CopySurface(rendertarget, bitmap_ptr, render_width, render_height, bitmap_stride);
        }
        public struct XYZDiffuse
        {
            public float X, Y, Z;
            public uint Diffuse;
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
        public static void RasterSolidD3D9(IScene scene, Matrix3D mvp, IntPtr bitmap_ptr, int render_width, int render_height, int bitmap_stride)
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
                var A = PrimitiveAssembly.CreateTriangles(transformedobject.Node.Primitive);
                var B = A.Select(i => new XYZDiffuse { X = (float)i.X, Y = (float)i.Y, Z = (float)i.Z, Diffuse = Rasterization.ColorToUInt32(transformedobject.Node.WireColor) });
                var vertexbuffer = B.ToArray();
                device.SetTransform(D3DTransformState.Projection, Marshal.UnsafeAddrOfPinnedArrayElement(D3DMatrix.Convert(transformedobject.Transform * mvp), 0));
                device.DrawPrimitiveUP(RenderToy.D3DPrimitiveType.TriangleList, (uint)(vertexbuffer.Length / 3), Marshal.UnsafeAddrOfPinnedArrayElement(vertexbuffer, 0), (uint)Marshal.SizeOf(typeof(XYZDiffuse)));
            }
            device.EndScene();
            D3DHelper.CopySurface(rendertarget, bitmap_ptr, render_width, render_height, bitmap_stride);
        }
        public struct XYZNorDiffuseTex1
        {
            public float Xp, Yp, Zp;
            public float Xn, Yn, Zn;
            public uint Diffuse;
            public float U, V;
            public float Tx, Ty, Tz;
            public float Bx, By, Bz;
        };
        /// <summary>
        /// Rasterize the scene using textured XYZ+UV 3D triangles and transforming via the hardware pipeline.
        /// </summary>
        /// <param name="scene">The scene to render.</param>
        /// <param name="mvp">The model-view-projection matrix.</param>
        /// <param name="bitmap_ptr">A pointer to an output bitmap.</param>
        /// <param name="render_width">The width of the output bitmap in pixels.</param>
        /// <param name="render_height">The height of the output bitmap in pixels.</param>
        /// <param name="bitmap_stride">The byte count between rasters of the output bitmap.</param>
        public static void RasterTexturedD3D9(IScene scene, Matrix3D mvp, IntPtr bitmap_ptr, int render_width, int render_height, int bitmap_stride)
        {
            const int TextureSize = 128;
            var device = direct3d.CreateDevice();
            var rendertarget = device.CreateRenderTarget((uint)render_width, (uint)render_height, D3DFormat.A8R8G8B8, D3DMultisample.None, 0, 1);
            var depthstencil = device.CreateDepthStencilSurface((uint)render_width, (uint)render_height, D3DFormat.D24X8, D3DMultisample.None, 0, 1);
            var texture = device.CreateTexture((uint)TextureSize, (uint)TextureSize, 1, 0U, D3DFormat.A8R8G8B8, D3DPool.Managed);
            D3DLockedRect lockit = texture.LockRect(0);
            var material = StockMaterials.MarbleTile;
            EvalContext context = new EvalContext();
            var param = System.Linq.Expressions.Expression.Parameter(typeof(EvalContext));
            var body = material.CreateExpression(param);
            var lambda = System.Linq.Expressions.Expression.Lambda<Func<EvalContext, Vector4D>>(body, param).Compile();
            unsafe
            {
                for (int y = 0; y < TextureSize; ++y)
                {
                    uint* raster = (uint*)((byte*)lockit.Bits + lockit.Pitch * y);
                    for (int x = 0; x < TextureSize; ++x)
                    {
                        context.U = x / (double)TextureSize;
                        context.V = y / (double)TextureSize;
                        raster[x] = Rasterization.ColorToUInt32(lambda(context));
                    }
                }
            }
            texture.UnlockRect(0);
            device.SetRenderTarget(0, rendertarget);
            device.SetDepthStencilSurface(depthstencil);
            device.BeginScene();
            device.Clear(D3DClear.Target | D3DClear.ZBuffer, 0x00000000, 1.0f, 0);
            device.SetFVF(D3DFvf.XYZ | D3DFvf.Normal | D3DFvf.Diffuse | D3DFvf.Tex1);
            device.SetRenderState(D3DRenderState.ZEnable, 1U);
            device.SetRenderState(D3DRenderState.CullMode, (uint)D3DCullMode.None);
            device.SetRenderState(D3DRenderState.Lighting, 0);
            device.SetTexture(0, texture);
            foreach (var transformedobject in TransformedObject.Enumerate(scene))
            {
                var A = PrimitiveAssembly.CreateTrianglesDX(transformedobject.Node.Primitive);
                var B = A.Select(i => new XYZNorDiffuseTex1 {
                    Xp = (float)i.Position.X, Yp = (float)i.Position.Y, Zp = (float)i.Position.Z,
                    Xn = (float)i.Normal.X, Yn = (float)i.Normal.Y, Zn = (float)i.Normal.Z,
                    Diffuse = Rasterization.ColorToUInt32(transformedobject.Node.WireColor),
                    U = (float)i.TexCoord.X, V = (float)i.TexCoord.Y,
                });
                var vertexbuffer = B.ToArray();
                device.SetTransform(D3DTransformState.Projection, Marshal.UnsafeAddrOfPinnedArrayElement(D3DMatrix.Convert(transformedobject.Transform * mvp), 0));
                device.DrawPrimitiveUP(RenderToy.D3DPrimitiveType.TriangleList, (uint)(vertexbuffer.Length / 3), Marshal.UnsafeAddrOfPinnedArrayElement(vertexbuffer, 0), (uint)Marshal.SizeOf(typeof(XYZNorDiffuseTex1)));
            }
            device.EndScene();
            D3DHelper.CopySurface(rendertarget, bitmap_ptr, render_width, render_height, bitmap_stride);
        }
        static Direct3D9 direct3d = new Direct3D9();
        #endregion
    }
}
