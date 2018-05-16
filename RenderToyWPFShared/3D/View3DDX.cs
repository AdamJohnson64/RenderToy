﻿////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Cameras;
using RenderToy.Materials;
using RenderToy.PipelineModel;
using RenderToy.SceneGraph;
using RenderToy.Utility;
using System;
using System.Linq;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Data;
using System.Windows.Input;
using System.Windows.Interop;
using System.Windows.Media;

namespace RenderToy.WPF
{
    public class View3DDX : FrameworkElement
    {
        public static DependencyProperty SceneProperty = DependencyProperty.Register("Scene", typeof(IScene), typeof(View3DDX), new FrameworkPropertyMetadata(null, FrameworkPropertyMetadataOptions.AffectsRender));
        public IScene Scene
        {
            get { return (IScene)GetValue(SceneProperty); }
            set { SetValue(SceneProperty, value); }
        }
        public static DependencyProperty ModelViewProjectionProperty = DependencyProperty.Register("ModelViewProjection", typeof(Matrix3D), typeof(View3DDX), new FrameworkPropertyMetadata(Matrix3D.Identity, FrameworkPropertyMetadataOptions.AffectsRender));
        public Matrix3D ModelViewProjection
        {
            get { return (Matrix3D)GetValue(ModelViewProjectionProperty); }
            set { SetValue(ModelViewProjectionProperty, value); }
        }
        public View3DDX()
        {
            d3dimage = new D3DImage();
            d3d = new Direct3D9();
            device = d3d.CreateDevice();
        }
        class Token
        {
        }
        static readonly Token GeneratedTextureToken = new Token();
        static readonly Token GeneratedVertexBufferToken = new Token();
        struct VertexBufferInfo
        {
            public Direct3DVertexBuffer9 VertexBuffer;
            public int PrimitiveCount;
        }
        protected override void OnRender(DrawingContext drawingContext)
        {
            if (rendertarget == null) return;
            device.SetRenderTarget(0, rendertarget);
            device.SetDepthStencilSurface(depthstencil);
            device.BeginScene();
            device.Clear(D3DClear.Target | D3DClear.ZBuffer, 0x00000000, 1.0f, 0);
            device.SetFVF(D3DFvf.XYZ | D3DFvf.Normal | D3DFvf.Diffuse | D3DFvf.Tex1);
            device.SetRenderState(D3DRenderState.ZEnable, 1U);
            device.SetRenderState(D3DRenderState.CullMode, (uint)D3DCullMode.None);
            device.SetRenderState(D3DRenderState.Lighting, 0);
            var mvp = ModelViewProjection * Perspective.AspectCorrectFit(ActualWidth, ActualHeight);
            foreach (var transformedobject in TransformedObject.Enumerate(Scene))
            {
                var createdvertexbuffer = MementoServer.Get(transformedobject.Node.GetPrimitive(), GeneratedVertexBufferToken, () =>
                {
                    var verticesin = PrimitiveAssembly.CreateTrianglesDX(transformedobject.Node.GetPrimitive());
                    var verticesout = verticesin.Select(i => new RenderD3D.XYZNorDiffuseTex1
                    {
                        Xp = (float)i.Position.X,
                        Yp = (float)i.Position.Y,
                        Zp = (float)i.Position.Z,
                        Xn = (float)i.Normal.X,
                        Yn = (float)i.Normal.Y,
                        Zn = (float)i.Normal.Z,
                        Diffuse = Rasterization.ColorToUInt32(transformedobject.Node.GetWireColor()),
                        U = (float)i.TexCoord.X,
                        V = (float)i.TexCoord.Y,
                    });
                    var data = verticesout.ToArray();
                    var size = (uint)(Marshal.SizeOf(typeof(RenderD3D.XYZNorDiffuseTex1)) * data.Length);
                    VertexBufferInfo buffer = new VertexBufferInfo();
                    if (data.Length > 0)
                    {
                        buffer.VertexBuffer = device.CreateVertexBuffer(size, 0, (uint)(D3DFvf.XYZ | D3DFvf.Normal | D3DFvf.Diffuse | D3DFvf.Tex1), D3DPool.Managed);
                        var locked = buffer.VertexBuffer.Lock(0U, size, 0U);
                        unsafe
                        {
                            Buffer.MemoryCopy(Marshal.UnsafeAddrOfPinnedArrayElement(data, 0).ToPointer(), locked.ToPointer(), size, size);
                        }
                        buffer.VertexBuffer.Unlock();
                    }
                    buffer.PrimitiveCount = data.Length / 3;
                    return buffer;
                });
                if (createdvertexbuffer.VertexBuffer == null) continue;
                var createdtexture = MementoServer.Get(transformedobject.Node.GetMaterial(), GeneratedTextureToken, () =>
                {
                    const int TextureSize = 256;
                    var material = transformedobject.Node.GetMaterial() as IMNNode<Vector4D>;
                    if (material == null) return null;
                    var texture = device.CreateTexture((uint)TextureSize, (uint)TextureSize, 1, 0U, D3DFormat.A8R8G8B8, D3DPool.Managed);
                    D3DLockedRect lockit = texture.LockRect(0);
                    EvalContext context = new EvalContext();
                    unsafe
                    {
                        for (int y = 0; y < TextureSize; ++y)
                        {
                            uint* raster = (uint*)((byte*)lockit.Bits + lockit.Pitch * y);
                            for (int x = 0; x < TextureSize; ++x)
                            {
                                context.U = x / (double)TextureSize;
                                context.V = y / (double)TextureSize;
                                raster[x] = Rasterization.ColorToUInt32(material.Eval(context));
                            }
                        }
                    }
                    return texture;
                });
                device.SetStreamSource(0, createdvertexbuffer.VertexBuffer, 0U, (uint)Marshal.SizeOf(typeof(RenderD3D.XYZNorDiffuseTex1)));
                device.SetTexture(0, createdtexture);
                device.SetTransform(D3DTransformState.Projection, Marshal.UnsafeAddrOfPinnedArrayElement(D3DMatrix.Convert(transformedobject.Transform * mvp), 0));
                device.DrawPrimitive(RenderToy.D3DPrimitiveType.TriangleList, 0U, (uint)createdvertexbuffer.PrimitiveCount);
            }
            device.EndScene();
            d3dimage.Lock();
            d3dimage.SetBackBuffer(D3DResourceType.IDirect3DSurface9, rendertarget.ManagedPtr);
            d3dimage.AddDirtyRect(new Int32Rect(0, 0, render_width, render_height));
            d3dimage.Unlock();
            drawingContext.DrawImage(d3dimage, new Rect(0, 0, ActualWidth, ActualHeight));
        }
        protected override void OnRenderSizeChanged(SizeChangedInfo sizeInfo)
        {
            render_width = (int)sizeInfo.NewSize.Width;
            render_height = (int)sizeInfo.NewSize.Height;
            rendertarget = device.CreateRenderTarget((uint)render_width, (uint)render_height, D3DFormat.A8R8G8B8, D3DMultisample.None, 0, 1);
            depthstencil = device.CreateDepthStencilSurface((uint)render_width, (uint)render_height, D3DFormat.D24X8, D3DMultisample.None, 0, 1);
            InvalidateVisual();
        }
        D3DImage d3dimage;
        Direct3D9 d3d;
        Direct3DDevice9 device;
        Direct3DSurface9 rendertarget;
        Direct3DSurface9 depthstencil;
        int render_width;
        int render_height;
    }
}