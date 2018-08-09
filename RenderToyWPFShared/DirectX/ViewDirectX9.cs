////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Cameras;
using RenderToy.DirectX;
using RenderToy.Shaders;
using System;
using System.Collections.Generic;
using System.Windows;
using System.Windows.Interop;
using System.Windows.Media;

namespace RenderToy.WPF
{
    public class ViewD3DImage : FrameworkElement
    {
        protected override void OnRender(DrawingContext drawingContext)
        {
            drawingContext.DrawImage(Target, new Rect(0, 0, ActualWidth, ActualHeight));
        }
        protected override Size MeasureOverride(Size availableSize)
        {
            return base.MeasureOverride(availableSize);
        }
        protected D3DImage Target = new D3DImage();
    }
    public class ViewD3DImageBuffered : ViewD3DImage
    {
        protected override Size MeasureOverride(Size availableSize)
        {
            NullablePtr<IntPtr> handle = new NullablePtr<IntPtr>(IntPtr.Zero);
            d3d9backbuffer = Direct3D9Helper.device.CreateRenderTarget((uint)availableSize.Width, (uint)availableSize.Height, D3DFormat.A8R8G8B8, D3DMultisample.None, 1, 0, handle);
            d3d9backbufferhandle = handle.Value;
            Target.Lock();
            Target.SetBackBuffer(D3DResourceType.IDirect3DSurface9, d3d9backbuffer.ManagedPtr);
            Target.Unlock();
            return base.MeasureOverride(availableSize);
        }
        // Direct3D9 Handling for D3DImage
        protected Direct3DSurface9 d3d9backbuffer;
        protected IntPtr d3d9backbufferhandle;
    }
    public abstract class ViewD3DImageDirect : ViewD3DImage
    {
        #region - Section : Overrides -
        protected abstract void RenderD3D();
        protected override void OnRender(DrawingContext drawingContext)
        {
            if (rendertarget == null || depthstencil == null) return;
            if (Target.TryLock(new Duration(TimeSpan.FromMilliseconds(500))))
            {
                Direct3D9Helper.device.SetRenderTarget(0, rendertarget);
                Direct3D9Helper.device.SetDepthStencilSurface(depthstencil);
                Direct3D9Helper.device.BeginScene();
                Direct3D9Helper.device.Clear(D3DClear.Target | D3DClear.ZBuffer, 0x00000000, 1.0f, 0);
                Direct3D9Helper.device.SetRenderState(D3DRenderState.ZEnable, 1U);
                Direct3D9Helper.device.SetRenderState(D3DRenderState.CullMode, (uint)D3DCullMode.None);
                Direct3D9Helper.device.SetRenderState(D3DRenderState.Lighting, 0);
                for (int i = 0; i < 8; ++i)
                {
                    Direct3D9Helper.device.SetSamplerState((uint)i, D3DSamplerState.MagFilter, (uint)D3DTextureFilter.Anisotropic);
                    Direct3D9Helper.device.SetSamplerState((uint)i, D3DSamplerState.MinFilter, (uint)D3DTextureFilter.Anisotropic);
                    Direct3D9Helper.device.SetSamplerState((uint)i, D3DSamplerState.MipFilter, (uint)D3DTextureFilter.Linear);
                    Direct3D9Helper.device.SetSamplerState((uint)i, D3DSamplerState.MaxAnisotropy, (uint)16);
                }
                Direct3D9Helper.device.SetVertexDeclaration(Direct3D9Helper.vertexdeclaration);
                RenderD3D();
                Direct3D9Helper.device.EndScene();
                Target.SetBackBuffer(D3DResourceType.IDirect3DSurface9, rendertarget.ManagedPtr);
                Target.AddDirtyRect(new Int32Rect(0, 0, render_width, render_height));
            }
            Target.Unlock();
            drawingContext.DrawImage(Target, new Rect(0, 0, ActualWidth, ActualHeight));
        }
        protected override Size MeasureOverride(Size availableSize)
        {
            render_width = (int)availableSize.Width;
            render_height = (int)availableSize.Height;
            rendertarget = Direct3D9Helper.device.CreateRenderTarget((uint)render_width, (uint)render_height, D3DFormat.A8R8G8B8, D3DMultisample.None, 0, 0, null);
            depthstencil = Direct3D9Helper.device.CreateDepthStencilSurface((uint)render_width, (uint)render_height, D3DFormat.D24X8, D3DMultisample.None, 0, 0, null);
            return base.MeasureOverride(availableSize);
        }
        #endregion
        #region - Section : Private Fields -
        Direct3DSurface9 rendertarget;
        Direct3DSurface9 depthstencil;
        int render_width;
        int render_height;
        #endregion
    }
    public class ViewDirectX9FixedFunction : ViewD3DImageDirect
    {
        protected override void RenderD3D()
        {
            var transformCamera = AttachedView.GetTransformCamera(this);
            var transformView = AttachedView.GetTransformView(this);
            var transformProjection = AttachedView.GetTransformProjection(this) * Perspective.AspectCorrectFit(ActualWidth, ActualHeight);
            var transformViewProjection = transformView * transformProjection;
            var constants = new Dictionary<string, object>();
            constants["transformCamera"] = transformCamera;
            constants["transformView"] = transformView;
            constants["transformProjection"] = transformProjection;
            constants["transformViewProjection"] = transformViewProjection;
            Direct3D9Helper.CreateSceneDrawFixedFunction(AttachedView.GetScene(this))(constants);
        }
    }
    public class ViewDirectX9 : ViewD3DImageDirect
    {
        public static DependencyProperty VertexShaderProperty = DependencyProperty.Register("VertexShader", typeof(byte[]), typeof(ViewDirectX9), new FrameworkPropertyMetadata(HLSL.D3D9VS, FrameworkPropertyMetadataOptions.AffectsRender));
        public byte[] VertexShader
        {
            get { return (byte[])GetValue(VertexShaderProperty); }
            set { SetValue(VertexShaderProperty, value); }
        }
        public static DependencyProperty PixelShaderProperty = DependencyProperty.Register("PixelShader", typeof(byte[]), typeof(ViewDirectX9), new FrameworkPropertyMetadata(HLSL.D3D9PS, FrameworkPropertyMetadataOptions.AffectsRender));
        public byte[] PixelShader
        {
            get { return (byte[])GetValue(PixelShaderProperty); }
            set { SetValue(PixelShaderProperty, value); }
        }
        protected override void RenderD3D()
        {
            if (VertexShader == null || PixelShader == null) return;
            var vertexshader = Direct3D9Helper.device.CreateVertexShader(VertexShader);
            var pixelshader = Direct3D9Helper.device.CreatePixelShader(PixelShader);
            Direct3D9Helper.device.SetVertexShader(vertexshader);
            Direct3D9Helper.device.SetPixelShader(pixelshader);
            var transformCamera = AttachedView.GetTransformCamera(this);
            var transformView = AttachedView.GetTransformView(this);
            var transformProjection = AttachedView.GetTransformProjection(this) * Perspective.AspectCorrectFit(ActualWidth, ActualHeight);
            var transformViewProjection = transformView * transformProjection;
            var constants = new Dictionary<string, object>();
            constants["transformCamera"] = transformCamera;
            constants["transformView"] = transformView;
            constants["transformProjection"] = transformProjection;
            constants["transformViewProjection"] = transformViewProjection;
            Direct3D9Helper.CreateSceneDraw(AttachedView.GetScene(this))(constants);
        }
    }
}