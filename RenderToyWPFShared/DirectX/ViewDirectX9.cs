////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Cameras;
using RenderToy.DirectX;
using RenderToy.Materials;
using RenderToy.Meshes;
using RenderToy.ModelFormat;
using RenderToy.Primitives;
using RenderToy.Shaders;
using System;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Interop;
using System.Windows.Media;

namespace RenderToy.WPF
{
    // Use a D3DImage to push the buffer directly to WDDM.
    // No copy is made; Fast but not always viable.
    public class ViewD3DSource : FrameworkElement
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
    public abstract class ViewDirectX9Base : ViewD3DSource
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
                var vertexdeclaration = Direct3D9Helper.device.CreateVertexDeclaration(new D3DVertexElement9[] {
                    new D3DVertexElement9 { Stream = 0, Offset = 0, Type = D3DDeclType.Float3, Method = D3DDeclMethod.Default, Usage = D3DDeclUsage.Position, UsageIndex = 0 },
                    new D3DVertexElement9 { Stream = 0, Offset = 12, Type = D3DDeclType.Float3, Method = D3DDeclMethod.Default, Usage = D3DDeclUsage.Normal, UsageIndex = 0 },
                    new D3DVertexElement9 { Stream = 0, Offset = 24, Type = D3DDeclType.D3DColor, Method = D3DDeclMethod.Default, Usage = D3DDeclUsage.Color, UsageIndex = 0 },
                    new D3DVertexElement9 { Stream = 0, Offset = 28, Type = D3DDeclType.Float2, Method = D3DDeclMethod.Default, Usage = D3DDeclUsage.TexCoord, UsageIndex = 0 },
                    new D3DVertexElement9 { Stream = 0, Offset = 36, Type = D3DDeclType.Float3, Method = D3DDeclMethod.Default, Usage = D3DDeclUsage.Tangent, UsageIndex = 0 },
                    new D3DVertexElement9 { Stream = 0, Offset = 48, Type = D3DDeclType.Float3, Method = D3DDeclMethod.Default, Usage = D3DDeclUsage.Binormal, UsageIndex = 0 },
                });
                Direct3D9Helper.device.SetVertexDeclaration(vertexdeclaration);
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
    public class ViewDirectX9FixedFunction : ViewDirectX9Base
    {
        protected override void RenderD3D()
        {
            var mvp = AttachedView.GetTransformModelViewProjection(this) * Perspective.AspectCorrectFit(ActualWidth, ActualHeight);
            foreach (var transformedobject in AttachedView.GetScene(this))
            {
                var createdvertexbuffer = Direct3D9Helper.CreateVertexBuffer(transformedobject.NodePrimitive);
                if (createdvertexbuffer.VertexBuffer == null) continue;
                Direct3D9Helper.device.SetStreamSource(0, createdvertexbuffer.VertexBuffer, 0U, (uint)Marshal.SizeOf(typeof(XYZNorDiffuseTex1)));
                Direct3D9Helper.device.SetTexture(0, Direct3D9Helper.CreateTexture(transformedobject.NodeMaterial, null));
                Direct3D9Helper.device.SetTransform(D3DTransformState.Projection, Marshal.UnsafeAddrOfPinnedArrayElement(DirectXHelper.ConvertToD3DMatrix(transformedobject.Transform * mvp), 0));
                Direct3D9Helper.device.DrawPrimitive(D3DPrimitiveType.TriangleList, 0U, (uint)createdvertexbuffer.PrimitiveCount);
            }
        }
    }
    public class ViewDirectX9 : ViewDirectX9Base
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
            foreach (var transformedobject in AttachedView.GetScene(this))
            {
                if (transformedobject.NodePrimitive == null) continue;
                var transformModel = transformedobject.Transform;
                var transformModelViewProjection = transformModel * transformViewProjection;
                var createdvertexbuffer = Direct3D9Helper.CreateVertexBuffer(transformedobject.NodePrimitive);
                if (createdvertexbuffer.VertexBuffer == null) continue;
                Direct3D9Helper.device.SetStreamSource(0, createdvertexbuffer.VertexBuffer, 0U, (uint)Marshal.SizeOf(typeof(XYZNorDiffuseTex1)));
                var objmat = transformedobject.NodeMaterial as LoaderOBJ.OBJMaterial;
                Direct3D9Helper.device.SetTexture(0, Direct3D9Helper.CreateTexture(objmat == null ? transformedobject.NodeMaterial : objmat.map_Kd, StockMaterials.PlasticWhite));
                Direct3D9Helper.device.SetTexture(1, Direct3D9Helper.CreateTexture(objmat == null ? null : objmat.map_d, StockMaterials.PlasticWhite));
                Direct3D9Helper.device.SetTexture(2, Direct3D9Helper.CreateTexture(objmat == null ? null : objmat.map_bump, StockMaterials.PlasticLightBlue));
                Direct3D9Helper.device.SetTexture(3, Direct3D9Helper.CreateTexture(objmat == null ? null : objmat.displacement, StockMaterials.PlasticWhite));
                Direct3D9Helper.device.SetVertexShaderConstantF(0, Marshal.UnsafeAddrOfPinnedArrayElement(DirectXHelper.ConvertToD3DMatrix(transformCamera), 0), 4);
                Direct3D9Helper.device.SetVertexShaderConstantF(4, Marshal.UnsafeAddrOfPinnedArrayElement(DirectXHelper.ConvertToD3DMatrix(transformModel), 0), 4);
                Direct3D9Helper.device.SetVertexShaderConstantF(8, Marshal.UnsafeAddrOfPinnedArrayElement(DirectXHelper.ConvertToD3DMatrix(transformView), 0), 4);
                Direct3D9Helper.device.SetVertexShaderConstantF(12, Marshal.UnsafeAddrOfPinnedArrayElement(DirectXHelper.ConvertToD3DMatrix(transformProjection), 0), 4);
                Direct3D9Helper.device.SetVertexShaderConstantF(16, Marshal.UnsafeAddrOfPinnedArrayElement(DirectXHelper.ConvertToD3DMatrix(transformModelViewProjection), 0), 4);
                Direct3D9Helper.device.DrawPrimitive(D3DPrimitiveType.TriangleList, 0U, (uint)createdvertexbuffer.PrimitiveCount);
            }
        }
    }
}