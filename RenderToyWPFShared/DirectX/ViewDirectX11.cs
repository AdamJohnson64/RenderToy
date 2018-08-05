////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToyCOM;
using RenderToy.Cameras;
using RenderToy.Diagnostics;
using RenderToy.DirectX;
using RenderToy.DocumentModel;
using RenderToy.Math;
using RenderToy.Shaders;
using RenderToy.Utility;
using System;
using System.Diagnostics;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Forms;
using System.Windows.Interop;
using System.Windows.Media;

namespace RenderToy.WPF
{
    class ViewDirectX11 : FrameworkElement
    {
        public static DependencyProperty VertexShaderProperty = DependencyProperty.Register("VertexShader", typeof(byte[]), typeof(ViewDirectX11), new FrameworkPropertyMetadata(HLSL.D3D11VS, OnVertexShaderChanged));
        public byte[] VertexShader
        {
            get { return (byte[])GetValue(VertexShaderProperty); }
            set { SetValue(VertexShaderProperty, value); }
        }
        private static void OnVertexShaderChanged(DependencyObject sender, DependencyPropertyChangedEventArgs e)
        {
            ((ViewDirectX11)sender).SetVertexShader((byte[])e.NewValue);
        }
        public static DependencyProperty PixelShaderProperty = DependencyProperty.Register("PixelShader", typeof(byte[]), typeof(ViewDirectX11), new FrameworkPropertyMetadata(HLSL.D3D11PS, OnPixelShaderChanged));
        public byte[] PixelShader
        {
            get { return (byte[])GetValue(PixelShaderProperty); }
            set { SetValue(PixelShaderProperty, value); }
        }
        private static void OnPixelShaderChanged(DependencyObject sender, DependencyPropertyChangedEventArgs e)
        {
            ((ViewDirectX11)sender).SetPixelShader((byte[])e.NewValue);
        }
        static ViewDirectX11()
        {
            AttachedView.SceneProperty.OverrideMetadata(typeof(ViewDirectX11), new FrameworkPropertyMetadata(null, (s, e) =>
            {
                ((ViewDirectX11)s).Execute_DrawScene = DirectX11Helper.CreateSceneDraw((SparseScene)e.NewValue);
                ((ViewDirectX11)s).RenderDX();
            }));
            AttachedView.TransformModelViewProjectionProperty.OverrideMetadata(typeof(ViewDirectX11), new FrameworkPropertyMetadata(Matrix3D.Identity, (s, e) =>
            {
                ((ViewDirectX11)s).RenderDX();
            }));
        }
        Action<ID3D11DeviceContext4, Matrix3D, Matrix3D, string> Execute_DrawScene = null;
        void RenderDX()
        {
            if (d3d11VertexShader == null || d3d11PixelShader == null || d3d11DepthStencilView == null || d3d11RenderTargetView == null) return;
            ID3D11DeviceContext context_old = null;
            DirectX11Helper.d3d11Device.GetImmediateContext(ref context_old);
            var context = (ID3D11DeviceContext4)context_old;
            var desc = new D3D11_TEXTURE2D_DESC();
            d3d11Texture2D_DS.GetDesc(ref desc);
            int width = (int)desc.Width;
            int height = (int)desc.Height;
            RenderToyEventSource.Default.RenderBegin();
            // Setup common global render state.
            ID3D11ClassInstance classInstance = null;
            context.VSSetShader(d3d11VertexShader, classInstance, 0);
            context.PSSetShader(d3d11PixelShader, classInstance, 0);
            // Draw the window view using the current camera.
            var scissorRect = new tagRECT { left = 0, top = 0, right = width, bottom = height };
            context.RSSetScissorRects(1, scissorRect);
            var viewportRect = new D3D11_VIEWPORT { TopLeftX = 0, TopLeftY = 0, Width = width, Height = height, MinDepth = 0, MaxDepth = 1 };
            context.RSSetViewports(1, viewportRect);
            context.OMSetRenderTargets(1, d3d11RenderTargetView, d3d11DepthStencilView);
            context.ClearDepthStencilView(d3d11DepthStencilView, (uint)D3D11_CLEAR_FLAG.D3D11_CLEAR_DEPTH, 1, 0);
            context.ClearRenderTargetView(d3d11RenderTargetView, new float[] { 0, 0, 0, 0 });
            if (Execute_DrawScene != null)
            {
                Execute_DrawScene(context, AttachedView.GetTransformCamera(this), AttachedView.GetTransformModelViewProjection(this) * Perspective.AspectCorrectFit(ActualWidth, ActualHeight), "Window");
            }
            context.Flush();
            d3dimage.Lock();
            d3dimage.AddDirtyRect(new Int32Rect(0, 0, width, height));
            d3dimage.Unlock();
            RenderToyEventSource.Default.RenderEnd();
        }
        void SetVertexShader(byte[] bytecode)
        {
            if (bytecode != null)
            {
                ID3D11ClassLinkage linkage = null;
                DirectX11Helper.d3d11Device.CreateVertexShader(UnmanagedCopy.Create(bytecode), (ulong)bytecode.Length, linkage, ref d3d11VertexShader);
            }
        }
        void SetPixelShader(byte[] bytecode)
        {
            if (bytecode != null)
            {
                ID3D11ClassLinkage linkage = null;
                DirectX11Helper.d3d11Device.CreatePixelShader(UnmanagedCopy.Create(bytecode), (ulong)bytecode.Length, linkage, ref d3d11PixelShader);
            }
        }
        protected override Size MeasureOverride(Size availableSize)
        {
            NullablePtr<IntPtr> handle = new NullablePtr<IntPtr>(IntPtr.Zero);
            d3d9backbuffer = d3d9device.CreateRenderTarget((uint)availableSize.Width, (uint)availableSize.Height, D3DFormat.A8R8G8B8, D3DMultisample.None, 1, 0, handle);
            var texptr = new IntPtr();
            DirectX11Helper.d3d11Device.OpenSharedResource(handle.Value, Marshal.GenerateGuidForType(typeof(ID3D11Texture2D)), ref texptr);
            d3d11Texture2D_RT = (ID3D11Texture2D)Marshal.GetTypedObjectForIUnknown(texptr, typeof(ID3D11Texture2D));
            var rtvd = new D3D11_RENDER_TARGET_VIEW_DESC { Format = DXGI_FORMAT.DXGI_FORMAT_B8G8R8A8_UNORM, ViewDimension = D3D11_RTV_DIMENSION.D3D11_RTV_DIMENSION_TEXTURE2D };
            rtvd.__MIDL____MIDL_itf_RenderToy_0005_00650002.Texture2D = new D3D11_TEX2D_RTV { MipSlice = 0 };
            DirectX11Helper.d3d11Device.CreateRenderTargetView(d3d11Texture2D_RT, rtvd, ref d3d11RenderTargetView);
            var d3d11Texture2DDesc_DS = new D3D11_TEXTURE2D_DESC { Width = (uint)availableSize.Width, Height = (uint)availableSize.Height, MipLevels = 1, ArraySize = 1, Format = DXGI_FORMAT.DXGI_FORMAT_D32_FLOAT, SampleDesc = new DXGI_SAMPLE_DESC { Count = 1, Quality = 0 }, Usage = D3D11_USAGE.D3D11_USAGE_DEFAULT, BindFlags = (uint)D3D11_BIND_FLAG.D3D11_BIND_DEPTH_STENCIL, CPUAccessFlags = 0 };
            unsafe
            {
                D3D11_SUBRESOURCE_DATA* subresource = null;
                DirectX11Helper.d3d11Device.CreateTexture2D(d3d11Texture2DDesc_DS, ref *subresource, ref d3d11Texture2D_DS);
            }
            var dsv = new D3D11_DEPTH_STENCIL_VIEW_DESC { Format = DXGI_FORMAT.DXGI_FORMAT_D32_FLOAT, ViewDimension = D3D11_DSV_DIMENSION.D3D11_DSV_DIMENSION_TEXTURE2D };
            dsv.__MIDL____MIDL_itf_RenderToy_0005_00660000.Texture2D = new D3D11_TEX2D_DSV { MipSlice = 0 };
            DirectX11Helper.d3d11Device.CreateDepthStencilView(d3d11Texture2D_DS, dsv, ref d3d11DepthStencilView);
            d3dimage.Lock();
            d3dimage.SetBackBuffer(D3DResourceType.IDirect3DSurface9, d3d9backbuffer.ManagedPtr);
            d3dimage.Unlock();
            RenderDX();
            return base.MeasureOverride(availableSize);
        }
        protected override void OnRender(DrawingContext drawingContext)
        {
            drawingContext.DrawImage(d3dimage, new Rect(0, 0, ActualWidth, ActualHeight));
        }
        // Direct3D9 Handling for D3DImage
        static Direct3D9Ex d3d9 = new Direct3D9Ex();
        static Form d3d9window = new Form();
        static Direct3DDevice9Ex d3d9device = d3d9.CreateDevice(d3d9window.Handle);
        Direct3DSurface9 d3d9backbuffer;
        D3DImage d3dimage = new D3DImage();
        // Direct3D11 Handling
        ID3D11VertexShader d3d11VertexShader;
        ID3D11PixelShader d3d11PixelShader;
        ID3D11Texture2D d3d11Texture2D_RT;
        ID3D11Texture2D d3d11Texture2D_DS;
        ID3D11RenderTargetView d3d11RenderTargetView;
        ID3D11DepthStencilView d3d11DepthStencilView;
    }
}