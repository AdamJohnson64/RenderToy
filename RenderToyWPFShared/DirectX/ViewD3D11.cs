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
using System.Runtime.InteropServices;
using System.Windows;
using System.Collections.Generic;
using System.Windows.Threading;

namespace RenderToy.WPF
{
    class ViewD3D11 : ViewD3DImageBuffered
    {
        DispatcherTimer timer = null;
        public ViewD3D11()
        {
            timer = new DispatcherTimer(TimeSpan.FromMilliseconds(1000 / 60), DispatcherPriority.ApplicationIdle, (s, e) => { RenderDX(); }, Application.Current.Dispatcher);
            timer.Start();
        }
        public static DependencyProperty VertexShaderProperty = DependencyProperty.Register("VertexShader", typeof(byte[]), typeof(ViewD3D11), new FrameworkPropertyMetadata(HLSL.D3D11VS, OnVertexShaderChanged));
        public byte[] VertexShader
        {
            get { return (byte[])GetValue(VertexShaderProperty); }
            set { SetValue(VertexShaderProperty, value); }
        }
        private static void OnVertexShaderChanged(DependencyObject sender, DependencyPropertyChangedEventArgs e)
        {
            if (e.NewValue is byte[] bytecode)
            {
                Direct3D11Helper.Dispatcher.Invoke(() =>
                {
                    Direct3D11Helper.d3d11Device.CreateVertexShader(UnmanagedCopy.Create(bytecode), (ulong)bytecode.Length, null, ref ((ViewD3D11)sender).d3d11VertexShader);
                });
            }
        }
        public static DependencyProperty PixelShaderProperty = DependencyProperty.Register("PixelShader", typeof(byte[]), typeof(ViewD3D11), new FrameworkPropertyMetadata(HLSL.D3D11PS, OnPixelShaderChanged));
        public byte[] PixelShader
        {
            get { return (byte[])GetValue(PixelShaderProperty); }
            set { SetValue(PixelShaderProperty, value); }
        }
        private static void OnPixelShaderChanged(DependencyObject sender, DependencyPropertyChangedEventArgs e)
        {
            if (e.NewValue is byte[] bytecode)
            {
                Direct3D11Helper.Dispatcher.Invoke(() =>
                {
                    Direct3D11Helper.d3d11Device.CreatePixelShader(UnmanagedCopy.Create(bytecode), (ulong)bytecode.Length, null, ref ((ViewD3D11)sender).d3d11PixelShader);
                });
            }
        }
        static ViewD3D11()
        {
            AttachedView.SceneProperty.OverrideMetadata(typeof(ViewD3D11), new FrameworkPropertyMetadata(null, (s, e) =>
            {
                ((ViewD3D11)s).Execute_DrawScene = Direct3D11Helper.CreateSceneDraw((SparseScene)e.NewValue);
                ((ViewD3D11)s).RenderDX();
            }));
            AttachedView.TransformModelViewProjectionProperty.OverrideMetadata(typeof(ViewD3D11), new FrameworkPropertyMetadata(Matrix3D.Identity, (s, e) =>
            {
                ((ViewD3D11)s).RenderDX();
            }));
        }
        Action<ID3D11DeviceContext4, Dictionary<string, object>> Execute_DrawScene = null;
        void RenderDX()
        {
            if (d3d11VertexShader == null || d3d11PixelShader == null || d3d11DepthStencilView == null || d3d11RenderTargetView == null) return;
            var constants = new Dictionary<string, object>();
            constants["profilingName"] = "Window";
            constants["transformAspect"] = Perspective.AspectCorrectFit(ActualWidth, ActualHeight);
            constants["transformCamera"] = AttachedView.GetTransformCamera(this);
            constants["transformView"] = AttachedView.GetTransformView(this);
            constants["transformProjection"] = AttachedView.GetTransformProjection(this); ;
            var desc = new D3D11_TEXTURE2D_DESC();
            Direct3D11Helper.Dispatcher.Invoke(() =>
            {
                RenderToyEventSource.Default.RenderBegin();
                ID3D11DeviceContext context_old = null;
                Direct3D11Helper.d3d11Device.GetImmediateContext(ref context_old);
                var context = (ID3D11DeviceContext4)context_old;
                d3d11Texture2D_DS.GetDesc(ref desc);
                // Setup common global render state.
                context.VSSetShader(d3d11VertexShader, null, 0);
                context.PSSetShader(d3d11PixelShader, null, 0);
                // Draw the window view using the current camera.
                var scissorRect = new tagRECT { left = 0, top = 0, right = (int)desc.Width, bottom = (int)desc.Height };
                context.RSSetScissorRects(1, scissorRect);
                var viewportRect = new D3D11_VIEWPORT { TopLeftX = 0, TopLeftY = 0, Width = (int)desc.Width, Height = (int)desc.Height, MinDepth = 0, MaxDepth = 1 };
                context.RSSetViewports(1, viewportRect);
                context.OMSetRenderTargets(1, d3d11RenderTargetView, d3d11DepthStencilView);
                context.ClearDepthStencilView(d3d11DepthStencilView, (uint)D3D11_CLEAR_FLAG.D3D11_CLEAR_DEPTH, 1, 0);
                context.ClearRenderTargetView(d3d11RenderTargetView, new float[] { 0, 0, 0, 0 });
                if (Execute_DrawScene != null)
                {
                    Execute_DrawScene(context, constants);
                }
                context.Flush();
                RenderToyEventSource.Default.RenderEnd();
            });
            Target.Lock();
            Target.AddDirtyRect(new Int32Rect(0, 0, (int)desc.Width, (int)desc.Height));
            Target.Unlock();
        }
        protected override Size MeasureOverride(Size availableSize)
        {
            var size = base.MeasureOverride(availableSize);
            Direct3D11Helper.Dispatcher.Invoke(() =>
            {
                var texptr = new IntPtr();
                Direct3D11Helper.d3d11Device.OpenSharedResource(d3d9backbufferhandle, Marshal.GenerateGuidForType(typeof(ID3D11Texture2D)), ref texptr);
                d3d11Texture2D_RT = (ID3D11Texture2D)Marshal.GetTypedObjectForIUnknown(texptr, typeof(ID3D11Texture2D));
                var rtvd = new D3D11_RENDER_TARGET_VIEW_DESC { Format = DXGI_FORMAT.DXGI_FORMAT_B8G8R8A8_UNORM, ViewDimension = D3D11_RTV_DIMENSION.D3D11_RTV_DIMENSION_TEXTURE2D };
                rtvd.__MIDL____MIDL_itf_RenderToy_0005_00650002.Texture2D = new D3D11_TEX2D_RTV { MipSlice = 0 };
                Direct3D11Helper.d3d11Device.CreateRenderTargetView(d3d11Texture2D_RT, rtvd, ref d3d11RenderTargetView);
                var d3d11Texture2DDesc_DS = new D3D11_TEXTURE2D_DESC { Width = (uint)availableSize.Width, Height = (uint)availableSize.Height, MipLevels = 1, ArraySize = 1, Format = DXGI_FORMAT.DXGI_FORMAT_D32_FLOAT, SampleDesc = new DXGI_SAMPLE_DESC { Count = 1, Quality = 0 }, Usage = D3D11_USAGE.D3D11_USAGE_DEFAULT, BindFlags = (uint)D3D11_BIND_FLAG.D3D11_BIND_DEPTH_STENCIL, CPUAccessFlags = 0 };
                unsafe
                {
                    D3D11_SUBRESOURCE_DATA* subresource = null;
                    Direct3D11Helper.d3d11Device.CreateTexture2D(d3d11Texture2DDesc_DS, ref *subresource, ref d3d11Texture2D_DS);
                }
                var dsv = new D3D11_DEPTH_STENCIL_VIEW_DESC { Format = DXGI_FORMAT.DXGI_FORMAT_D32_FLOAT, ViewDimension = D3D11_DSV_DIMENSION.D3D11_DSV_DIMENSION_TEXTURE2D };
                dsv.__MIDL____MIDL_itf_RenderToy_0005_00660000.Texture2D = new D3D11_TEX2D_DSV { MipSlice = 0 };
                Direct3D11Helper.d3d11Device.CreateDepthStencilView(d3d11Texture2D_DS, dsv, ref d3d11DepthStencilView);
            });
            RenderDX();
            return size;
        }
        // Direct3D11 Handling
        ID3D11VertexShader d3d11VertexShader;
        ID3D11PixelShader d3d11PixelShader;
        ID3D11Texture2D d3d11Texture2D_RT;
        ID3D11Texture2D d3d11Texture2D_DS;
        ID3D11RenderTargetView d3d11RenderTargetView;
        ID3D11DepthStencilView d3d11DepthStencilView;
    }
}