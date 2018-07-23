using RenderToy.Cameras;
using RenderToy.DirectX;
using RenderToy.DocumentModel;
using RenderToy.Math;
using RenderToy.Shaders;
using System;
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
        Action<D3D11DeviceContext4, Matrix3D> Execute_DrawScene = null;
        void RenderDX()
        {
            if (d3d11VertexShader == null || d3d11PixelShader == null || d3d11DepthStencilView == null || d3d11RenderTargetView == null) return;
            var contextold = DirectX11Helper.d3d11Device.GetImmediateContext();
            var context = contextold.QueryInterfaceD3D11DeviceContext4();
            int width = d3d11Texture2D_RT.GetWidth();
            int height = d3d11Texture2D_RT.GetHeight();
            RenderToyETWEventSource.Default.BeginFrame();
            // Setup common global render state.
            context.VSSetShader(d3d11VertexShader);
            context.PSSetShader(d3d11PixelShader);
            // Draw the window view using the current camera.
            context.RSSetScissorRects(new[] { new D3D11Rect { left = 0, top = 0, right = width, bottom = height } });
            context.RSSetViewports(new[] { new D3D11Viewport { TopLeftX = 0, TopLeftY = 0, Width = width, Height = height, MinDepth = 0, MaxDepth = 1 } });
            context.OMSetRenderTargets(new[] { d3d11RenderTargetView }, d3d11DepthStencilView);
            context.ClearDepthStencilView(d3d11DepthStencilView, D3D11ClearFlag.Depth, 1, 0);
            context.ClearRenderTargetView(d3d11RenderTargetView, 0, 0, 0, 0);
            if (Execute_DrawScene != null)
            {
                Execute_DrawScene(context, AttachedView.GetTransformModelViewProjection(this) * Perspective.AspectCorrectFit(ActualWidth, ActualHeight));
            }
            context.Flush();
            d3dimage.Lock();
            d3dimage.AddDirtyRect(new Int32Rect(0, 0, d3d11Texture2D_RT.GetWidth(), d3d11Texture2D_RT.GetHeight()));
            d3dimage.Unlock();
            RenderToyETWEventSource.Default.EndFrame();
        }
        void SetVertexShader(byte[] bytecode)
        {
            d3d11VertexShader = DirectX11Helper.d3d11Device.CreateVertexShader(bytecode);
        }
        void SetPixelShader(byte[] bytecode)
        {
            d3d11PixelShader = DirectX11Helper.d3d11Device.CreatePixelShader(bytecode);
        }
        protected override Size MeasureOverride(Size availableSize)
        {
            NullablePtr<IntPtr> handle = new NullablePtr<IntPtr>(IntPtr.Zero);
            d3d9backbuffer = d3d9device.CreateRenderTarget((uint)availableSize.Width, (uint)availableSize.Height, D3DFormat.A8R8G8B8, D3DMultisample.None, 1, 0, handle);
            d3d11Texture2D_RT = DirectX11Helper.d3d11Device.OpenSharedTexture2D(handle.Value);
            d3d11RenderTargetView = DirectX11Helper.d3d11Device.CreateRenderTargetView(d3d11Texture2D_RT, new D3D11RenderTargetViewDesc { Format = DXGIFormat.B8G8R8A8_Unorm, ViewDimension = D3D11RtvDimension.Texture2D, Texture2D = new D3D11Tex2DRtv { MipSlice = 0 } });
            var d3d11Texture2DDesc_DS = new D3D11Texture2DDesc { Width = (uint)availableSize.Width, Height = (uint)availableSize.Height, MipLevels = 1, ArraySize = 1, Format = DXGIFormat.D32_Float, SampleDesc = new DXGISampleDesc { Count = 1, Quality = 0 }, Usage = D3D11Usage.Default, BindFlags = D3D11BindFlag.DepthStencil, CPUAccessFlags = 0 };
            d3d11Texture2D_DS = DirectX11Helper.d3d11Device.CreateTexture2D(d3d11Texture2DDesc_DS, null);
            d3d11DepthStencilView = DirectX11Helper.d3d11Device.CreateDepthStencilView(d3d11Texture2D_DS, new D3D11DepthStencilViewDesc { Format = DXGIFormat.D32_Float, ViewDimension = D3D11DsvDimension.Texture2D, Texture2D = new D3D11Tex2DDsv { MipSlice = 0 } });
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
        D3D11VertexShader d3d11VertexShader;
        D3D11PixelShader d3d11PixelShader;
        D3D11Texture2D d3d11Texture2D_RT;
        D3D11Texture2D d3d11Texture2D_DS;
        D3D11RenderTargetView d3d11RenderTargetView;
        D3D11DepthStencilView d3d11DepthStencilView;
    }
}