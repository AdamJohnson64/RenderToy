using RenderToy.Cameras;
using RenderToy.DirectX;
using RenderToy.Materials;
using RenderToy.Math;
using RenderToy.Meshes;
using RenderToy.ModelFormat;
using RenderToy.SceneGraph;
using RenderToy.Shaders;
using System;
using System.Collections.Generic;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Forms;
using System.Windows.Interop;
using System.Windows.Media;
#if OPENVR_INSTALLED
using System.Windows.Threading;
#endif // OPENVR_INSTALLED

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
                ((ViewDirectX11)s).RenderDX();
            }));
            AttachedView.TransformModelViewProjectionProperty.OverrideMetadata(typeof(ViewDirectX11), new FrameworkPropertyMetadata(Matrix3D.Identity, (s, e) =>
            {
                ((ViewDirectX11)s).RenderDX();
            }));
        }
        public ViewDirectX11()
        {
            d3d11constantbufferCPU = new byte[256 * 1024];
            d3d11constantbufferGPU = DirectX11Helper.d3d11Device.CreateBuffer(new D3D11BufferDesc { ByteWidth = (uint)d3d11constantbufferCPU.Length, Usage = D3D11Usage.Default, BindFlags = D3D11BindFlag.ConstantBuffer, CPUAccessFlags = 0, MiscFlags = 0, StructureByteStride = 4 * 16 }, null);
#if OPENVR_INSTALLED
            uint vrwidth = 0, vrheight = 0;
            OpenVR.GetRecommendedRenderTargetSize(ref vrwidth, ref vrheight);
            var d3d11Texture2DDesc_DS_Eye = new D3D11Texture2DDesc { Width = (uint)vrwidth, Height = (uint)vrheight, MipLevels = 1, ArraySize = 1, Format = DXGIFormat.D32_Float, SampleDesc = new DXGISampleDesc { Count = 1, Quality = 0 }, Usage = D3D11Usage.Default, BindFlags = D3D11BindFlag.DepthStencil, CPUAccessFlags = 0 };
            d3d11Texture2D_DS_Eye = DirectX11Helper.d3d11Device.CreateTexture2D(d3d11Texture2DDesc_DS_Eye, null);
            d3d11DepthStencilView_Eye = DirectX11Helper.d3d11Device.CreateDepthStencilView(d3d11Texture2D_DS_Eye, new D3D11DepthStencilViewDesc { Format = DXGIFormat.D32_Float, ViewDimension = D3D11DsvDimension.Texture2D, Texture2D = new D3D11Tex2DDsv { MipSlice = 0 } });
            var d3d11Texture2DDesc_RT_Eye = new D3D11Texture2DDesc { Width = (uint)vrwidth, Height = (uint)vrheight, MipLevels = 1, ArraySize = 1, Format = DXGIFormat.B8G8R8A8_Unorm, SampleDesc = new DXGISampleDesc { Count = 1, Quality = 0 }, Usage = D3D11Usage.Default, BindFlags = D3D11BindFlag.RenderTarget | D3D11BindFlag.ShaderResource, CPUAccessFlags = 0 };
            d3d11Texture2D_RT_EyeLeft = DirectX11Helper.d3d11Device.CreateTexture2D(d3d11Texture2DDesc_RT_Eye, null);
            d3d11RenderTargetView_EyeLeft = DirectX11Helper.d3d11Device.CreateRenderTargetView(d3d11Texture2D_RT_EyeLeft, new D3D11RenderTargetViewDesc { Format = DXGIFormat.B8G8R8A8_Unorm, ViewDimension = D3D11RtvDimension.Texture2D, Texture2D = new D3D11Tex2DRtv { MipSlice = 0 } });
            d3d11Texture2D_RT_EyeRight = DirectX11Helper.d3d11Device.CreateTexture2D(d3d11Texture2DDesc_RT_Eye, null);
            d3d11RenderTargetView_EyeRight = DirectX11Helper.d3d11Device.CreateRenderTargetView(d3d11Texture2D_RT_EyeRight, new D3D11RenderTargetViewDesc { Format = DXGIFormat.B8G8R8A8_Unorm, ViewDimension = D3D11RtvDimension.Texture2D, Texture2D = new D3D11Tex2DRtv { MipSlice = 0 } });
            timer = new DispatcherTimer(TimeSpan.FromMilliseconds(10), DispatcherPriority.Normal, (s, e) =>
            {
                RenderDX();
            }, Dispatcher.CurrentDispatcher);
            timer.Start();
#endif // OPENVR_INSTALLED
        }
#if OPENVR_INSTALLED
        DispatcherTimer timer = new DispatcherTimer();
#endif // OPENVR_INSTALLED
        IScene Execute_SceneLast = null;
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
            context.IASetPrimitiveTopology(D3DPrimitiveTopology.TriangleList);
            context.IASetInputLayout(DirectX11Helper.d3d11InputLayout);
            context.VSSetShader(d3d11VertexShader);
            context.PSSetShader(d3d11PixelShader);
            context.RSSetState(DirectX11Helper.d3d11RasterizerState);
            context.PSSetSamplers(0, new[] { DirectX11Helper.d3d11SamplerState });
            ////////////////////////////////////////////////////////////////////////////////
            // Assemble the scene parts.
            if (Execute_SceneLast != AttachedView.GetScene(this))
            {
                Execute_SceneLast = AttachedView.GetScene(this);
                List<Action<Matrix3D>> execute_retransform = new List<Action<Matrix3D>>();
                List<Action<D3D11DeviceContext4>> execute_drawprimitive = new List<Action<D3D11DeviceContext4>>();
                // We're collecting constant buffers because DX11 hates to do actual work.
                int constantbufferoffset = 0;
                {
                    var constantbufferlist = new[] { d3d11constantbufferGPU };
                    foreach (var transformedobject in TransformedObject.Enumerate(AttachedView.GetScene(this)))
                    {
                        var vertexbuffer = DirectX11Helper.CreateVertexBuffer(transformedobject.Node.Primitive);
                        if (vertexbuffer == null) continue;
                        var thisconstantbufferoffset = constantbufferoffset;
                        execute_retransform.Add((transformViewProjection) =>
                        {
                            var transformModelViewProjection = transformedobject.Transform * transformViewProjection;
                            Buffer.BlockCopy(DirectXHelper.ConvertToD3DMatrix(transformModelViewProjection), 0, d3d11constantbufferCPU, thisconstantbufferoffset, 4 * 16);
                        });
                        var objmat = transformedobject.Node.Material as LoaderOBJ.OBJMaterial;
                        var collecttextures = new[]
                        {
                            DirectX11Helper.CreateTextureView(objmat == null ? transformedobject.Node.Material : objmat.map_Kd, StockMaterials.PlasticWhite),
                            DirectX11Helper.CreateTextureView(objmat == null ? null : objmat.map_d, StockMaterials.PlasticWhite),
                            DirectX11Helper.CreateTextureView(objmat == null ? null : objmat.map_bump, StockMaterials.PlasticLightBlue),
                            DirectX11Helper.CreateTextureView(objmat == null ? null : objmat.displacement, StockMaterials.PlasticWhite)
                        };
                        execute_drawprimitive.Add((context2) =>
                        {
                            context2.VSSetConstantBuffers1(0, constantbufferlist, new[] { (uint)thisconstantbufferoffset / 16U }, new[] { 4U * 16U });
                            context2.IASetVertexBuffers(0, new[] { vertexbuffer.d3d11Buffer }, new[] { (uint)Marshal.SizeOf(typeof(XYZNorDiffuseTex1)) }, new[] { 0U });
                            context2.PSSetShaderResources(0, collecttextures);
                            context2.Draw(vertexbuffer.vertexCount, 0);
                        });
                        // Pad up to 256 bytes.
                        constantbufferoffset += 4 * 16;
                        if ((constantbufferoffset & 0xFF) != 0)
                        {
                            constantbufferoffset = constantbufferoffset & (~0xFF);
                            constantbufferoffset += 256;
                        }
                    }
                }
                Execute_DrawScene = (context2, transformViewProjection) =>
                {
                    foreach (var retransform in execute_retransform)
                    {
                        retransform(transformViewProjection);
                    }
                    context.UpdateSubresource1(d3d11constantbufferGPU, 0, new D3D11Box { right = (uint)constantbufferoffset }, d3d11constantbufferCPU, 0, 0, D3D11CopyFlags.Discard);
                    foreach (var draw in execute_drawprimitive)
                    {
                        draw(context2);
                    }
                };
            }
            // Draw the window view using the current camera.
            context.RSSetScissorRects(new[] { new D3D11Rect { left = 0, top = 0, right = width, bottom = height } });
            context.RSSetViewports(new[] { new D3D11Viewport { TopLeftX = 0, TopLeftY = 0, Width = width, Height = height, MinDepth = 0, MaxDepth = 1 } });
            context.OMSetRenderTargets(new[] { d3d11RenderTargetView }, d3d11DepthStencilView);
            context.ClearDepthStencilView(d3d11DepthStencilView, D3D11ClearFlag.Depth, 1, 0);
            context.ClearRenderTargetView(d3d11RenderTargetView, 0, 0, 0, 0);
#if !OPENVR_DRIVE_UI_VIEW
            Execute_DrawScene(context, MathHelp.Invert(OpenVRHelper.LocateDeviceId(0)) * OpenVRHelper.GetProjectionMatrix(Eye.Left, 0.1f, 2000.0f) * Perspective.AspectCorrectFit(ActualWidth, ActualHeight));
#else
            Execute_DrawScene(context, AttachedView.GetTransformModelViewProjection(this) * Perspective.AspectCorrectFit(ActualWidth, ActualHeight));
#endif
            context.Flush();
            d3dimage.Lock();
            d3dimage.AddDirtyRect(new Int32Rect(0, 0, d3d11Texture2D_RT.GetWidth(), d3d11Texture2D_RT.GetHeight()));
            d3dimage.Unlock();
#if OPENVR_INSTALLED
            OpenVRCompositor.WaitGetPoses();
            {
                uint vrwidth = 0, vrheight = 0;
                OpenVR.GetRecommendedRenderTargetSize(ref vrwidth, ref vrheight);
                context.RSSetScissorRects(new[] { new D3D11Rect { left = 0, top = 0, right = (int)vrwidth, bottom = (int)vrheight } });
                context.RSSetViewports(new[] { new D3D11Viewport { TopLeftX = 0, TopLeftY = 0, Width = vrwidth, Height = vrheight, MinDepth = 0, MaxDepth = 1 } });
            }
            Matrix3D transformHead = MathHelp.Invert(OpenVRHelper.LocateDeviceId(0));
            {
                context.OMSetRenderTargets(new[] { d3d11RenderTargetView_EyeLeft }, d3d11DepthStencilView_Eye);
                context.ClearDepthStencilView(d3d11DepthStencilView_Eye, D3D11ClearFlag.Depth, 1, 0);
                context.ClearRenderTargetView(d3d11RenderTargetView_EyeLeft, 0, 0, 0, 0);
                Execute_DrawScene(context, transformHead * MathHelp.Invert(OpenVRHelper.GetEyeToHeadTransform(Eye.Left)) * OpenVRHelper.GetProjectionMatrix(Eye.Left, 0.1f, 2000.0f));
                OpenVRCompositor.Submit(Eye.Left, d3d11Texture2D_RT_EyeLeft.ManagedPtr);
            }
            {
                context.OMSetRenderTargets(new[] { d3d11RenderTargetView_EyeRight }, d3d11DepthStencilView_Eye);
                context.ClearDepthStencilView(d3d11DepthStencilView_Eye, D3D11ClearFlag.Depth, 1, 0);
                context.ClearRenderTargetView(d3d11RenderTargetView_EyeRight, 0, 0, 0, 0);
                Execute_DrawScene(context, transformHead * MathHelp.Invert(OpenVRHelper.GetEyeToHeadTransform(Eye.Right)) * OpenVRHelper.GetProjectionMatrix(Eye.Right, 0.1f, 2000.0f));
                OpenVRCompositor.Submit(Eye.Right, d3d11Texture2D_RT_EyeRight.ManagedPtr);
            }
#endif // OPENVR_INSTALLED
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
        byte[] d3d11constantbufferCPU;
        D3D11Buffer d3d11constantbufferGPU;
        D3D11VertexShader d3d11VertexShader;
        D3D11PixelShader d3d11PixelShader;
        D3D11Texture2D d3d11Texture2D_RT;
        D3D11Texture2D d3d11Texture2D_DS;
        D3D11RenderTargetView d3d11RenderTargetView;
        D3D11DepthStencilView d3d11DepthStencilView;
#if OPENVR_INSTALLED
        D3D11Texture2D d3d11Texture2D_DS_Eye;
        D3D11DepthStencilView d3d11DepthStencilView_Eye;
        D3D11Texture2D d3d11Texture2D_RT_EyeLeft;
        D3D11RenderTargetView d3d11RenderTargetView_EyeLeft;
        D3D11Texture2D d3d11Texture2D_RT_EyeRight;
        D3D11RenderTargetView d3d11RenderTargetView_EyeRight;
#endif // OPENVR_INSTALLED
    }
}