////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.DirectX;
using RenderToy.Materials;
using RenderToy.Math;
using RenderToy.Primitives;
using RenderToy.SceneGraph;
using RenderToy.Shaders;
using RenderToy.Textures;
using RenderToy.Transforms;
using System;
using System.Linq;
using System.Threading;

namespace RenderToy
{
#if OPENVR_INSTALLED
    public static class OpenVRPump
    {
        public static Action CreateRenderer(IScene scene)
        {
            var openvr = TransformedObject.Enumerate(scene).Select(i => i.Node.Transform).OfType<IVRHost>().Select(i => i.VRHost).Distinct().SingleOrDefault();
            if (openvr == null) return () => { };
            var d3d11VertexShader = DirectX11Helper.d3d11Device.CreateVertexShader(HLSL.D3D11VS);
            var d3d11PixelShader = DirectX11Helper.d3d11Device.CreatePixelShader(HLSL.D3D11PS);
            var Execute_DrawScene = DirectX11Helper.CreateSceneDraw(scene);
            uint vrwidth = 0, vrheight = 0;
            openvr.System.GetRecommendedRenderTargetSize(ref vrwidth, ref vrheight);
            var d3d11Texture2DDesc_DS_Eye = new D3D11Texture2DDesc { Width = (uint)vrwidth, Height = (uint)vrheight, MipLevels = 1, ArraySize = 1, Format = DXGIFormat.D32_Float, SampleDesc = new DXGISampleDesc { Count = 1, Quality = 0 }, Usage = D3D11Usage.Default, BindFlags = D3D11BindFlag.DepthStencil, CPUAccessFlags = 0 };
            var d3d11Texture2D_DS_Eye = DirectX11Helper.d3d11Device.CreateTexture2D(d3d11Texture2DDesc_DS_Eye, null);
            var d3d11DepthStencilView_Eye = DirectX11Helper.d3d11Device.CreateDepthStencilView(d3d11Texture2D_DS_Eye, new D3D11DepthStencilViewDesc { Format = DXGIFormat.D32_Float, ViewDimension = D3D11DsvDimension.Texture2D, Texture2D = new D3D11Tex2DDsv { MipSlice = 0 } });
            var d3d11Texture2DDesc_RT_Eye = new D3D11Texture2DDesc { Width = (uint)vrwidth, Height = (uint)vrheight, MipLevels = 1, ArraySize = 1, Format = DXGIFormat.B8G8R8A8_Unorm, SampleDesc = new DXGISampleDesc { Count = 1, Quality = 0 }, Usage = D3D11Usage.Default, BindFlags = D3D11BindFlag.RenderTarget | D3D11BindFlag.ShaderResource, CPUAccessFlags = 0 };
            var d3d11Texture2D_RT_EyeLeft = DirectX11Helper.d3d11Device.CreateTexture2D(d3d11Texture2DDesc_RT_Eye, null);
            var d3d11RenderTargetView_EyeLeft = DirectX11Helper.d3d11Device.CreateRenderTargetView(d3d11Texture2D_RT_EyeLeft, new D3D11RenderTargetViewDesc { Format = DXGIFormat.B8G8R8A8_Unorm, ViewDimension = D3D11RtvDimension.Texture2D, Texture2D = new D3D11Tex2DRtv { MipSlice = 0 } });
            var d3d11Texture2D_RT_EyeRight = DirectX11Helper.d3d11Device.CreateTexture2D(d3d11Texture2DDesc_RT_Eye, null);
            var d3d11RenderTargetView_EyeRight = DirectX11Helper.d3d11Device.CreateRenderTargetView(d3d11Texture2D_RT_EyeRight, new D3D11RenderTargetViewDesc { Format = DXGIFormat.B8G8R8A8_Unorm, ViewDimension = D3D11RtvDimension.Texture2D, Texture2D = new D3D11Tex2DRtv { MipSlice = 0 } });
            return () =>
            {
                var contextold = DirectX11Helper.d3d11Device.GetImmediateContext();
                var context = contextold.QueryInterfaceD3D11DeviceContext4();
                context.VSSetShader(d3d11VertexShader);
                context.PSSetShader(d3d11PixelShader);
                context.RSSetScissorRects(new[] { new D3D11Rect { left = 0, top = 0, right = (int)vrwidth, bottom = (int)vrheight } });
                context.RSSetViewports(new[] { new D3D11Viewport { TopLeftX = 0, TopLeftY = 0, Width = vrwidth, Height = vrheight, MinDepth = 0, MaxDepth = 1 } });
                openvr.Update();
                {
                    context.OMSetRenderTargets(new[] { d3d11RenderTargetView_EyeLeft }, d3d11DepthStencilView_Eye);
                    context.ClearDepthStencilView(d3d11DepthStencilView_Eye, D3D11ClearFlag.Depth, 1, 0);
                    context.ClearRenderTargetView(d3d11RenderTargetView_EyeLeft, 0, 0, 0, 0);
                    Execute_DrawScene(context, openvr._head * MathHelp.Invert(openvr.GetEyeToHeadTransform(Eye.Left)) * openvr.GetProjectionMatrix(Eye.Left, 0.1f, 2000.0f));
                    openvr.Compositor.Submit(Eye.Left, d3d11Texture2D_RT_EyeLeft.ManagedPtr);
                }
                {
                    context.OMSetRenderTargets(new[] { d3d11RenderTargetView_EyeRight }, d3d11DepthStencilView_Eye);
                    context.ClearDepthStencilView(d3d11DepthStencilView_Eye, D3D11ClearFlag.Depth, 1, 0);
                    context.ClearRenderTargetView(d3d11RenderTargetView_EyeRight, 0, 0, 0, 0);
                    Execute_DrawScene(context, openvr._head * MathHelp.Invert(openvr.GetEyeToHeadTransform(Eye.Right)) * openvr.GetProjectionMatrix(Eye.Right, 0.1f, 2000.0f));
                    openvr.Compositor.Submit(Eye.Right, d3d11Texture2D_RT_EyeRight.ManagedPtr);
                }
            };
        }
        public static void CreateThread(IScene scene)
        {
            var renderer = CreateRenderer(scene);
            var thread = new Thread((param) =>
            {
                while (true)
                {
                    renderer();
                }
            });
            thread.Start();
        }
    }
#endif // OPENVR_INSTALLED
}