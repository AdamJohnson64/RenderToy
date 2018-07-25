////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Diagnostics;
using RenderToy.DirectX;
using RenderToy.DocumentModel;
using RenderToy.Math;
using RenderToy.Shaders;
using System;
using System.Collections.Generic;
using System.Linq;
using System.Threading;
using System.Threading.Tasks;

namespace RenderToy
{
#if OPENVR_INSTALLED
    public static class OpenVRPump
    {
        public static Action CreateRenderer(SparseScene scene)
        {
            var openvr = scene.Select(i => i.NodeTransform).OfType<IVRHost>().Select(i => i.VRHost).Distinct().SingleOrDefault();
            if (openvr == null) return () => { };
            var d3d11VertexShader = DirectX11Helper.d3d11Device.CreateVertexShader(HLSL.D3D11VS);
            var d3d11PixelShader = DirectX11Helper.d3d11Device.CreatePixelShader(HLSL.D3D11PS);
            uint vrwidth = 0, vrheight = 0;
            openvr.System.GetRecommendedRenderTargetSize(ref vrwidth, ref vrheight);

            var d3d11Texture2DDesc_RT_Eye = new D3D11Texture2DDesc { Width = (uint)vrwidth, Height = (uint)vrheight, MipLevels = 1, ArraySize = 1, Format = DXGIFormat.B8G8R8A8_Unorm, SampleDesc = new DXGISampleDesc { Count = 1, Quality = 0 }, Usage = D3D11Usage.Default, BindFlags = D3D11BindFlag.RenderTarget | D3D11BindFlag.ShaderResource, CPUAccessFlags = 0 };
            var d3d11Texture2D_RT_EyeLeft = DirectX11Helper.d3d11Device.CreateTexture2D(d3d11Texture2DDesc_RT_Eye, null);
            var d3d11RenderTargetView_EyeLeft = DirectX11Helper.d3d11Device.CreateRenderTargetView(d3d11Texture2D_RT_EyeLeft, new D3D11RenderTargetViewDesc { Format = DXGIFormat.B8G8R8A8_Unorm, ViewDimension = D3D11RtvDimension.Texture2D, Texture2D = new D3D11Tex2DRtv { MipSlice = 0 } });

            var d3d11Texture2DDesc_DS_EyeLeft = new D3D11Texture2DDesc { Width = (uint)vrwidth, Height = (uint)vrheight, MipLevels = 1, ArraySize = 1, Format = DXGIFormat.D32_Float, SampleDesc = new DXGISampleDesc { Count = 1, Quality = 0 }, Usage = D3D11Usage.Default, BindFlags = D3D11BindFlag.DepthStencil, CPUAccessFlags = 0 };
            var d3d11Texture2D_DS_EyeLeft = DirectX11Helper.d3d11Device.CreateTexture2D(d3d11Texture2DDesc_DS_EyeLeft, null);
            var d3d11DepthStencilView_EyeLeft = DirectX11Helper.d3d11Device.CreateDepthStencilView(d3d11Texture2D_DS_EyeLeft, new D3D11DepthStencilViewDesc { Format = DXGIFormat.D32_Float, ViewDimension = D3D11DsvDimension.Texture2D, Texture2D = new D3D11Tex2DDsv { MipSlice = 0 } });

            var d3d11Texture2D_RT_EyeRight = DirectX11Helper.d3d11Device.CreateTexture2D(d3d11Texture2DDesc_RT_Eye, null);
            var d3d11RenderTargetView_EyeRight = DirectX11Helper.d3d11Device.CreateRenderTargetView(d3d11Texture2D_RT_EyeRight, new D3D11RenderTargetViewDesc { Format = DXGIFormat.B8G8R8A8_Unorm, ViewDimension = D3D11RtvDimension.Texture2D, Texture2D = new D3D11Tex2DRtv { MipSlice = 0 } });

            var d3d11Texture2DDesc_DS_EyeRight = new D3D11Texture2DDesc { Width = (uint)vrwidth, Height = (uint)vrheight, MipLevels = 1, ArraySize = 1, Format = DXGIFormat.D32_Float, SampleDesc = new DXGISampleDesc { Count = 1, Quality = 0 }, Usage = D3D11Usage.Default, BindFlags = D3D11BindFlag.DepthStencil, CPUAccessFlags = 0 };
            var d3d11Texture2D_DS_EyeRight = DirectX11Helper.d3d11Device.CreateTexture2D(d3d11Texture2DDesc_DS_EyeRight, null);
            var d3d11DepthStencilView_EyeRight = DirectX11Helper.d3d11Device.CreateDepthStencilView(d3d11Texture2D_DS_EyeRight, new D3D11DepthStencilViewDesc { Format = DXGIFormat.D32_Float, ViewDimension = D3D11DsvDimension.Texture2D, Texture2D = new D3D11Tex2DDsv { MipSlice = 0 } });

            var Execute_RenderSceneLeft = DirectX11Helper.CreateSceneDraw(scene);
            var Execute_RenderSceneRight = DirectX11Helper.CreateSceneDraw(scene);
            return () =>
            {
                openvr.Update();
                RenderToyEventSource.Default.RenderBegin();
                RenderToyEventSource.Default.MarkerBegin("Update");
                int COUNT_OBJECT = scene.IndexToNodePrimitive.Count;
                for (int i = 0; i < COUNT_OBJECT; ++i)
                {
                    scene.TableTransform[i] = scene.TableNodeTransform[scene.IndexToNodeTransform[i]].Transform;
                }
                RenderToyEventSource.Default.MarkerEnd("Update");
                RenderToyEventSource.Default.MarkerBegin("Prepare All RTs");
                var contextold = DirectX11Helper.d3d11Device.GetImmediateContext();
                var context = contextold.QueryInterfaceD3D11DeviceContext4();
                Task<D3D11CommandList> do_left = Task.Factory.StartNew(() =>
                {
                    var deferred_left_old = DirectX11Helper.d3d11Device.CreateDeferredContext(0);
                    var deferred_left = deferred_left_old.QueryInterfaceD3D11DeviceContext4();
                    deferred_left.VSSetShader(d3d11VertexShader);
                    deferred_left.PSSetShader(d3d11PixelShader);
                    deferred_left.RSSetScissorRects(new[] { new D3D11Rect { left = 0, top = 0, right = (int)vrwidth, bottom = (int)vrheight } });
                    deferred_left.RSSetViewports(new[] { new D3D11Viewport { TopLeftX = 0, TopLeftY = 0, Width = vrwidth, Height = vrheight, MinDepth = 0, MaxDepth = 1 } });
                    deferred_left.OMSetRenderTargets(new[] { d3d11RenderTargetView_EyeLeft }, d3d11DepthStencilView_EyeLeft);
                    deferred_left.ClearDepthStencilView(d3d11DepthStencilView_EyeLeft, D3D11ClearFlag.Depth, 1, 0);
                    deferred_left.ClearRenderTargetView(d3d11RenderTargetView_EyeLeft, 0, 0, 0, 0);
                    Matrix3D transformView = openvr._head * MathHelp.Invert(openvr.GetEyeToHeadTransform(Eye.Left));
                    Matrix3D transformProjection = openvr.GetProjectionMatrix(Eye.Left, 0.1f, 2000.0f);
                    Execute_RenderSceneLeft(deferred_left, MathHelp.Invert(transformView), transformView * transformProjection, "Left Eye");
                    return deferred_left.FinishCommandList(0);
                });
                Task<D3D11CommandList> do_right = Task.Factory.StartNew(() =>
                {
                    var deferred_right_old = DirectX11Helper.d3d11Device.CreateDeferredContext(0);
                    var deferred_right = deferred_right_old.QueryInterfaceD3D11DeviceContext4();
                    deferred_right.VSSetShader(d3d11VertexShader);
                    deferred_right.PSSetShader(d3d11PixelShader);
                    deferred_right.RSSetScissorRects(new[] { new D3D11Rect { left = 0, top = 0, right = (int)vrwidth, bottom = (int)vrheight } });
                    deferred_right.RSSetViewports(new[] { new D3D11Viewport { TopLeftX = 0, TopLeftY = 0, Width = vrwidth, Height = vrheight, MinDepth = 0, MaxDepth = 1 } });
                    deferred_right.OMSetRenderTargets(new[] { d3d11RenderTargetView_EyeRight }, d3d11DepthStencilView_EyeRight);
                    deferred_right.ClearDepthStencilView(d3d11DepthStencilView_EyeRight, D3D11ClearFlag.Depth, 1, 0);
                    deferred_right.ClearRenderTargetView(d3d11RenderTargetView_EyeRight, 0, 0, 0, 0);
                    Matrix3D transformView = openvr._head * MathHelp.Invert(openvr.GetEyeToHeadTransform(Eye.Right));
                    Matrix3D transformProjection = openvr.GetProjectionMatrix(Eye.Right, 0.1f, 2000.0f);
                    Execute_RenderSceneRight(deferred_right, MathHelp.Invert(transformView), transformView * transformProjection, "Right Eye");
                    return deferred_right.FinishCommandList(0);
                });
                RenderToyEventSource.Default.MarkerEnd("Prepare All RTs");
                do_left.Wait();
                do_right.Wait();
                RenderToyEventSource.Default.MarkerBegin("Execute All RTs");
                context.ExecuteCommandList(do_left.Result, 1);
                context.ExecuteCommandList(do_right.Result, 1);
                RenderToyEventSource.Default.MarkerEnd("Execute All RTs");
                RenderToyEventSource.Default.MarkerBegin("Submit To OpenVR");
                openvr.Compositor.Submit(Eye.Left, d3d11Texture2D_RT_EyeLeft.ManagedPtr);
                openvr.Compositor.Submit(Eye.Right, d3d11Texture2D_RT_EyeRight.ManagedPtr);
                RenderToyEventSource.Default.MarkerEnd("Submit To OpenVR");
                RenderToyEventSource.Default.RenderEnd();
            };
        }
        public static void CreateThread(SparseScene scene)
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