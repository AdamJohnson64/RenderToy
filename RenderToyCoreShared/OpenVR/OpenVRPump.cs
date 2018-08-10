////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToyCOM;
using RenderToy.Diagnostics;
using RenderToy.DirectX;
using RenderToy.DocumentModel;
using RenderToy.Math;
using RenderToy.RenderMode;
using RenderToy.SceneGraph;
using RenderToy.Shaders;
using RenderToy.Utility;
using System;
using System.Collections.Generic;
using System.Threading.Tasks;
using System.Runtime.InteropServices;
using Valve.VR;

namespace RenderToy
{
#if OPENVR_INSTALLED
    public static class OpenVRPump
    {
        public static Action CreateRenderer(SparseScene scene)
        {
            uint vrwidth = 0, vrheight = 0;
            OpenVRHelper.System.GetRecommendedRenderTargetSize(ref vrwidth, ref vrheight);
            ID3D11VertexShader d3d11VertexShader = null;
            ID3D11PixelShader d3d11PixelShader = null;
            ID3D11Texture2D d3d11Texture2D_RT_EyeLeft = null;
            ID3D11RenderTargetView d3d11RenderTargetView_EyeLeft = null;
            ID3D11Texture2D d3d11Texture2D_DS_EyeLeft = null;
            ID3D11DepthStencilView d3d11DepthStencilView_EyeLeft = null;
            ID3D11Texture2D d3d11Texture2D_RT_EyeRight = null;
            ID3D11RenderTargetView d3d11RenderTargetView_EyeRight = null;
            ID3D11Texture2D d3d11Texture2D_DS_EyeRight = null;
            ID3D11DepthStencilView d3d11DepthStencilView_EyeRight = null;
            {
                ID3D11ClassLinkage linkage = null;
                Direct3D11Helper.d3d11Device.CreateVertexShader(UnmanagedCopy.Create(HLSL.D3D11VS), (ulong)HLSL.D3D11VS.Length, linkage, ref d3d11VertexShader);
                Direct3D11Helper.d3d11Device.CreatePixelShader(UnmanagedCopy.Create(HLSL.D3D11PS), (ulong)HLSL.D3D11PS.Length, linkage, ref d3d11PixelShader);
                var d3d11Texture2DDesc_RT_Eye = new D3D11_TEXTURE2D_DESC { Width = (uint)vrwidth, Height = (uint)vrheight, MipLevels = 1, ArraySize = 1, Format = DXGI_FORMAT.DXGI_FORMAT_B8G8R8A8_UNORM, SampleDesc = new DXGI_SAMPLE_DESC { Count = 1, Quality = 0 }, Usage = D3D11_USAGE.D3D11_USAGE_DEFAULT, BindFlags = (uint)D3D11_BIND_FLAG.D3D11_BIND_RENDER_TARGET | (uint)D3D11_BIND_FLAG.D3D11_BIND_SHADER_RESOURCE, CPUAccessFlags = 0 };
                var d3d11Texture2DDesc_DS_Eye = new D3D11_TEXTURE2D_DESC { Width = (uint)vrwidth, Height = (uint)vrheight, MipLevels = 1, ArraySize = 1, Format = DXGI_FORMAT.DXGI_FORMAT_D32_FLOAT, SampleDesc = new DXGI_SAMPLE_DESC { Count = 1, Quality = 0 }, Usage = D3D11_USAGE.D3D11_USAGE_DEFAULT, BindFlags = (uint)D3D11_BIND_FLAG.D3D11_BIND_DEPTH_STENCIL, CPUAccessFlags = 0 };
                var d3d11RenderTargetView_RT_Eye = new D3D11_RENDER_TARGET_VIEW_DESC { Format = DXGI_FORMAT.DXGI_FORMAT_B8G8R8A8_UNORM, ViewDimension = D3D11_RTV_DIMENSION.D3D11_RTV_DIMENSION_TEXTURE2D };
                d3d11RenderTargetView_RT_Eye.__MIDL____MIDL_itf_RenderToy_0005_00650002.Texture2D = new D3D11_TEX2D_RTV { MipSlice = 0 };
                var d3d11DepthStencilView_DS_Eye = new D3D11_DEPTH_STENCIL_VIEW_DESC { Format = DXGI_FORMAT.DXGI_FORMAT_D32_FLOAT, ViewDimension = D3D11_DSV_DIMENSION.D3D11_DSV_DIMENSION_TEXTURE2D };
                d3d11DepthStencilView_DS_Eye.__MIDL____MIDL_itf_RenderToy_0005_00660000.Texture2D = new D3D11_TEX2D_DSV { MipSlice = 0 };
                unsafe
                {
                    D3D11_SUBRESOURCE_DATA *pInitialData = null;
                    Direct3D11Helper.d3d11Device.CreateTexture2D(d3d11Texture2DDesc_RT_Eye, ref *pInitialData, ref d3d11Texture2D_RT_EyeLeft);
                    Direct3D11Helper.d3d11Device.CreateTexture2D(d3d11Texture2DDesc_DS_Eye, ref *pInitialData, ref d3d11Texture2D_DS_EyeLeft);
                    Direct3D11Helper.d3d11Device.CreateTexture2D(d3d11Texture2DDesc_RT_Eye, ref *pInitialData, ref d3d11Texture2D_RT_EyeRight);
                    Direct3D11Helper.d3d11Device.CreateTexture2D(d3d11Texture2DDesc_DS_Eye, ref *pInitialData, ref d3d11Texture2D_DS_EyeRight);
                }
                Direct3D11Helper.d3d11Device.CreateRenderTargetView(d3d11Texture2D_RT_EyeLeft, d3d11RenderTargetView_RT_Eye, ref d3d11RenderTargetView_EyeLeft);
                Direct3D11Helper.d3d11Device.CreateRenderTargetView(d3d11Texture2D_RT_EyeRight, d3d11RenderTargetView_RT_Eye, ref d3d11RenderTargetView_EyeRight);
                Direct3D11Helper.d3d11Device.CreateDepthStencilView(d3d11Texture2D_DS_EyeLeft, d3d11DepthStencilView_DS_Eye, ref d3d11DepthStencilView_EyeLeft);
                Direct3D11Helper.d3d11Device.CreateDepthStencilView(d3d11Texture2D_DS_EyeRight, d3d11DepthStencilView_DS_Eye, ref d3d11DepthStencilView_EyeRight);
            }
            var Execute_RenderSceneLeft = Direct3D11Helper.CreateSceneDraw(scene);
            var Execute_RenderSceneRight = Direct3D11Helper.CreateSceneDraw(scene);
            var sceneData = SceneSerializer.CreateFlatMemoryF32(TransformedObject.Enumerate(TestScenes.DefaultScene));
            return () =>
            {
                OpenVRHelper.Update();
                RenderToyEventSource.Default.RenderBegin();
                RenderToyEventSource.Default.MarkerBegin("Update");
                int COUNT_OBJECT = scene.IndexToNodePrimitive.Count;
                for (int i = 0; i < COUNT_OBJECT; ++i)
                {
                    scene.TableTransform[i] = scene.TableNodeTransform[scene.IndexToNodeTransform[i]].Transform;
                }
                RenderToyEventSource.Default.MarkerEnd("Update");
                RenderToyEventSource.Default.MarkerBegin("Build & Execute Command Buffers");
                ID3D11DeviceContext context_old = null;
                Direct3D11Helper.d3d11Device.GetImmediateContext(ref context_old);
                var context = (ID3D11DeviceContext4)context_old;
                Task<ID3D11CommandList> do_left = Task.Factory.StartNew(() =>
                {
                    ID3D11DeviceContext deferred_left_old = null;
                    Direct3D11Helper.d3d11Device.CreateDeferredContext(0, ref deferred_left_old);
                    ID3D11DeviceContext4 deferred_left = (ID3D11DeviceContext4)deferred_left_old;
                    ID3D11ClassInstance classInstance = null;
                    deferred_left.VSSetShader(d3d11VertexShader, classInstance, 0);
                    deferred_left.PSSetShader(d3d11PixelShader, classInstance, 0);
                    var scissorRect = new tagRECT { left = 0, top = 0, right = (int)vrwidth, bottom = (int)vrheight };
                    deferred_left.RSSetScissorRects(1, scissorRect);
                    var viewportRect = new D3D11_VIEWPORT { TopLeftX = 0, TopLeftY = 0, Width = vrwidth, Height = vrheight, MinDepth = 0, MaxDepth = 1 };
                    deferred_left.RSSetViewports(1, viewportRect);
                    deferred_left.OMSetRenderTargets(1, d3d11RenderTargetView_EyeLeft, d3d11DepthStencilView_EyeLeft);
                    deferred_left.ClearDepthStencilView(d3d11DepthStencilView_EyeLeft, (uint)D3D11_CLEAR_FLAG.D3D11_CLEAR_DEPTH, 1, 0);
                    deferred_left.ClearRenderTargetView(d3d11RenderTargetView_EyeLeft, new float[] { 0, 0, 0, 0 });
                    var transformView = OpenVRHelper._head * MathHelp.Invert(OpenVRHelper.GetEyeToHeadTransform(EVREye.Eye_Left));
                    var transformCamera = MathHelp.Invert(transformView);
                    var constants = new Dictionary<string, object>();
                    constants["profilingName"] = "Left Eye";
                    constants["transformAspect"] = Matrix3D.Identity;
                    constants["transformCamera"] = transformCamera;
                    constants["transformView"] = transformView;
                    constants["transformProjection"] = OpenVRHelper.GetProjectionMatrix(EVREye.Eye_Left, 0.1f, 2000.0f);
                    Execute_RenderSceneLeft(deferred_left, constants);
                    ID3D11CommandList commandList = null;
                    deferred_left.FinishCommandList(0, ref commandList);
                    return commandList;
                });
                Task<ID3D11CommandList> do_right = Task.Factory.StartNew(() =>
                {
                    ID3D11DeviceContext deferred_right_old = null;
                    Direct3D11Helper.d3d11Device.CreateDeferredContext(0, ref deferred_right_old);
                    ID3D11DeviceContext4 deferred_right = (ID3D11DeviceContext4)deferred_right_old;
                    ID3D11ClassInstance classInstance = null;
                    deferred_right.VSSetShader(d3d11VertexShader, ref classInstance, 0);
                    deferred_right.PSSetShader(d3d11PixelShader, ref classInstance, 0);
                    var scissorRect = new tagRECT { left = 0, top = 0, right = (int)vrwidth, bottom = (int)vrheight };
                    deferred_right.RSSetScissorRects(1, ref scissorRect);
                    var viewportRect = new D3D11_VIEWPORT { TopLeftX = 0, TopLeftY = 0, Width = vrwidth, Height = vrheight, MinDepth = 0, MaxDepth = 1 };
                    deferred_right.RSSetViewports(1, ref viewportRect);
                    deferred_right.OMSetRenderTargets(1, ref d3d11RenderTargetView_EyeRight, d3d11DepthStencilView_EyeRight);
                    deferred_right.ClearDepthStencilView(d3d11DepthStencilView_EyeRight, (uint)D3D11_CLEAR_FLAG.D3D11_CLEAR_DEPTH, 1, 0);
                    deferred_right.ClearRenderTargetView(d3d11RenderTargetView_EyeRight, new float[] { 0, 0, 0, 0 });
                    var transformView = OpenVRHelper._head * MathHelp.Invert(OpenVRHelper.GetEyeToHeadTransform(EVREye.Eye_Right));
                    var transformCamera = MathHelp.Invert(transformView);
                    var constants = new Dictionary<string, object>();
                    constants["profilingName"] = "Right Eye";
                    constants["transformAspect"] = Matrix3D.Identity;
                    constants["transformCamera"] = transformCamera;
                    constants["transformView"] = transformView;
                    constants["transformProjection"] = OpenVRHelper.GetProjectionMatrix(EVREye.Eye_Right, 0.1f, 2000.0f);
                    Execute_RenderSceneRight(deferred_right, constants);
                    ID3D11CommandList commandList = null;
                    deferred_right.FinishCommandList(0, ref commandList);
                    return commandList;
                });
                do_left.Wait();
                do_right.Wait();
                RenderToyEventSource.Default.MarkerEnd("Build & Execute Command Buffers");
                RenderToyEventSource.Default.MarkerBegin("Execute All RTs");
                context.ExecuteCommandList(do_left.Result, 1);
                context.ExecuteCommandList(do_right.Result, 1);
                RenderToyEventSource.Default.MarkerEnd("Execute All RTs");
                RenderToyEventSource.Default.MarkerBegin("Submit To OpenVR");
                unsafe
                {
                    VRTextureBounds_t *pBounds = null;
                    var textureLeft = new Texture_t { handle = Marshal.GetComInterfaceForObject<ID3D11Texture2D, ID3D11Texture2D>(d3d11Texture2D_RT_EyeLeft), eType = ETextureType.DirectX, eColorSpace = EColorSpace.Auto };
                    OpenVRHelper.Compositor.Submit(EVREye.Eye_Left, ref textureLeft, ref *pBounds, EVRSubmitFlags.Submit_Default);
                    var textureRight = new Texture_t { handle = Marshal.GetComInterfaceForObject<ID3D11Texture2D, ID3D11Texture2D>(d3d11Texture2D_RT_EyeRight), eType = ETextureType.DirectX, eColorSpace = EColorSpace.Auto };
                    OpenVRHelper.Compositor.Submit(EVREye.Eye_Right, ref textureRight, ref *pBounds, EVRSubmitFlags.Submit_Default);
                }
                RenderToyEventSource.Default.MarkerEnd("Submit To OpenVR");
                RenderToyEventSource.Default.RenderEnd();
            };
        }
        public static Action CreateRendererRaytraced(SparseScene scene)
        {
            uint vrwidth = 0, vrheight = 0;
            OpenVRHelper.System.GetRecommendedRenderTargetSize(ref vrwidth, ref vrheight);
            vrwidth /= 2;
            vrheight /= 2;
            ID3D11VertexShader d3d11VertexShader = null;
            ID3D11PixelShader d3d11PixelShader = null;
            ID3D11Texture2D d3d11Texture2D_RT_EyeLeft = null;
            ID3D11RenderTargetView d3d11RenderTargetView_EyeLeft = null;
            ID3D11Texture2D d3d11Texture2D_RT_EyeRight = null;
            ID3D11RenderTargetView d3d11RenderTargetView_EyeRight = null;
            {
                ID3D11ClassLinkage linkage = null;
                Direct3D11Helper.d3d11Device.CreateVertexShader(UnmanagedCopy.Create(HLSL.D3D11VS), (ulong)HLSL.D3D11VS.Length, linkage, ref d3d11VertexShader);
                Direct3D11Helper.d3d11Device.CreatePixelShader(UnmanagedCopy.Create(HLSL.D3D11PS), (ulong)HLSL.D3D11PS.Length, linkage, ref d3d11PixelShader);
                var d3d11Texture2DDesc_RT_Eye = new D3D11_TEXTURE2D_DESC { Width = (uint)vrwidth, Height = (uint)vrheight, MipLevels = 1, ArraySize = 1, Format = DXGI_FORMAT.DXGI_FORMAT_B8G8R8A8_UNORM, SampleDesc = new DXGI_SAMPLE_DESC { Count = 1, Quality = 0 }, Usage = D3D11_USAGE.D3D11_USAGE_DEFAULT, BindFlags = (uint)D3D11_BIND_FLAG.D3D11_BIND_RENDER_TARGET | (uint)D3D11_BIND_FLAG.D3D11_BIND_UNORDERED_ACCESS, CPUAccessFlags = 0 };
                var d3d11RenderTargetView_RT_Eye = new D3D11_RENDER_TARGET_VIEW_DESC { Format = DXGI_FORMAT.DXGI_FORMAT_B8G8R8A8_UNORM, ViewDimension = D3D11_RTV_DIMENSION.D3D11_RTV_DIMENSION_TEXTURE2D };
                d3d11RenderTargetView_RT_Eye.__MIDL____MIDL_itf_RenderToy_0005_00650002.Texture2D = new D3D11_TEX2D_RTV { MipSlice = 0 };
                var d3d11DepthStencilView_DS_Eye = new D3D11_DEPTH_STENCIL_VIEW_DESC { Format = DXGI_FORMAT.DXGI_FORMAT_D32_FLOAT, ViewDimension = D3D11_DSV_DIMENSION.D3D11_DSV_DIMENSION_TEXTURE2D };
                d3d11DepthStencilView_DS_Eye.__MIDL____MIDL_itf_RenderToy_0005_00660000.Texture2D = new D3D11_TEX2D_DSV { MipSlice = 0 };
                unsafe
                {
                    D3D11_SUBRESOURCE_DATA* pInitialData = null;
                    Direct3D11Helper.d3d11Device.CreateTexture2D(d3d11Texture2DDesc_RT_Eye, ref *pInitialData, ref d3d11Texture2D_RT_EyeLeft);
                    Direct3D11Helper.d3d11Device.CreateTexture2D(d3d11Texture2DDesc_RT_Eye, ref *pInitialData, ref d3d11Texture2D_RT_EyeRight);
                }
                Direct3D11Helper.d3d11Device.CreateRenderTargetView(d3d11Texture2D_RT_EyeLeft, d3d11RenderTargetView_RT_Eye, ref d3d11RenderTargetView_EyeLeft);
                Direct3D11Helper.d3d11Device.CreateRenderTargetView(d3d11Texture2D_RT_EyeRight, d3d11RenderTargetView_RT_Eye, ref d3d11RenderTargetView_EyeRight);
            }
            var Execute_RenderSceneLeft = Direct3D11Helper.CreateSceneDraw(scene);
            var Execute_RenderSceneRight = Direct3D11Helper.CreateSceneDraw(scene);
            var sceneData = SceneSerializer.CreateFlatMemoryF32(TransformedObject.Enumerate(TestScenes.DefaultScene));
            return () =>
            {
                OpenVRHelper.Update();
                RenderToyEventSource.Default.RenderBegin();
                RenderToyEventSource.Default.MarkerBegin("Update");
                int COUNT_OBJECT = scene.IndexToNodePrimitive.Count;
                for (int i = 0; i < COUNT_OBJECT; ++i)
                {
                    scene.TableTransform[i] = scene.TableNodeTransform[scene.IndexToNodeTransform[i]].Transform;
                }
                RenderToyEventSource.Default.MarkerEnd("Update");
                RenderToyEventSource.Default.MarkerBegin("Execute Compute");
                Task render = Task.Factory.StartNew(() =>
                {
                    var devicePtr = Marshal.GetComInterfaceForObjectInContext(Direct3D11Helper.d3d11Device, typeof(ID3D11Device));
                    {
                        var transformView = OpenVRHelper._head * MathHelp.Invert(OpenVRHelper.GetEyeToHeadTransform(EVREye.Eye_Left));
                        var transformCamera = MathHelp.Invert(transformView);
                        var transformProjection = OpenVRHelper.GetProjectionMatrix(EVREye.Eye_Left, 0.1f, 2000.0f);
                        var matrixData = SceneSerializer.CreateFlatMemoryF32(MathHelp.Invert(transformView * transformProjection));
                        var texturePtr = Marshal.GetComInterfaceForObject(d3d11Texture2D_RT_EyeLeft, typeof(ID3D11Texture2D));
                        RenderToyCLI.TEST_RaycastNormalsAMPF32D3D(sceneData, matrixData, devicePtr, texturePtr);
                    }
                    {
                        var transformView = OpenVRHelper._head * MathHelp.Invert(OpenVRHelper.GetEyeToHeadTransform(EVREye.Eye_Right));
                        var transformCamera = MathHelp.Invert(transformView);
                        var transformProjection = OpenVRHelper.GetProjectionMatrix(EVREye.Eye_Right, 0.1f, 2000.0f);
                        var matrixData = SceneSerializer.CreateFlatMemoryF32(MathHelp.Invert(transformView * transformProjection));
                        var texturePtr = Marshal.GetComInterfaceForObject(d3d11Texture2D_RT_EyeRight, typeof(ID3D11Texture2D));
                        RenderToyCLI.TEST_RaycastNormalsAMPF32D3D(sceneData, matrixData, devicePtr, texturePtr);
                    }
                });
                render.Wait();
                RenderToyEventSource.Default.MarkerEnd("Execute Compute");
                RenderToyEventSource.Default.MarkerBegin("Submit To OpenVR");
                unsafe
                {
                    VRTextureBounds_t* pBounds = null;
                    var textureLeft = new Texture_t { handle = Marshal.GetComInterfaceForObject<ID3D11Texture2D, ID3D11Texture2D>(d3d11Texture2D_RT_EyeLeft), eType = ETextureType.DirectX, eColorSpace = EColorSpace.Auto };
                    OpenVRHelper.Compositor.Submit(EVREye.Eye_Left, ref textureLeft, ref *pBounds, EVRSubmitFlags.Submit_Default);
                    var textureRight = new Texture_t { handle = Marshal.GetComInterfaceForObject<ID3D11Texture2D, ID3D11Texture2D>(d3d11Texture2D_RT_EyeRight), eType = ETextureType.DirectX, eColorSpace = EColorSpace.Auto };
                    OpenVRHelper.Compositor.Submit(EVREye.Eye_Right, ref textureRight, ref *pBounds, EVRSubmitFlags.Submit_Default);
                }
                RenderToyEventSource.Default.MarkerEnd("Submit To OpenVR");
                RenderToyEventSource.Default.RenderEnd();
            };
        }
        public static void CreateThread(SparseScene scene)
        {
            Task.Factory.StartNew(() =>
            {
                var renderer = CreateRenderer(scene);
                while (true)
                {
                    renderer();
                }
            }, TaskCreationOptions.LongRunning);
        }
    }
#endif // OPENVR_INSTALLED
}