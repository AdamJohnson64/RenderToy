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
        public static SparseScene Scene
        {
            set
            {
                _Scene = value;
                Execute_RenderSceneLeft = Direct3D11Helper.CreateSceneDraw(_Scene);
                Execute_RenderSceneRight = Direct3D11Helper.CreateSceneDraw(_Scene);
            }
        }
        static Action CreateRenderer()
        {
            uint vrwidth = 0, vrheight = 0;
            OpenVRHelper.System.GetRecommendedRenderTargetSize(ref vrwidth, ref vrheight);
            ID3D11Texture2D d3d11Texture2D_RT_EyeLeft = null;
            ID3D11RenderTargetView d3d11RenderTargetView_EyeLeft = null;
            ID3D11Texture2D d3d11Texture2D_DS_EyeLeft = null;
            ID3D11DepthStencilView d3d11DepthStencilView_EyeLeft = null;
            ID3D11Texture2D d3d11Texture2D_RT_EyeRight = null;
            ID3D11RenderTargetView d3d11RenderTargetView_EyeRight = null;
            ID3D11Texture2D d3d11Texture2D_DS_EyeRight = null;
            ID3D11DepthStencilView d3d11DepthStencilView_EyeRight = null;
            {
                var d3d11Texture2DDesc_RT = new D3D11_TEXTURE2D_DESC { Width = (uint)vrwidth, Height = (uint)vrheight, MipLevels = 1, ArraySize = 1, Format = DXGI_FORMAT.DXGI_FORMAT_B8G8R8A8_UNORM, SampleDesc = new DXGI_SAMPLE_DESC { Count = 1, Quality = 0 }, Usage = D3D11_USAGE.D3D11_USAGE_DEFAULT, BindFlags = (uint)D3D11_BIND_FLAG.D3D11_BIND_RENDER_TARGET | (uint)D3D11_BIND_FLAG.D3D11_BIND_SHADER_RESOURCE, CPUAccessFlags = 0 };
                var d3d11Texture2DDesc_DS = new D3D11_TEXTURE2D_DESC { Width = (uint)vrwidth, Height = (uint)vrheight, MipLevels = 1, ArraySize = 1, Format = DXGI_FORMAT.DXGI_FORMAT_D32_FLOAT, SampleDesc = new DXGI_SAMPLE_DESC { Count = 1, Quality = 0 }, Usage = D3D11_USAGE.D3D11_USAGE_DEFAULT, BindFlags = (uint)D3D11_BIND_FLAG.D3D11_BIND_DEPTH_STENCIL, CPUAccessFlags = 0 };
                var d3d11RenderTargetView_RT = new D3D11_RENDER_TARGET_VIEW_DESC { Format = DXGI_FORMAT.DXGI_FORMAT_B8G8R8A8_UNORM, ViewDimension = D3D11_RTV_DIMENSION.D3D11_RTV_DIMENSION_TEXTURE2D };
                d3d11RenderTargetView_RT.__MIDL____MIDL_itf_RenderToy_0005_00650002.Texture2D = new D3D11_TEX2D_RTV { MipSlice = 0 };
                var d3d11DepthStencilView_DS = new D3D11_DEPTH_STENCIL_VIEW_DESC { Format = DXGI_FORMAT.DXGI_FORMAT_D32_FLOAT, ViewDimension = D3D11_DSV_DIMENSION.D3D11_DSV_DIMENSION_TEXTURE2D };
                d3d11DepthStencilView_DS.__MIDL____MIDL_itf_RenderToy_0005_00660000.Texture2D = new D3D11_TEX2D_DSV { MipSlice = 0 };
                Direct3D11Helper.Dispatcher.Invoke(() =>
                {
                    unsafe
                    {
                        D3D11_SUBRESOURCE_DATA* pInitialData = null;
                        Direct3D11Helper.d3d11Device.CreateTexture2D(d3d11Texture2DDesc_RT, ref *pInitialData, ref d3d11Texture2D_RT_EyeLeft);
                        Direct3D11Helper.d3d11Device.CreateTexture2D(d3d11Texture2DDesc_DS, ref *pInitialData, ref d3d11Texture2D_DS_EyeLeft);
                        Direct3D11Helper.d3d11Device.CreateTexture2D(d3d11Texture2DDesc_RT, ref *pInitialData, ref d3d11Texture2D_RT_EyeRight);
                        Direct3D11Helper.d3d11Device.CreateTexture2D(d3d11Texture2DDesc_DS, ref *pInitialData, ref d3d11Texture2D_DS_EyeRight);
                    }
                    Direct3D11Helper.d3d11Device.CreateRenderTargetView(d3d11Texture2D_RT_EyeLeft, d3d11RenderTargetView_RT, ref d3d11RenderTargetView_EyeLeft);
                    Direct3D11Helper.d3d11Device.CreateRenderTargetView(d3d11Texture2D_RT_EyeRight, d3d11RenderTargetView_RT, ref d3d11RenderTargetView_EyeRight);
                    Direct3D11Helper.d3d11Device.CreateDepthStencilView(d3d11Texture2D_DS_EyeLeft, d3d11DepthStencilView_DS, ref d3d11DepthStencilView_EyeLeft);
                    Direct3D11Helper.d3d11Device.CreateDepthStencilView(d3d11Texture2D_DS_EyeRight, d3d11DepthStencilView_DS, ref d3d11DepthStencilView_EyeRight);
                });
            }
            ////////////////////////////////////////////////////////////////////////////////
            // Standard viewport constants.
            var scissorRect = new tagRECT { left = 0, top = 0, right = (int)vrwidth, bottom = (int)vrheight };
            var viewportRect = new D3D11_VIEWPORT { TopLeftX = 0, TopLeftY = 0, Width = vrwidth, Height = vrheight, MinDepth = 0, MaxDepth = 1 };
            ////////////////////////////////////////////////////////////////////////////////
            // Rendering constants for the left eye.
            var constants_left = new Dictionary<string, object>();
            constants_left["profilingName"] = "Left Eye";
            constants_left["transformAspect"] = Matrix3D.Identity;
            constants_left["transformCamera"] = Matrix3D.Identity;
            constants_left["transformView"] = Matrix3D.Identity;
            constants_left["transformProjection"] = OpenVRHelper.GetProjectionMatrix(EVREye.Eye_Left, 0.1f, 2000.0f);
            ////////////////////////////////////////////////////////////////////////////////
            // Command buffer builder for the left eye.
            Func<ID3D11CommandList> action_left = () =>
            {
                RenderToyEventSource.Default.MarkerBegin("Command Buffer (Left Eye)");
                ID3D11DeviceContext deferred_left_old = null;
                Direct3D11Helper.d3d11Device.CreateDeferredContext(0, ref deferred_left_old);
                var deferred_left = (ID3D11DeviceContext4)deferred_left_old;
                deferred_left.RSSetScissorRects(1, scissorRect);
                deferred_left.RSSetViewports(1, viewportRect);
                deferred_left.OMSetRenderTargets(1, d3d11RenderTargetView_EyeLeft, d3d11DepthStencilView_EyeLeft);
                deferred_left.ClearDepthStencilView(d3d11DepthStencilView_EyeLeft, (uint)D3D11_CLEAR_FLAG.D3D11_CLEAR_DEPTH, 1, 0);
                deferred_left.ClearRenderTargetView(d3d11RenderTargetView_EyeLeft, new float[] { 0, 0, 0, 0 });
                var transformView = OpenVRHelper.TransformHead * MathHelp.Invert(OpenVRHelper.GetEyeToHeadTransform(EVREye.Eye_Left));
                constants_left["transformCamera"] = MathHelp.Invert(transformView);
                constants_left["transformView"] = transformView;
                Execute_RenderSceneLeft(deferred_left, constants_left);
                ID3D11CommandList commandList = null;
                deferred_left.FinishCommandList(0, ref commandList);
                RenderToyEventSource.Default.MarkerEnd("Command Buffer (Left Eye)");
                return commandList;
            };
            ////////////////////////////////////////////////////////////////////////////////
            // Rendering constants for the right eye.
            var constants_right = new Dictionary<string, object>();
            constants_right["profilingName"] = "Right Eye";
            constants_right["transformAspect"] = Matrix3D.Identity;
            constants_right["transformCamera"] = Matrix3D.Identity;
            constants_right["transformView"] = Matrix3D.Identity;
            constants_right["transformProjection"] = OpenVRHelper.GetProjectionMatrix(EVREye.Eye_Right, 0.1f, 2000.0f);
            ////////////////////////////////////////////////////////////////////////////////
            // Command buffer builder for the right eye.
            Func<ID3D11CommandList> action_right = () =>
            {
                RenderToyEventSource.Default.MarkerBegin("Command Buffer (Right Eye)");
                ID3D11DeviceContext deferred_right_old = null;
                Direct3D11Helper.d3d11Device.CreateDeferredContext(0, ref deferred_right_old);
                var deferred_right = (ID3D11DeviceContext4)deferred_right_old;
                deferred_right.RSSetScissorRects(1, ref scissorRect);
                deferred_right.RSSetViewports(1, ref viewportRect);
                deferred_right.OMSetRenderTargets(1, ref d3d11RenderTargetView_EyeRight, d3d11DepthStencilView_EyeRight);
                deferred_right.ClearDepthStencilView(d3d11DepthStencilView_EyeRight, (uint)D3D11_CLEAR_FLAG.D3D11_CLEAR_DEPTH, 1, 0);
                deferred_right.ClearRenderTargetView(d3d11RenderTargetView_EyeRight, new float[] { 0, 0, 0, 0 });
                var transformView = OpenVRHelper.TransformHead * MathHelp.Invert(OpenVRHelper.GetEyeToHeadTransform(EVREye.Eye_Right));
                constants_right["transformCamera"] = MathHelp.Invert(transformView);
                constants_right["transformView"] = transformView;
                Execute_RenderSceneRight(deferred_right, constants_right);
                ID3D11CommandList commandList = null;
                deferred_right.FinishCommandList(0, ref commandList);
                RenderToyEventSource.Default.MarkerEnd("Command Buffer (Right Eye)");
                return commandList;
            };
            ////////////////////////////////////////////////////////////////////////////////
            // Single frame renderer function (renders both eyes and submits to OpenVR).
            return () =>
            {
                OpenVRHelper.Update();
                RenderToyEventSource.Default.RenderBegin();
                // If there's nothing to draw then just exit now.
                if (!(_Scene == null || Execute_RenderSceneLeft == null || Execute_RenderSceneRight == null))
                {
                    RenderToyEventSource.Default.MarkerBegin("Update");
                    int COUNT_OBJECT = _Scene.IndexToNodePrimitive.Count;
                    for (int i = 0; i < COUNT_OBJECT; ++i)
                    {
                        _Scene.TableTransform[i] = _Scene.TableNodeTransform[_Scene.IndexToNodeTransform[i]].Transform;
                    }
                    RenderToyEventSource.Default.MarkerEnd("Update");
                    var do_left = Task.Run(action_left);
                    var do_right = Task.Run(action_right);
                    RenderToyEventSource.Default.MarkerBegin("Wait Render Completion");
                    var cmd_left = do_left.Result;
                    var cmd_right = do_right.Result;
                    RenderToyEventSource.Default.MarkerEnd("Wait Render Completion");
                    {
                        RenderToyEventSource.Default.MarkerBegin("Execute All RTs");
                        ID3D11DeviceContext context_old = null;
                        Direct3D11Helper.Dispatcher.Invoke(() =>
                        {
                            Direct3D11Helper.d3d11Device.GetImmediateContext(ref context_old);
                            var context = (ID3D11DeviceContext4)context_old;
                            context.ExecuteCommandList(cmd_left, 1);
                            context.ExecuteCommandList(cmd_right, 1);
                        });
                        RenderToyEventSource.Default.MarkerEnd("Execute All RTs");
                    }
                }
                {
                    RenderToyEventSource.Default.MarkerBegin("Submit To OpenVR");
                    Direct3D11Helper.Dispatcher.Invoke(() =>
                    {
                        unsafe
                        {
                            VRTextureBounds_t* pBounds = null;
                            var textureLeft = new Texture_t { handle = Marshal.GetComInterfaceForObject<ID3D11Texture2D, ID3D11Texture2D>(d3d11Texture2D_RT_EyeLeft), eType = ETextureType.DirectX, eColorSpace = EColorSpace.Auto };
                            OpenVRHelper.Compositor.Submit(EVREye.Eye_Left, ref textureLeft, ref *pBounds, EVRSubmitFlags.Submit_Default);
                            var textureRight = new Texture_t { handle = Marshal.GetComInterfaceForObject<ID3D11Texture2D, ID3D11Texture2D>(d3d11Texture2D_RT_EyeRight), eType = ETextureType.DirectX, eColorSpace = EColorSpace.Auto };
                            OpenVRHelper.Compositor.Submit(EVREye.Eye_Right, ref textureRight, ref *pBounds, EVRSubmitFlags.Submit_Default);
                        }
                    });
                    RenderToyEventSource.Default.MarkerEnd("Submit To OpenVR");
                    RenderToyEventSource.Default.RenderEnd();
                }
            };
        }
        public static void CreateThread(SparseScene scene)
        {
            Task.Factory.StartNew(() =>
            {
                var renderer = CreateRenderer();
                Scene = scene;
                while (true)
                {
                    renderer();
                }
            }, TaskCreationOptions.LongRunning);
        }
        static SparseScene _Scene = null;
        static Action<ID3D11DeviceContext4, Dictionary<string, object>> Execute_RenderSceneLeft = null;
        static Action<ID3D11DeviceContext4, Dictionary<string, object>> Execute_RenderSceneRight = null;
    }
#endif // OPENVR_INSTALLED
}