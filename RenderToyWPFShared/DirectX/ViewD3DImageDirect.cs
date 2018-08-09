////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.DirectX;
using System;
using System.Windows;
using System.Windows.Interop;
using System.Windows.Media;

namespace RenderToy.WPF
{
    /// <summary>
    /// Render the contents of a D3DImage into a control.
    /// This class should be used if you are rendering to a D3D9 surface.
    /// Clients of this class will draw directly into a displayable D3D9 RT with no surface copy.
    /// </summary>
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
}