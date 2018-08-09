////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.DirectX;
using System;
using System.Windows;
using System.Windows.Interop;

namespace RenderToy.WPF
{
    /// <summary>
    /// Render the contents of a D3DImage into a control.
    /// This class should be used if you are rendering to a non-D3D9 surface and need a D3D9 backbuffer.
    /// Clients of this class may open a shared handle to the D3D9 image buffer here to interop with D3DImage.
    /// </summary>
    public class ViewD3DImageBuffered : ViewD3DImage
    {
        protected override Size MeasureOverride(Size availableSize)
        {
            NullablePtr<IntPtr> handle = new NullablePtr<IntPtr>(IntPtr.Zero);
            d3d9backbuffer = Direct3D9Helper.device.CreateRenderTarget((uint)availableSize.Width, (uint)availableSize.Height, D3DFormat.A8R8G8B8, D3DMultisample.None, 1, 0, handle);
            d3d9backbufferhandle = handle.Value;
            Target.Lock();
            Target.SetBackBuffer(D3DResourceType.IDirect3DSurface9, d3d9backbuffer.ManagedPtr);
            Target.Unlock();
            return base.MeasureOverride(availableSize);
        }
        // Direct3D9 Handling for D3DImage
        protected Direct3DSurface9 d3d9backbuffer;
        protected IntPtr d3d9backbufferhandle;
    }
}