using System;
using System.Windows;
using System.Windows.Interop;
using System.Windows.Media;

namespace Arcturus.Managed
{
    /// <summary>
    /// Render the contents of a D3DImage into a control.
    /// This class should be used if you are rendering to a non-D3D9 surface and need a D3D9 backbuffer.
    /// Clients of this class may open a shared handle to the D3D9 image buffer here to interop with D3DImage.
    /// </summary>
    public class ViewD3DImage : FrameworkElement
    {
        protected override Size MeasureOverride(Size availableSize)
        {
            d3d9backbuffer = Direct3D9.Device.CreateRenderTarget(new RenderTargetDeclaration { width = (uint)availableSize.Width, height = (uint)availableSize.Height });
            d3d9backbufferhandle = d3d9backbuffer.GetIDirect3DSurface9Handle();
            RenderOptions.SetBitmapScalingMode(Target, BitmapScalingMode.NearestNeighbor);
            Target.Lock();
            Target.SetBackBuffer(D3DResourceType.IDirect3DSurface9, d3d9backbuffer.GetIDirect3DSurface9Pointer());
            Target.Unlock();
            return base.MeasureOverride(availableSize);
        }
        protected override void OnRender(DrawingContext drawingContext)
        {
            drawingContext.DrawImage(Target, new Rect(0, 0, ActualWidth, ActualHeight));
        }
        protected D3DImage Target = new D3DImage();
        protected IRenderTarget_D3D9 d3d9backbuffer;
        protected IntPtr d3d9backbufferhandle;
    }
}