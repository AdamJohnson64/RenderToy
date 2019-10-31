////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using Arcturus.Managed;
using System;
using System.Windows;
using System.Windows.Threading;

namespace RenderToy.WPF
{
    class ViewArcturus : ViewD3DImageBuffered
    {
        public ViewArcturus()
        {
            timer = new DispatcherTimer(TimeSpan.FromMilliseconds(1000 / 60), DispatcherPriority.ApplicationIdle, (s, e) => { Render(); }, Application.Current.Dispatcher);
            timer.Start();
            device = IDevice3D.CreateDevice3D_Direct3D12();
        }
        void Render()
        {
            device.BeginRender();
            device.BeginPass(rendertarget, new Color { r = 0, g = 0, b = 0.25f, a = 1 });
            device.EndPass();
            device.EndRender();
            Target.Lock();
            Target.AddDirtyRect(new Int32Rect(0, 0, rendertargetwidth, rendertargetheight));
            Target.Unlock();
        }
        protected override Size MeasureOverride(Size availableSize)
        {
            Size size = base.MeasureOverride(availableSize);
            rendertargetwidth = (int)availableSize.Width;
            rendertargetheight = (int)availableSize.Height;
            RenderTargetDeclaration descRenderTarget;
            descRenderTarget.width = (uint)rendertargetwidth;
            descRenderTarget.height = (uint)rendertargetheight;
            rendertarget = device.OpenRenderTarget(descRenderTarget, d3d9backbufferhandle);
            return size;
        }
        DispatcherTimer timer;
        IDevice3D device;
        IRenderTarget rendertarget;
        int rendertargetwidth;
        int rendertargetheight;
    }
}