////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace RenderToy
{
    class ViewWire : ViewBase
    {
        #region - Overrides : RenderViewportBase -
        protected override void OnRenderToy(DrawingContext drawingContext)
        {
            int RENDER_WIDTH = (int)Math.Ceiling(ActualWidth);
            int RENDER_HEIGHT = (int)Math.Ceiling(ActualHeight);
            if (RENDER_WIDTH == 0 || RENDER_HEIGHT == 0) return;
            WriteableBitmap bitmap = new WriteableBitmap(RENDER_WIDTH, RENDER_HEIGHT, 0, 0, PixelFormats.Bgra32, null);
            bitmap.Lock();
            RenderCS.WireframeCPUF64(Scene, MVP, bitmap.BackBuffer, RENDER_WIDTH, RENDER_HEIGHT, bitmap.BackBufferStride);
            bitmap.AddDirtyRect(new Int32Rect(0, 0, RENDER_WIDTH, RENDER_HEIGHT));
            bitmap.Unlock();
            drawingContext.DrawImage(bitmap, new Rect(0, 0, ActualWidth, ActualHeight));
        }
        #endregion
    }
}