﻿////////////////////////////////////////////////////////////////////////////////
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
        public ViewWire()
        {
            render = new SinglePassAsyncAdaptor(RenderCall.Generate(typeof(RenderCS).GetMethod("WireframeCPUF64")), () => Dispatcher.Invoke(InvalidateVisual));
        }
        SinglePassAsyncAdaptor render;
        #region - Overrides : RenderViewportBase -
        protected override void OnSceneChanged(Scene scene)
        {
            base.OnSceneChanged(scene);
            if (render == null) return;
            render.SetScene(scene);
        }
        protected override void OnRenderToy(DrawingContext drawingContext)
        {
            int RENDER_WIDTH = (int)Math.Ceiling(ActualWidth);
            int RENDER_HEIGHT = (int)Math.Ceiling(ActualHeight);
            render.SetCamera(MVP);
            render.SetTarget(RENDER_WIDTH, RENDER_HEIGHT);
            if (RENDER_WIDTH == 0 || RENDER_HEIGHT == 0) return;
            WriteableBitmap bitmap = new WriteableBitmap(RENDER_WIDTH, RENDER_HEIGHT, 0, 0, PixelFormats.Bgra32, null);
            bitmap.Lock();
            render.CopyTo(bitmap.BackBuffer, RENDER_WIDTH, RENDER_HEIGHT, bitmap.BackBufferStride);
            bitmap.AddDirtyRect(new Int32Rect(0, 0, RENDER_WIDTH, RENDER_HEIGHT));
            bitmap.Unlock();
            drawingContext.DrawImage(bitmap, new Rect(0, 0, ActualWidth, ActualHeight));
        }
        #endregion
    }
}