﻿////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2018
////////////////////////////////////////////////////////////////////////////////

using RenderToy.Cameras;
using RenderToy.RenderControl;
using RenderToy.SceneGraph;
using RenderToy.Utility;
using System;
using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;

namespace RenderToy.WPF
{
    class View3DWire : View3DBase
    {
        public View3DWire()
        {
            IsVisibleChanged += (s, e) =>
            {
                if ((bool)e.NewValue) {
                    render = new SinglePassAsyncAdaptor(RenderCall.Generate(typeof(RenderModeCS).GetMethod("WireframeCPUF64")), () => Dispatcher.Invoke(InvalidateVisual));
                    render.SetScene(Scene);
                } else {
                    render = null;
                }
            };
        }
        SinglePassAsyncAdaptor render;
        #region - Overrides : RenderViewportBase -
        protected override void OnSceneChanged(IScene scene)
        {
            base.OnSceneChanged(scene);
            if (render == null) return;
            render.SetScene(scene);
        }
        protected override void OnRenderToy(DrawingContext drawingContext)
        {
            if (render == null) return;
            int RENDER_WIDTH = (int)Math.Ceiling(ActualWidth);
            int RENDER_HEIGHT = (int)Math.Ceiling(ActualHeight);
            render.SetCamera(ModelViewProjection * Perspective.AspectCorrectFit(ActualWidth, ActualHeight));
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