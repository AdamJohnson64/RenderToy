﻿////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System;
using System.Globalization;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Media3D;

namespace RenderToy
{
    public abstract class RenderViewportBase : FrameworkElement
    {
        public static DependencyProperty DrawExtraProperty = DependencyProperty.Register("DrawExtra", typeof(RenderViewport), typeof(RenderViewport));
        public RenderViewportBase DrawExtra { get { return (RenderViewportBase)GetValue(DrawExtraProperty); } set { SetValue(DrawExtraProperty, value);  } }
        public Scene Scene = Scene.Default;
        public RenderViewportBase()
        {
            ReduceQuality_Init();
            AllowDrop = true;
        }
        #region - Section : Camera -
        protected Matrix3D View
        {
            get
            {
                return MathHelp.Invert(Camera.Transform);
            }
        }
        protected Matrix3D Projection
        {
            get
            {
                return CameraMat.Projection;
            }
        }
        protected Matrix3D MVP
        {
            get
            {
                return View * Projection * AspectCorrectFit(ActualWidth, ActualHeight);
            }
        }
        TransformPosQuat Camera = new TransformPosQuat { Position = new Vector3D(0, 10, -20) };
        CameraPerspective CameraMat = new CameraPerspective();
        #endregion
        #region - Section : Aspect Correction -
        public static Matrix3D AspectCorrectFit(double width, double height)
        {
            double aspect = width / height;
            if (aspect > 1)
            {
                return MathHelp.CreateScaleMatrix(1 / aspect, 1, 1);
            }
            else
            {
                return MathHelp.CreateScaleMatrix(1, aspect, 1);
            }
        }
        #endregion
        #region - Section : Input Handling -
        protected override void OnDrop(DragEventArgs e)
        {
            base.OnDrop(e);
            if (e.Data.GetDataPresent(typeof(Sphere)))
            {
                int test = 0;
            }
        }
        protected override void OnMouseLeftButtonDown(MouseButtonEventArgs e)
        {
            base.OnMouseLeftButtonDown(e);
            CaptureMouse();
            Mouse.OverrideCursor = Cursors.None;
            isDragging = true;
            clickOrigin = System.Windows.Forms.Cursor.Position;
            dragOrigin = e.GetPosition(this);
        }
        protected override void OnMouseLeftButtonUp(MouseButtonEventArgs e)
        {
            base.OnMouseLeftButtonUp(e);
            Mouse.OverrideCursor = null;
            ReleaseMouseCapture();
            isDragging = false;
            InvalidateVisual();
        }
        protected override void OnMouseMove(MouseEventArgs e)
        {
            base.OnMouseMove(e);
            if (!isDragging) return;
            System.Windows.Point dragTo = e.GetPosition(this);
            double dx = dragTo.X - dragOrigin.X;
            double dy = dragTo.Y - dragOrigin.Y;
            System.Windows.Forms.Cursor.Position = clickOrigin;
            // Truck Mode (CTRL + SHIFT).
            if (Keyboard.IsKeyDown(Key.LeftShift) && Keyboard.IsKeyDown(Key.LeftCtrl))
            {
                Camera.TranslatePost(new Vector3D(0, 0, dy * -0.05));
                InvalidateVisual();
            }
            else if (!Keyboard.IsKeyDown(Key.LeftShift) && Keyboard.IsKeyDown(Key.LeftCtrl))
            {
                // Rotate Mode (CTRL Only)
                Camera.RotatePre(new Quaternion(new Vector3D(0, 1, 0), dx * 0.05));
                Camera.RotatePost(new Quaternion(new Vector3D(1, 0, 0), dy * 0.05));
                InvalidateVisual();
            }
            else if (!Keyboard.IsKeyDown(Key.LeftShift) && !Keyboard.IsKeyDown(Key.LeftCtrl))
            {
                // Translation Mode (no modifier keys).
                Camera.TranslatePost(new Vector3D(dx * -0.05, dy * 0.05, 0));
                InvalidateVisual();
            }
        }
        protected override void OnMouseWheel(MouseWheelEventArgs e)
        {
            base.OnMouseWheel(e);
            Camera.TranslatePost(new Vector3D(0, 0, e.Delta * 0.01));
            InvalidateVisual();
        }
        bool isDragging = false;
        System.Drawing.Point clickOrigin;
        System.Windows.Point dragOrigin;
        #endregion
        #region - Section : Rendering -
        protected override void OnRender(DrawingContext drawingContext)
        {
            base.OnRender(drawingContext);
            DateTime timeStart = DateTime.Now;
            // Draw our intended visual.
            OnRenderToy(drawingContext);
            // If we're connected to another view camera then show it here.
            if (DrawExtra != null)
            {
                // Draw the clip space of the Model-View-Projection.
                Matrix3D other = MathHelp.Invert(DrawExtra.MVP);
                IWireframeRenderer renderer = new WireframeWPF(drawingContext);
                DrawHelp.fnDrawLineViewport lineviewport = Render.CreateLineViewportFunction(renderer, ActualWidth, ActualHeight);
                DrawHelp.fnDrawLineWorld line = Render.CreateLineWorldFunction(lineviewport, MVP);
                renderer.WireframeBegin();
                renderer.WireframeColor(0.0, 1.0, 1.0);
                DrawHelp.DrawClipSpace(line, other);
                renderer.WireframeEnd();
            }
            DateTime timeEnd = DateTime.Now;
            // Try to maintain a reasonable framerate by reducing quality.
            ReduceQuality_Decide(timeStart, timeEnd);
        }
        protected abstract void OnRenderToy(DrawingContext drawingContext);
        #endregion
        #region - Section : Quality Control -
        protected bool ReduceQuality
        {
            get { return reduceQuality; }
        }
        void ReduceQuality_Init()
        {
            reduceQualityTimer = new System.Windows.Forms.Timer();
            reduceQualityTimer.Interval = 500;
            reduceQualityTimer.Tick += (s, e) =>
            {
                reduceQualityTimer.Enabled = false;
                reduceQualityFrames = 0;
                reduceQuality = false;
                InvalidateVisual();
            };
        }
        void ReduceQuality_Decide(DateTime timeStart, DateTime timeEnd)
        {
            if (timeEnd.Subtract(timeStart).Milliseconds > 1000 / 30)
            {
                ++reduceQualityFrames;
                if (reduceQualityFrames >= 2 && !reduceQuality)
                {
                    reduceQualityFrames = 0;
                    reduceQuality = true;
                }
            }
            // Restart the quality reduction timer.
            if (reduceQuality)
            {
                reduceQualityTimer.Enabled = false;
                reduceQualityTimer.Enabled = true;
            }
        }
        bool reduceQuality = false;
        int reduceQualityFrames = 0;
        System.Windows.Forms.Timer reduceQualityTimer;
        #endregion
    }
    class RenderViewport : RenderViewportBase
    {
        static RoutedUICommand CommandRenderPoint = new RoutedUICommand("Point Render", "CommandRenderPoint", typeof(RenderViewport));
        static RoutedUICommand CommandRenderWireframe = new RoutedUICommand("Wireframe Render", "CommandRenderWireframe", typeof(RenderViewport));
        static RoutedUICommand CommandRenderRaster = new RoutedUICommand("Raster Render", "CommandRenderRaster", typeof(RenderViewport));
        static RoutedUICommand CommandRenderRaytrace = new RoutedUICommand("Raytrace Render", "CommandRenderRaytrace", typeof(RenderViewport));
        static RoutedUICommand CommandRenderD3D9 = new RoutedUICommand("D3D9 Render", "CommandRenderD3D9", typeof(RenderViewport));
        static RoutedUICommand CommandRenderPreviewsToggle = new RoutedUICommand("Toggle Render Previews", "CommandRenderPreviewsToggle", typeof(RenderViewport));
        static RoutedUICommand CommandRenderWireframeToggle = new RoutedUICommand("Toggle Render Wireframe", "CommandRenderWireframeToggle", typeof(RenderViewport));
        static RenderViewport()
        {
            CommandRenderPoint.InputGestures.Add(new KeyGesture(Key.D1, ModifierKeys.Control));
            CommandRenderWireframe.InputGestures.Add(new KeyGesture(Key.D2, ModifierKeys.Control));
            CommandRenderRaster.InputGestures.Add(new KeyGesture(Key.D3, ModifierKeys.Control));
            CommandRenderRaytrace.InputGestures.Add(new KeyGesture(Key.D4, ModifierKeys.Control));
            CommandRenderD3D9.InputGestures.Add(new KeyGesture(Key.D0, ModifierKeys.Control));
            CommandRenderPreviewsToggle.InputGestures.Add(new KeyGesture(Key.P, ModifierKeys.Control));
            CommandRenderWireframeToggle.InputGestures.Add(new KeyGesture(Key.W, ModifierKeys.Control));
        }
        public RenderViewport()
        {
            CommandBindings.Add(new CommandBinding(CommandRenderPoint, (s, e) => { renderMode = RenderMode.Point; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderWireframe, (s, e) => { renderMode = RenderMode.Wireframe; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderRaster, (s, e) => { renderMode = RenderMode.Raster; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderRaytrace, (s, e) => { renderMode = RenderMode.Raytrace; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderD3D9, (s, e) => { renderMode = RenderMode.Direct3D9; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderPreviewsToggle, (s, e) => { renderPreviews = !renderPreviews; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderWireframeToggle, (s, e) => { renderWireframe = !renderWireframe; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            InputBindings.Add(new KeyBinding(CommandRenderPoint, Key.D1, ModifierKeys.Control));
            InputBindings.Add(new KeyBinding(CommandRenderWireframe, Key.D2, ModifierKeys.Control));
            InputBindings.Add(new KeyBinding(CommandRenderRaster, Key.D3, ModifierKeys.Control));
            InputBindings.Add(new KeyBinding(CommandRenderRaytrace, Key.D4, ModifierKeys.Control));
            InputBindings.Add(new KeyBinding(CommandRenderD3D9, Key.D0, ModifierKeys.Control));
            InputBindings.Add(new KeyBinding(CommandRenderPreviewsToggle, Key.P, ModifierKeys.Control));
            InputBindings.Add(new KeyBinding(CommandRenderWireframeToggle, Key.W, ModifierKeys.Control));
            Focusable = true;
        }
        enum RenderMode { Point, Wireframe, Raster, Raytrace, Direct3D9 }
        RenderMode renderMode = RenderMode.Wireframe;
        bool renderPreviews = true;
        bool renderWireframe = false;
        protected override void OnMouseDown(MouseButtonEventArgs e)
        {
            base.OnMouseDown(e);
            Focus();
        }
        protected override void OnRenderToy(DrawingContext drawingContext)
        {
            switch (renderMode)
            {
                case RenderMode.Point:
                    ImageHelp.DrawImage(Render.Point, Scene, MVP, ReduceQuality ? 256 : (int)Math.Ceiling(ActualWidth), ReduceQuality ? 256 : (int)Math.Ceiling(ActualHeight), drawingContext, ActualWidth, ActualHeight);
                    break;
                case RenderMode.Wireframe:
                    ImageHelp.DrawImage(Render.Wireframe, Scene, MVP, ReduceQuality ? 256 : (int)Math.Ceiling(ActualWidth), ReduceQuality ? 256 : (int)Math.Ceiling(ActualHeight), drawingContext, ActualWidth, ActualHeight);
                    break;
                case RenderMode.Raster:
                    ImageHelp.DrawImage(Render.Raster, Scene, MVP, ReduceQuality ? 256 : (int)Math.Ceiling(ActualWidth), ReduceQuality ? 256 : (int)Math.Ceiling(ActualHeight), drawingContext, ActualWidth, ActualHeight);
                    break;
                case RenderMode.Raytrace:
                    ImageHelp.DrawImage(Render.Raytrace, Scene, MVP, ReduceQuality ? 128 : (int)Math.Ceiling(ActualWidth), ReduceQuality ? 128 : (int)Math.Ceiling(ActualHeight), drawingContext, ActualWidth, ActualHeight);
                    break;
                case RenderMode.Direct3D9:
                    ImageHelp.DrawImage(Render.RasterD3D9, Scene, MVP, (int)Math.Ceiling(ActualWidth), (int)Math.Ceiling(ActualHeight), drawingContext, ActualWidth, ActualHeight);
                    break;
            }
            if (renderWireframe)
            {
                ImageHelp.DrawImage(Render.Wireframe, Scene, MVP, ReduceQuality ? 256 : (int)Math.Ceiling(ActualWidth), ReduceQuality ? 256 : (int)Math.Ceiling(ActualHeight), drawingContext, ActualWidth, ActualHeight);
            }
            if (renderPreviews)
            {
                Action<ImageHelp.FillFunction, int, int, int> drawpreview = (fillwith, stacky, render_width, render_height) =>
                {
                    double frame_l = ActualWidth - 128 - 8;
                    double frame_t = 8 + (96 + 8) * stacky;
                    double frame_w = 128;
                    double frame_h = 96;
                    double image_l = frame_l + 8;
                    double image_t = frame_t + 8;
                    double image_w = frame_w - 8 * 2;
                    double image_h = frame_h - 8 * 2;
                    drawingContext.DrawRoundedRectangle(Brushes.White, new Pen(Brushes.DarkGray, 2), new Rect(frame_l, frame_t, frame_w, frame_h), 8, 8);
                    var imagesource = ImageHelp.CreateImage(fillwith, Scene, View * Projection * AspectCorrectFit(image_w, image_h), render_width, render_height);
                    drawingContext.DrawImage(imagesource, new Rect(image_l, image_t, image_w, image_h));
                };
                drawpreview(Render.Point, 0, ReduceQuality ? 32 : 64, ReduceQuality ? 32 : 64);
                drawpreview(Render.Wireframe, 1, ReduceQuality ? 32 : 128, ReduceQuality ? 32 : 128);
                drawpreview(Render.Raster, 2, ReduceQuality ? 32 : 128, ReduceQuality ? 32 : 128);
                drawpreview(Render.Raytrace, 3, ReduceQuality ? 32 : 128, ReduceQuality ? 32 : 128);
            }
            drawingContext.DrawText(new FormattedText(renderMode.ToString(), CultureInfo.InvariantCulture, FlowDirection.LeftToRight, new Typeface("Arial"), 24, Brushes.LightGray), new Point(10, 10));
            drawingContext.DrawText(new FormattedText(renderMode.ToString(), CultureInfo.InvariantCulture, FlowDirection.LeftToRight, new Typeface("Arial"), 24, Brushes.DarkGray), new Point(8, 8));
        }
    }
    class RenderViewportPoint : RenderViewportBase
    {
        protected override void OnRenderToy(DrawingContext drawingContext)
        {
            ImageHelp.DrawImage(Render.Point, Scene, MVP, (int)Math.Ceiling(ActualWidth), (int)Math.Ceiling(ActualHeight), drawingContext, ActualWidth, ActualHeight);
        }
    }
    class RenderViewportWireframeGDI : RenderViewportBase
    {
        protected override void OnRenderToy(DrawingContext drawingContext)
        {
            Render.WireframeGDI(Scene, MVP, (int)Math.Ceiling(ActualWidth), (int)Math.Ceiling(ActualHeight), drawingContext, ActualWidth, ActualHeight);
        }
    }
    class RenderViewportWireframeWPF : RenderViewportBase
    {
        protected override void OnRenderToy(DrawingContext drawingContext)
        {
            Render.WireframeWPF(Scene, MVP, drawingContext, ActualWidth, ActualHeight);
        }
    }
    class RenderViewportRaster : RenderViewportBase
    {
        protected override void OnRenderToy(DrawingContext drawingContext)
        {
            ImageHelp.DrawImage(Render.Raster, Scene, MVP, ReduceQuality ? 128 : 512, ReduceQuality ? 128 : 512, drawingContext, ActualWidth, ActualHeight);
        }
    }
    class RenderViewportRasterD3D : RenderViewportBase
    {
        protected override void OnRenderToy(DrawingContext drawingContext)
        {
            ImageHelp.DrawImage(Render.RasterD3D9, Scene, MVP, (int)Math.Ceiling(ActualWidth), (int)Math.Ceiling(ActualHeight), drawingContext, ActualWidth, ActualHeight);
        }
    }
    class RenderViewportRaytrace : RenderViewportBase
    {
        protected override void OnRenderToy(DrawingContext drawingContext)
        {
            ImageHelp.DrawImage(Render.Raytrace, Scene, MVP, ReduceQuality ? 128 : 512, ReduceQuality ? 128 : 512, drawingContext, ActualWidth, ActualHeight);
        }
    }
}