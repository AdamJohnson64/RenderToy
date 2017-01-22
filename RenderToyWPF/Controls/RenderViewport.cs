////////////////////////////////////////////////////////////////////////////////
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
        public static DependencyProperty DrawExtraProperty = DependencyProperty.Register("DrawExtra", typeof(RenderViewportBase), typeof(RenderViewportBase));
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
        TransformPosQuat Camera = new TransformPosQuat { Position = new Point3D(0, 2, -5) };
        CameraPerspective CameraMat = new CameraPerspective();
        #endregion
        #region - Section : Aspect Correction -
        public static Matrix3D AspectCorrectFit(double width, double height)
        {
            double aspect = width / height;
            if (aspect > 1)
            {
                return MathHelp.CreateMatrixScale(1 / aspect, 1, 1);
            }
            else
            {
                return MathHelp.CreateMatrixScale(1, aspect, 1);
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
            dragFrom = System.Windows.Forms.Cursor.Position;
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
            System.Drawing.Point dragTo = System.Windows.Forms.Cursor.Position;
            double dx = dragTo.X - dragFrom.X;
            double dy = dragTo.Y - dragFrom.Y;
            System.Windows.Forms.Cursor.Position = dragFrom;
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
        System.Drawing.Point dragFrom;
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
                DrawHelp.fnDrawLineWorld line = AbstractLineRenderer.CreateLineWorldFunction(renderer, ActualWidth, ActualHeight, MVP);
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
        static RoutedUICommand CommandRenderRaycastCPU = new RoutedUICommand("Raycast Render (CPU)", "CommandRenderRaycastCPU", typeof(RenderViewport));
        static RoutedUICommand CommandRenderRaycastNormalsCPU = new RoutedUICommand("Raycast Normals Render (CPU)", "CommandRenderRaycastNormalsCPU", typeof(RenderViewport));
        static RoutedUICommand CommandRenderRaytraceCPU = new RoutedUICommand("Raytrace Render (CPU)", "CommandRenderRaytraceCPU", typeof(RenderViewport));
        static RoutedUICommand CommandRenderRaycastCUDA = new RoutedUICommand("Raycast Render (CUDA)", "CommandRenderRaycastCUDA", typeof(RenderViewport));
        static RoutedUICommand CommandRenderRaycastNormalsCUDA = new RoutedUICommand("Raycast Normals Render (CUDA)", "CommandRenderRaycastNormalsCUDA", typeof(RenderViewport));
        static RoutedUICommand CommandRenderRaytraceCUDA = new RoutedUICommand("Raytrace Render (CUDA)", "CommandRenderRaytraceCUDA", typeof(RenderViewport));
        static RoutedUICommand CommandRenderD3D9 = new RoutedUICommand("D3D9 Render", "CommandRenderD3D9", typeof(RenderViewport));
        static RoutedUICommand CommandRenderPreviewsToggle = new RoutedUICommand("Toggle Render Previews", "CommandRenderPreviewsToggle", typeof(RenderViewport));
        static RoutedUICommand CommandRenderWireframeToggle = new RoutedUICommand("Toggle Render Wireframe", "CommandRenderWireframeToggle", typeof(RenderViewport));
        public RenderViewport()
        {
            CommandBindings.Add(new CommandBinding(CommandRenderPoint, (s, e) => { renderMode = RenderMode.Point; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderWireframe, (s, e) => { renderMode = RenderMode.Wireframe; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderRaster, (s, e) => { renderMode = RenderMode.Raster; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderRaycastCPU, (s, e) => { renderMode = RenderMode.RaycastCPU; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderRaycastNormalsCPU, (s, e) => { renderMode = RenderMode.RaycastNormalsCPU; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderRaytraceCPU, (s, e) => { renderMode = RenderMode.RaytraceCPU; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderRaycastCUDA, (s, e) => { renderMode = RenderMode.RaycastCUDA; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderRaycastNormalsCUDA, (s, e) => { renderMode = RenderMode.RaycastNormalsCUDA; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderRaytraceCUDA, (s, e) => { renderMode = RenderMode.RaytraceCUDA; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderD3D9, (s, e) => { renderMode = RenderMode.Direct3D9; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderPreviewsToggle, (s, e) => { renderPreviews = !renderPreviews; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderWireframeToggle, (s, e) => { renderWireframe = !renderWireframe; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            InputBindings.Add(new KeyBinding(CommandRenderPoint, Key.D1, ModifierKeys.Control));
            InputBindings.Add(new KeyBinding(CommandRenderWireframe, Key.D2, ModifierKeys.Control));
            InputBindings.Add(new KeyBinding(CommandRenderRaster, Key.D3, ModifierKeys.Control));
            InputBindings.Add(new KeyBinding(CommandRenderRaycastCPU, Key.D4, ModifierKeys.Control));
            InputBindings.Add(new KeyBinding(CommandRenderRaycastNormalsCPU, Key.D5, ModifierKeys.Control));
            InputBindings.Add(new KeyBinding(CommandRenderRaytraceCPU, Key.D6, ModifierKeys.Control));
            InputBindings.Add(new KeyBinding(CommandRenderRaycastCUDA, Key.D7, ModifierKeys.Control));
            InputBindings.Add(new KeyBinding(CommandRenderRaycastNormalsCUDA, Key.D8, ModifierKeys.Control));
            InputBindings.Add(new KeyBinding(CommandRenderRaytraceCUDA, Key.D9, ModifierKeys.Control));
            InputBindings.Add(new KeyBinding(CommandRenderD3D9, Key.D0, ModifierKeys.Control));
            InputBindings.Add(new KeyBinding(CommandRenderPreviewsToggle, Key.P, ModifierKeys.Control));
            InputBindings.Add(new KeyBinding(CommandRenderWireframeToggle, Key.W, ModifierKeys.Control));
            Focusable = true;
        }
        enum RenderMode { Point, Wireframe, Raster, RaycastCPU, RaycastNormalsCPU, RaytraceCPU, RaycastCUDA, RaycastNormalsCUDA, RaytraceCUDA, Direct3D9 }
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
                    drawingContext.DrawImage(ImageHelp.CreateImage(Render.Point, Scene, MVP, ReduceQuality ? 256 : (int)Math.Ceiling(ActualWidth), ReduceQuality ? 256 : (int)Math.Ceiling(ActualHeight)), new Rect(0, 0, ActualWidth, ActualHeight));
                    break;
                case RenderMode.Wireframe:
                    drawingContext.DrawImage(ImageHelp.CreateImage(Render.Wireframe, Scene, MVP, ReduceQuality ? 256 : (int)Math.Ceiling(ActualWidth), ReduceQuality ? 256 : (int)Math.Ceiling(ActualHeight)), new Rect(0, 0, ActualWidth, ActualHeight));
                    break;
                case RenderMode.Raster:
                    drawingContext.DrawImage(ImageHelp.CreateImage(Render.Raster, Scene, MVP, ReduceQuality ? 256 : (int)Math.Ceiling(ActualWidth), ReduceQuality ? 256 : (int)Math.Ceiling(ActualHeight)), new Rect(0, 0, ActualWidth, ActualHeight));
                    break;
                case RenderMode.RaycastCPU:
                    drawingContext.DrawImage(ImageHelp.CreateImage(Render.RaycastCPU, Scene, MVP, (int)Math.Ceiling(ActualWidth) / (ReduceQuality ? 2 : 1), (int)Math.Ceiling(ActualHeight) / (ReduceQuality ? 2 : 1)), new Rect(0, 0, ActualWidth, ActualHeight));
                    break;
                case RenderMode.RaycastNormalsCPU:
                    drawingContext.DrawImage(ImageHelp.CreateImage(Render.RaycastNormalsCPU, Scene, MVP, (int)Math.Ceiling(ActualWidth) / (ReduceQuality ? 2 : 1), (int)Math.Ceiling(ActualHeight) / (ReduceQuality ? 2 : 1)), new Rect(0, 0, ActualWidth, ActualHeight));
                    break;
                case RenderMode.RaytraceCPU:
                    drawingContext.DrawImage(ImageHelp.CreateImage(Render.RaytraceCPU, Scene, MVP, (int)Math.Ceiling(ActualWidth) / (ReduceQuality ? 2 : 1), (int)Math.Ceiling(ActualHeight) / (ReduceQuality ? 2 : 1)), new Rect(0, 0, ActualWidth, ActualHeight));
                    break;
                case RenderMode.RaycastCUDA:
                    if (Render.CUDAAvailable())
                    {
                        drawingContext.DrawImage(ImageHelp.CreateImage(Render.RaycastCUDA, Scene, MVP, (int)Math.Ceiling(ActualWidth) / (ReduceQuality ? 2 : 1), (int)Math.Ceiling(ActualHeight) / (ReduceQuality ? 2 : 1)), new Rect(0, 0, ActualWidth, ActualHeight));
                    }
                    else
                    {
                        drawingContext.DrawText(new FormattedText("CUDA is not available or was not built for this configuration.", CultureInfo.InvariantCulture, FlowDirection.LeftToRight, new Typeface("Arial"), 24, Brushes.Red), new Point(8, ActualHeight / 2));
                    }
                    break;
                case RenderMode.RaycastNormalsCUDA:
                    if (Render.CUDAAvailable())
                    {
                        drawingContext.DrawImage(ImageHelp.CreateImage(Render.RaycastNormalsCUDA, Scene, MVP, (int)Math.Ceiling(ActualWidth) / (ReduceQuality ? 2 : 1), (int)Math.Ceiling(ActualHeight) / (ReduceQuality ? 2 : 1)), new Rect(0, 0, ActualWidth, ActualHeight));
                    }
                    else
                    {
                        drawingContext.DrawText(new FormattedText("CUDA is not available or was not built for this configuration.", CultureInfo.InvariantCulture, FlowDirection.LeftToRight, new Typeface("Arial"), 24, Brushes.Red), new Point(8, ActualHeight / 2));
                    }
                    break;
                case RenderMode.RaytraceCUDA:
                    if (Render.CUDAAvailable())
                    {
                        drawingContext.DrawImage(ImageHelp.CreateImage(Render.RaytraceCUDA, Scene, MVP, (int)Math.Ceiling(ActualWidth) / (ReduceQuality ? 2 : 1), (int)Math.Ceiling(ActualHeight) / (ReduceQuality ? 2 : 1)), new Rect(0, 0, ActualWidth, ActualHeight));
                    }
                    else
                    {
                        drawingContext.DrawText(new FormattedText("CUDA is not available or was not built for this configuration.", CultureInfo.InvariantCulture, FlowDirection.LeftToRight, new Typeface("Arial"), 24, Brushes.Red), new Point(8, ActualHeight / 2));
                    }
                    break;
                case RenderMode.Direct3D9:
                    drawingContext.DrawImage(ImageHelp.CreateImage(Render.RasterD3D9, Scene, MVP, (int)Math.Ceiling(ActualWidth), (int)Math.Ceiling(ActualHeight)), new Rect(0, 0, ActualWidth, ActualHeight));
                    break;
            }
            if (renderWireframe)
            {
                drawingContext.PushOpacity(0.5);
                drawingContext.DrawImage(ImageHelp.CreateImage(Render.Wireframe, Scene, MVP, ReduceQuality ? 256 : (int)Math.Ceiling(ActualWidth), ReduceQuality ? 256 : (int)Math.Ceiling(ActualHeight)), new Rect(0, 0, ActualWidth, ActualHeight));
                drawingContext.Pop();
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
            }
            drawingContext.DrawText(new FormattedText(renderMode.ToString(), CultureInfo.InvariantCulture, FlowDirection.LeftToRight, new Typeface("Arial"), 24, Brushes.LightGray), new Point(10, 10));
            drawingContext.DrawText(new FormattedText(renderMode.ToString(), CultureInfo.InvariantCulture, FlowDirection.LeftToRight, new Typeface("Arial"), 24, Brushes.DarkGray), new Point(8, 8));
        }
    }
    class RenderViewportPoint : RenderViewportBase
    {
        protected override void OnRenderToy(DrawingContext drawingContext)
        {
            drawingContext.DrawImage(ImageHelp.CreateImage(Render.Point, Scene, MVP, (int)Math.Ceiling(ActualWidth), (int)Math.Ceiling(ActualHeight)), new Rect(0, 0, ActualWidth, ActualHeight));
        }
    }
    class RenderViewportWireframeGDI : RenderViewportBase
    {
        protected override void OnRenderToy(DrawingContext drawingContext)
        {
            AbstractLineRenderer.DrawWireframe(Scene, MVP, new WireframeGDI(drawingContext, (int)Math.Ceiling(ActualWidth), (int)Math.Ceiling(ActualHeight)), ActualWidth, ActualHeight);
        }
    }
    class RenderViewportWireframeWPF : RenderViewportBase
    {
        protected override void OnRenderToy(DrawingContext drawingContext)
        {
            AbstractLineRenderer.DrawWireframe(Scene, MVP, new WireframeWPF(drawingContext), ActualWidth, ActualHeight);
        }
    }
    class RenderViewportRaster : RenderViewportBase
    {
        protected override void OnRenderToy(DrawingContext drawingContext)
        {
            drawingContext.DrawImage(ImageHelp.CreateImage(Render.Raster, Scene, MVP, ReduceQuality ? 128 : 512, ReduceQuality ? 128 : 512), new Rect(0, 0, ActualWidth, ActualHeight));
        }
    }
    class RenderViewportRasterD3D : RenderViewportBase
    {
        protected override void OnRenderToy(DrawingContext drawingContext)
        {
            drawingContext.DrawImage(ImageHelp.CreateImage(Render.RasterD3D9, Scene, MVP, (int)Math.Ceiling(ActualWidth), (int)Math.Ceiling(ActualHeight)), new Rect(0, 0, ActualWidth, ActualHeight));
        }
    }
}