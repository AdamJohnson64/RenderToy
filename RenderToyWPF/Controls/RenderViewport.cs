﻿////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System;
using System.Globalization;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;

namespace RenderToy
{
    public abstract class RenderViewportBase : FrameworkElement
    {
        public Scene Scene = Scene.Default;
        public RenderViewportBase()
        {
            ReduceQuality_Init();
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
                return View * Projection * CameraPerspective.AspectCorrectFit(ActualWidth, ActualHeight);
            }
        }
        TransformPosQuat Camera = new TransformPosQuat { Position = new Point3D(0, 2, -5) };
        CameraPerspective CameraMat = new CameraPerspective();
        #endregion
        #region - Section : Input Handling -
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
            // Detect modifier keys.
            bool isPressedLeftControl = Keyboard.IsKeyDown(Key.LeftCtrl);
            bool isPressedLeftShift = Keyboard.IsKeyDown(Key.LeftShift);
            // Process camera motion with modifier keys.
            if (isPressedLeftShift && isPressedLeftControl)
            {
                // Truck Mode (CTRL + SHIFT).
                Camera.TranslatePost(new Vector3D(0, 0, dy * -0.05));
            }
            else if (!isPressedLeftShift && isPressedLeftControl)
            {
                // Rotate Mode (CTRL Only)
                Camera.RotatePre(new Quaternion(new Vector3D(0, 1, 0), dx * 0.05));
                Camera.RotatePost(new Quaternion(new Vector3D(1, 0, 0), dy * 0.05));
            }
            else if (!isPressedLeftShift && !isPressedLeftControl)
            {
                // Translation Mode (no modifier keys).
                Camera.TranslatePost(new Vector3D(dx * -0.05, dy * 0.05, 0));
            }
            // Update the view.
            InvalidateVisual();
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
        static RoutedUICommand CommandRenderPoint = new RoutedUICommand("Point", "CommandRenderPoint", typeof(RenderViewport));
        static RoutedUICommand CommandRenderWireframe = new RoutedUICommand("Wireframe", "CommandRenderWireframe", typeof(RenderViewport));
        static RoutedUICommand CommandRenderRaster = new RoutedUICommand("Raster", "CommandRenderRaster", typeof(RenderViewport));
        static RoutedUICommand CommandRenderRasterD3D9 = new RoutedUICommand("Raster (D3D9)", "CommandRenderRasterD3D9", typeof(RenderViewport));
        static RoutedUICommand CommandRenderRaycastCPU = new RoutedUICommand("Raycast (CPU)", "CommandRenderRaycastCPU", typeof(RenderViewport));
        static RoutedUICommand CommandRenderRaycastNormalsCPU = new RoutedUICommand("Raycast Normals (CPU)", "CommandRenderRaycastNormalsCPU", typeof(RenderViewport));
        static RoutedUICommand CommandRenderRaycastTangentsCPU = new RoutedUICommand("Raycast Tangents (CPU)", "CommandRenderRaycastTangentsCPU", typeof(RenderViewport));
        static RoutedUICommand CommandRenderRaycastBitangentsCPU = new RoutedUICommand("Raycast Bitangents (CPU)", "CommandRenderRaycastBitangentsCPU", typeof(RenderViewport));
        static RoutedUICommand CommandRenderRaytraceCPUF32 = new RoutedUICommand("Raytrace (CPU/F32)", "CommandRenderRaytraceCPUF32", typeof(RenderViewport));
        static RoutedUICommand CommandRenderRaytraceCPUF64 = new RoutedUICommand("Raytrace (CPU/F64)", "CommandRenderRaytraceCPUF64", typeof(RenderViewport));
        static RoutedUICommand CommandRenderRaycastCUDA = new RoutedUICommand("Raycast (CUDA)", "CommandRenderRaycastCUDA", typeof(RenderViewport));
        static RoutedUICommand CommandRenderRaycastNormalsCUDA = new RoutedUICommand("Raycast Normals (CUDA)", "CommandRenderRaycastNormalsCUDA", typeof(RenderViewport));
        static RoutedUICommand CommandRenderRaycastTangentsCUDA = new RoutedUICommand("Raycast Tangents (CUDA)", "CommandRenderRaycastTangentsCUDA", typeof(RenderViewport));
        static RoutedUICommand CommandRenderRaycastBitangentsCUDA = new RoutedUICommand("Raycast Bitangents (CUDA)", "CommandRenderRaycastBitangentsCUDA", typeof(RenderViewport));
        static RoutedUICommand CommandRenderRaytraceCUDAF32 = new RoutedUICommand("Raytrace (CUDA/F32)", "CommandRenderRaytraceCUDAF32", typeof(RenderViewport));
        static RoutedUICommand CommandRenderRaytraceCUDAF64 = new RoutedUICommand("Raytrace (CUDA/F64)", "CommandRenderRaytraceCUDAF64", typeof(RenderViewport));
        static RoutedUICommand CommandRenderPreviewsToggle = new RoutedUICommand("Toggle Render Previews", "CommandRenderPreviewsToggle", typeof(RenderViewport));
        static RoutedUICommand CommandRenderWireframeToggle = new RoutedUICommand("Toggle Render Wireframe", "CommandRenderWireframeToggle", typeof(RenderViewport));
        public RenderViewport()
        {
            // Define command bindings to select display modes.
            CommandBindings.Add(new CommandBinding(CommandRenderPoint, (s, e) => { renderMode = RenderMode.Point; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderWireframe, (s, e) => { renderMode = RenderMode.Wireframe; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderRaster, (s, e) => { renderMode = RenderMode.Raster; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderRasterD3D9, (s, e) => { renderMode = RenderMode.Direct3D9; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderRaycastCPU, (s, e) => { renderMode = RenderMode.RaycastCPU; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderRaycastNormalsCPU, (s, e) => { renderMode = RenderMode.RaycastNormalsCPU; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderRaycastTangentsCPU, (s, e) => { renderMode = RenderMode.RaycastTangentsCPU; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderRaycastBitangentsCPU, (s, e) => { renderMode = RenderMode.RaycastBitangentsCPU; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderRaytraceCPUF32, (s, e) => { renderMode = RenderMode.RaytraceCPUF32; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderRaytraceCPUF64, (s, e) => { renderMode = RenderMode.RaytraceCPUF64; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderRaycastCUDA, (s, e) => { renderMode = RenderMode.RaycastCUDA; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderRaycastNormalsCUDA, (s, e) => { renderMode = RenderMode.RaycastNormalsCUDA; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderRaycastTangentsCUDA, (s, e) => { renderMode = RenderMode.RaycastTangentsCUDA; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderRaycastBitangentsCUDA, (s, e) => { renderMode = RenderMode.RaycastBitangentsCUDA; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderRaytraceCUDAF32, (s, e) => { renderMode = RenderMode.RaytraceCUDAF32; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderRaytraceCUDAF64, (s, e) => { renderMode = RenderMode.RaytraceCUDAF64; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderPreviewsToggle, (s, e) => { renderPreviews = !renderPreviews; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderWireframeToggle, (s, e) => { renderWireframe = !renderWireframe; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            // Define input bindings for common display modes.
            InputBindings.Add(new KeyBinding(CommandRenderPoint, Key.D1, ModifierKeys.Control));
            InputBindings.Add(new KeyBinding(CommandRenderWireframe, Key.D2, ModifierKeys.Control));
            InputBindings.Add(new KeyBinding(CommandRenderRaster, Key.D3, ModifierKeys.Control));
            InputBindings.Add(new KeyBinding(CommandRenderPreviewsToggle, Key.P, ModifierKeys.Control));
            InputBindings.Add(new KeyBinding(CommandRenderWireframeToggle, Key.W, ModifierKeys.Control));
            // Attach a context menu.
            {
                var menu = new ContextMenu();
                var submenu = new MenuItem { Header = "Render Mode" };
                submenu.Items.Add(new MenuItem { Command = CommandRenderPoint });
                submenu.Items.Add(new MenuItem { Command = CommandRenderWireframe });
                submenu.Items.Add(new MenuItem { Command = CommandRenderRaster });
                submenu.Items.Add(new MenuItem { Command = CommandRenderRasterD3D9 });
                submenu.Items.Add(new MenuItem { Command = CommandRenderRaycastCPU });
                submenu.Items.Add(new MenuItem { Command = CommandRenderRaycastNormalsCPU });
                submenu.Items.Add(new MenuItem { Command = CommandRenderRaycastTangentsCPU });
                submenu.Items.Add(new MenuItem { Command = CommandRenderRaycastBitangentsCPU });
                submenu.Items.Add(new MenuItem { Command = CommandRenderRaytraceCPUF32 });
                submenu.Items.Add(new MenuItem { Command = CommandRenderRaytraceCPUF64 });
                submenu.Items.Add(new MenuItem { Command = CommandRenderRaycastCUDA });
                submenu.Items.Add(new MenuItem { Command = CommandRenderRaycastNormalsCUDA });
                submenu.Items.Add(new MenuItem { Command = CommandRenderRaycastTangentsCUDA });
                submenu.Items.Add(new MenuItem { Command = CommandRenderRaycastBitangentsCUDA });
                submenu.Items.Add(new MenuItem { Command = CommandRenderRaytraceCUDAF32 });
                submenu.Items.Add(new MenuItem { Command = CommandRenderRaytraceCUDAF64 });
                menu.Items.Add(submenu);
                this.ContextMenu = menu;
            }
            Focusable = true;
        }
        enum RenderMode { Point, Wireframe, Raster, RaycastCPU, RaycastNormalsCPU, RaycastTangentsCPU, RaycastBitangentsCPU, RaytraceCPUF32, RaytraceCPUF64, RaycastCUDA, RaycastNormalsCUDA, RaycastTangentsCUDA, RaycastBitangentsCUDA, RaytraceCUDAF32, RaytraceCUDAF64, Direct3D9 }
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
                    drawingContext.DrawImage(ImageHelp.CreateImage(RenderCS.Point, Scene, MVP, ReduceQuality ? 256 : (int)Math.Ceiling(ActualWidth), ReduceQuality ? 256 : (int)Math.Ceiling(ActualHeight)), new Rect(0, 0, ActualWidth, ActualHeight));
                    break;
                case RenderMode.Wireframe:
                    drawingContext.DrawImage(ImageHelp.CreateImage(RenderCS.Wireframe, Scene, MVP, ReduceQuality ? 256 : (int)Math.Ceiling(ActualWidth), ReduceQuality ? 256 : (int)Math.Ceiling(ActualHeight)), new Rect(0, 0, ActualWidth, ActualHeight));
                    break;
                case RenderMode.Raster:
                    drawingContext.DrawImage(ImageHelp.CreateImage(RenderCS.Raster, Scene, MVP, ReduceQuality ? 256 : (int)Math.Ceiling(ActualWidth), ReduceQuality ? 256 : (int)Math.Ceiling(ActualHeight)), new Rect(0, 0, ActualWidth, ActualHeight));
                    break;
                case RenderMode.RaycastCPU:
                    drawingContext.DrawImage(ImageHelp.CreateImage(Render.RaycastCPU, Scene, MVP, (int)Math.Ceiling(ActualWidth) / (ReduceQuality ? 2 : 1), (int)Math.Ceiling(ActualHeight) / (ReduceQuality ? 2 : 1)), new Rect(0, 0, ActualWidth, ActualHeight));
                    break;
                case RenderMode.RaycastNormalsCPU:
                    drawingContext.DrawImage(ImageHelp.CreateImage(Render.RaycastNormalsCPU, Scene, MVP, (int)Math.Ceiling(ActualWidth) / (ReduceQuality ? 2 : 1), (int)Math.Ceiling(ActualHeight) / (ReduceQuality ? 2 : 1)), new Rect(0, 0, ActualWidth, ActualHeight));
                    break;
                case RenderMode.RaycastTangentsCPU:
                    drawingContext.DrawImage(ImageHelp.CreateImage(Render.RaycastTangentsCPU, Scene, MVP, (int)Math.Ceiling(ActualWidth) / (ReduceQuality ? 2 : 1), (int)Math.Ceiling(ActualHeight) / (ReduceQuality ? 2 : 1)), new Rect(0, 0, ActualWidth, ActualHeight));
                    break;
                case RenderMode.RaycastBitangentsCPU:
                    drawingContext.DrawImage(ImageHelp.CreateImage(Render.RaycastBitangentsCPU, Scene, MVP, (int)Math.Ceiling(ActualWidth) / (ReduceQuality ? 2 : 1), (int)Math.Ceiling(ActualHeight) / (ReduceQuality ? 2 : 1)), new Rect(0, 0, ActualWidth, ActualHeight));
                    break;
                case RenderMode.RaytraceCPUF32:
                    drawingContext.DrawImage(ImageHelp.CreateImage(Render.RaytraceCPUF32, Scene, MVP, (int)Math.Ceiling(ActualWidth) / (ReduceQuality ? 2 : 1), (int)Math.Ceiling(ActualHeight) / (ReduceQuality ? 2 : 1)), new Rect(0, 0, ActualWidth, ActualHeight));
                    break;
                case RenderMode.RaytraceCPUF64:
                    drawingContext.DrawImage(ImageHelp.CreateImage(Render.RaytraceCPUF64, Scene, MVP, (int)Math.Ceiling(ActualWidth) / (ReduceQuality ? 2 : 1), (int)Math.Ceiling(ActualHeight) / (ReduceQuality ? 2 : 1)), new Rect(0, 0, ActualWidth, ActualHeight));
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
                case RenderMode.RaycastTangentsCUDA:
                    if (Render.CUDAAvailable())
                    {
                        drawingContext.DrawImage(ImageHelp.CreateImage(Render.RaycastTangentsCUDA, Scene, MVP, (int)Math.Ceiling(ActualWidth) / (ReduceQuality ? 2 : 1), (int)Math.Ceiling(ActualHeight) / (ReduceQuality ? 2 : 1)), new Rect(0, 0, ActualWidth, ActualHeight));
                    }
                    else
                    {
                        drawingContext.DrawText(new FormattedText("CUDA is not available or was not built for this configuration.", CultureInfo.InvariantCulture, FlowDirection.LeftToRight, new Typeface("Arial"), 24, Brushes.Red), new Point(8, ActualHeight / 2));
                    }
                    break;
                case RenderMode.RaycastBitangentsCUDA:
                    if (Render.CUDAAvailable())
                    {
                        drawingContext.DrawImage(ImageHelp.CreateImage(Render.RaycastBitangentsCUDA, Scene, MVP, (int)Math.Ceiling(ActualWidth) / (ReduceQuality ? 2 : 1), (int)Math.Ceiling(ActualHeight) / (ReduceQuality ? 2 : 1)), new Rect(0, 0, ActualWidth, ActualHeight));
                    }
                    else
                    {
                        drawingContext.DrawText(new FormattedText("CUDA is not available or was not built for this configuration.", CultureInfo.InvariantCulture, FlowDirection.LeftToRight, new Typeface("Arial"), 24, Brushes.Red), new Point(8, ActualHeight / 2));
                    }
                    break;
                case RenderMode.RaytraceCUDAF32:
                    if (Render.CUDAAvailable())
                    {
                        drawingContext.DrawImage(ImageHelp.CreateImage(Render.RaytraceCUDAF32, Scene, MVP, (int)Math.Ceiling(ActualWidth) / (ReduceQuality ? 2 : 1), (int)Math.Ceiling(ActualHeight) / (ReduceQuality ? 2 : 1)), new Rect(0, 0, ActualWidth, ActualHeight));
                    }
                    else
                    {
                        drawingContext.DrawText(new FormattedText("CUDA is not available or was not built for this configuration.", CultureInfo.InvariantCulture, FlowDirection.LeftToRight, new Typeface("Arial"), 24, Brushes.Red), new Point(8, ActualHeight / 2));
                    }
                    break;
                case RenderMode.RaytraceCUDAF64:
                    if (Render.CUDAAvailable())
                    {
                        drawingContext.DrawImage(ImageHelp.CreateImage(Render.RaytraceCUDAF64, Scene, MVP, (int)Math.Ceiling(ActualWidth) / (ReduceQuality ? 2 : 1), (int)Math.Ceiling(ActualHeight) / (ReduceQuality ? 2 : 1)), new Rect(0, 0, ActualWidth, ActualHeight));
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
                drawingContext.DrawImage(ImageHelp.CreateImage(RenderCS.Wireframe, Scene, MVP, ReduceQuality ? 256 : (int)Math.Ceiling(ActualWidth), ReduceQuality ? 256 : (int)Math.Ceiling(ActualHeight)), new Rect(0, 0, ActualWidth, ActualHeight));
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
                    var imagesource = ImageHelp.CreateImage(fillwith, Scene, View * Projection * CameraPerspective.AspectCorrectFit(image_w, image_h), render_width, render_height);
                    drawingContext.DrawImage(imagesource, new Rect(image_l, image_t, image_w, image_h));
                };
                drawpreview(RenderCS.Point, 0, ReduceQuality ? 32 : 64, ReduceQuality ? 32 : 64);
                drawpreview(RenderCS.Wireframe, 1, ReduceQuality ? 32 : 128, ReduceQuality ? 32 : 128);
                drawpreview(RenderCS.Raster, 2, ReduceQuality ? 32 : 128, ReduceQuality ? 32 : 128);
            }
            drawingContext.DrawText(new FormattedText(renderMode.ToString(), CultureInfo.InvariantCulture, FlowDirection.LeftToRight, new Typeface("Arial"), 24, Brushes.LightGray), new Point(10, 10));
            drawingContext.DrawText(new FormattedText(renderMode.ToString(), CultureInfo.InvariantCulture, FlowDirection.LeftToRight, new Typeface("Arial"), 24, Brushes.DarkGray), new Point(8, 8));
        }
    }
    class RenderViewportPoint : RenderViewportBase
    {
        protected override void OnRenderToy(DrawingContext drawingContext)
        {
            drawingContext.DrawImage(ImageHelp.CreateImage(RenderCS.Point, Scene, MVP, (int)Math.Ceiling(ActualWidth), (int)Math.Ceiling(ActualHeight)), new Rect(0, 0, ActualWidth, ActualHeight));
        }
    }
    class RenderViewportRaster : RenderViewportBase
    {
        protected override void OnRenderToy(DrawingContext drawingContext)
        {
            drawingContext.DrawImage(ImageHelp.CreateImage(RenderCS.Raster, Scene, MVP, ReduceQuality ? 128 : 512, ReduceQuality ? 128 : 512), new Rect(0, 0, ActualWidth, ActualHeight));
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