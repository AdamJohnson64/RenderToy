////////////////////////////////////////////////////////////////////////////////
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
        static RoutedUICommand CommandRenderPoint = new RoutedUICommand("Point (CPU/F64)", "CommandRenderPoint", typeof(RenderViewport));
        static RoutedUICommand CommandRenderWireframe = new RoutedUICommand("Wireframe (CPU/F64)", "CommandRenderWireframe", typeof(RenderViewport));
        static RoutedUICommand CommandRenderRaster = new RoutedUICommand("Raster (CPU/F64)", "CommandRenderRaster", typeof(RenderViewport));
        static RoutedUICommand CommandRenderRasterD3D9 = new RoutedUICommand("Raster (D3D9/F32)", "CommandRenderRasterD3D9", typeof(RenderViewport));
        static RoutedUICommand CommandRenderRaycastCPUF32 = new RoutedUICommand("Raycast (CPU/F32)", "CommandRenderRaycastCPUF32", typeof(RenderViewport));
        static RoutedUICommand CommandRenderRaycastCPUF64 = new RoutedUICommand("Raycast (CPU/F64)", "CommandRenderRaycastCPUF64", typeof(RenderViewport));
        static RoutedUICommand CommandRenderRaycastCUDAF32 = new RoutedUICommand("Raycast (CUDA/F32)", "CommandRenderRaycastCUDAF32", typeof(RenderViewport));
        static RoutedUICommand CommandRenderRaycastCUDAF64 = new RoutedUICommand("Raycast (CUDA/F64)", "CommandRenderRaycastCUDAF64", typeof(RenderViewport));
        static RoutedUICommand CommandRenderRaycastNormalsCPUF32 = new RoutedUICommand("Raycast Normals (CPU/F32)", "CommandRenderRaycastNormalsCPUF32", typeof(RenderViewport));
        static RoutedUICommand CommandRenderRaycastNormalsCPUF64 = new RoutedUICommand("Raycast Normals (CPU/F64)", "CommandRenderRaycastNormalsCPUF64", typeof(RenderViewport));
        static RoutedUICommand CommandRenderRaycastNormalsCUDAF32 = new RoutedUICommand("Raycast Normals (CUDA/F32)", "CommandRenderRaycastNormalsCUDAF32", typeof(RenderViewport));
        static RoutedUICommand CommandRenderRaycastNormalsCUDAF64 = new RoutedUICommand("Raycast Normals (CUDA/F64)", "CommandRenderRaycastNormalsCUDAF64", typeof(RenderViewport));
        static RoutedUICommand CommandRenderRaycastTangentsCPUF32 = new RoutedUICommand("Raycast Tangents (CPU/F32)", "CommandRenderRaycastTangentsCPUF32", typeof(RenderViewport));
        static RoutedUICommand CommandRenderRaycastTangentsCPUF64 = new RoutedUICommand("Raycast Tangents (CPU/F64)", "CommandRenderRaycastTangentsCPUF64", typeof(RenderViewport));
        static RoutedUICommand CommandRenderRaycastTangentsCUDAF32 = new RoutedUICommand("Raycast Tangents (CUDA/F32)", "CommandRenderRaycastTangentsCUDAF32", typeof(RenderViewport));
        static RoutedUICommand CommandRenderRaycastTangentsCUDAF64 = new RoutedUICommand("Raycast Tangents (CUDA/F64)", "CommandRenderRaycastTangentsCUDAF64", typeof(RenderViewport));
        static RoutedUICommand CommandRenderRaycastBitangentsCPUF32 = new RoutedUICommand("Raycast Bitangents (CPU/F32)", "CommandRenderRaycastBitangentsCPUF32", typeof(RenderViewport));
        static RoutedUICommand CommandRenderRaycastBitangentsCPUF64 = new RoutedUICommand("Raycast Bitangents (CPU/F64)", "CommandRenderRaycastBitangentsCPUF64", typeof(RenderViewport));
        static RoutedUICommand CommandRenderRaycastBitangentsCUDAF32 = new RoutedUICommand("Raycast Bitangents (CUDA/F32)", "CommandRenderRaycastBitangentsCUDAF32", typeof(RenderViewport));
        static RoutedUICommand CommandRenderRaycastBitangentsCUDAF64 = new RoutedUICommand("Raycast Bitangents (CUDA/F64)", "CommandRenderRaycastBitangentsCUDAF64", typeof(RenderViewport));
        static RoutedUICommand CommandRenderRaytraceCPUF32 = new RoutedUICommand("Raytrace (CPU/F32)", "CommandRenderRaytraceCPUF32", typeof(RenderViewport));
        static RoutedUICommand CommandRenderRaytraceCPUF64 = new RoutedUICommand("Raytrace (CPU/F64)", "CommandRenderRaytraceCPUF64", typeof(RenderViewport));
        static RoutedUICommand CommandRenderRaytraceCUDAF32 = new RoutedUICommand("Raytrace (CUDA/F32)", "CommandRenderRaytraceCUDAF32", typeof(RenderViewport));
        static RoutedUICommand CommandRenderRaytraceCUDAF64 = new RoutedUICommand("Raytrace (CUDA/F64)", "CommandRenderRaytraceCUDAF64", typeof(RenderViewport));
        static RoutedUICommand CommandRenderAmbientOcclusionCUDAF32 = new RoutedUICommand("Ambient Occlusion (CUDA/F32)", "CommandRenderAmbientOcclusionCUDAF32", typeof(RenderViewport));
        static RoutedUICommand CommandRenderPreviewsToggle = new RoutedUICommand("Toggle Render Previews", "CommandRenderPreviewsToggle", typeof(RenderViewport));
        static RoutedUICommand CommandRenderWireframeToggle = new RoutedUICommand("Toggle Render Wireframe", "CommandRenderWireframeToggle", typeof(RenderViewport));
        public RenderViewport()
        {
            // Define command bindings to select display modes.
            CommandBindings.Add(new CommandBinding(CommandRenderPoint, (s, e) => { renderMode = RenderMode.Point; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderWireframe, (s, e) => { renderMode = RenderMode.Wireframe; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderRaster, (s, e) => { renderMode = RenderMode.Raster; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderRasterD3D9, (s, e) => { renderMode = RenderMode.Direct3D9; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderRaycastCPUF32, (s, e) => { renderMode = RenderMode.RaycastCPUF32; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderRaycastCPUF64, (s, e) => { renderMode = RenderMode.RaycastCPUF64; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderRaycastCUDAF32, (s, e) => { renderMode = RenderMode.RaycastCUDAF32; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderRaycastCUDAF64, (s, e) => { renderMode = RenderMode.RaycastCUDAF64; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderRaycastNormalsCPUF32, (s, e) => { renderMode = RenderMode.RaycastNormalsCPUF32; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderRaycastNormalsCPUF64, (s, e) => { renderMode = RenderMode.RaycastNormalsCPUF64; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderRaycastNormalsCUDAF32, (s, e) => { renderMode = RenderMode.RaycastNormalsCUDAF32; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderRaycastNormalsCUDAF64, (s, e) => { renderMode = RenderMode.RaycastNormalsCUDAF64; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderRaycastTangentsCPUF32, (s, e) => { renderMode = RenderMode.RaycastTangentsCPUF32; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderRaycastTangentsCPUF64, (s, e) => { renderMode = RenderMode.RaycastTangentsCPUF64; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderRaycastTangentsCUDAF32, (s, e) => { renderMode = RenderMode.RaycastTangentsCUDAF32; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderRaycastTangentsCUDAF64, (s, e) => { renderMode = RenderMode.RaycastTangentsCUDAF64; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderRaycastBitangentsCPUF32, (s, e) => { renderMode = RenderMode.RaycastBitangentsCPUF32; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderRaycastBitangentsCPUF64, (s, e) => { renderMode = RenderMode.RaycastBitangentsCPUF64; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderRaycastBitangentsCUDAF32, (s, e) => { renderMode = RenderMode.RaycastBitangentsCUDAF32; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderRaycastBitangentsCUDAF64, (s, e) => { renderMode = RenderMode.RaycastBitangentsCUDAF64; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderRaytraceCPUF32, (s, e) => { renderMode = RenderMode.RaytraceCPUF32; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderRaytraceCPUF64, (s, e) => { renderMode = RenderMode.RaytraceCPUF64; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderRaytraceCUDAF32, (s, e) => { renderMode = RenderMode.RaytraceCUDAF32; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderRaytraceCUDAF64, (s, e) => { renderMode = RenderMode.RaytraceCUDAF64; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderAmbientOcclusionCUDAF32, (s, e) => { renderMode = RenderMode.AmbientOcclusionCUDAF32; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
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
                menu.Items.Add(new MenuItem { Command = CommandRenderPoint });
                menu.Items.Add(new MenuItem { Command = CommandRenderWireframe });
                {
                    var submenu = new MenuItem { Header = "Raster" };
                    submenu.Items.Add(new MenuItem { Command = CommandRenderRaster });
                    submenu.Items.Add(new MenuItem { Command = CommandRenderRasterD3D9 });
                    menu.Items.Add(submenu);
                }
                {
                    var submenu = new MenuItem { Header = "Raycast" };
                    submenu.Items.Add(new MenuItem { Command = CommandRenderRaycastCPUF32 });
                    submenu.Items.Add(new MenuItem { Command = CommandRenderRaycastCPUF64 });
                    submenu.Items.Add(new MenuItem { Command = CommandRenderRaycastCUDAF32 });
                    submenu.Items.Add(new MenuItem { Command = CommandRenderRaycastCUDAF64 });
                    menu.Items.Add(submenu);
                }
                {
                    var submenu = new MenuItem { Header = "Raycast Normals" };
                    submenu.Items.Add(new MenuItem { Command = CommandRenderRaycastNormalsCPUF32 });
                    submenu.Items.Add(new MenuItem { Command = CommandRenderRaycastNormalsCPUF64 });
                    submenu.Items.Add(new MenuItem { Command = CommandRenderRaycastNormalsCUDAF32 });
                    submenu.Items.Add(new MenuItem { Command = CommandRenderRaycastNormalsCUDAF64 });
                    menu.Items.Add(submenu);
                }
                {
                    var submenu = new MenuItem { Header = "Raycast Tangents" };
                    submenu.Items.Add(new MenuItem { Command = CommandRenderRaycastTangentsCPUF32 });
                    submenu.Items.Add(new MenuItem { Command = CommandRenderRaycastTangentsCPUF64 });
                    submenu.Items.Add(new MenuItem { Command = CommandRenderRaycastTangentsCUDAF32 });
                    submenu.Items.Add(new MenuItem { Command = CommandRenderRaycastTangentsCUDAF64 });
                    menu.Items.Add(submenu);
                }
                {
                    var submenu = new MenuItem { Header = "Raycast Bitangents" };
                    submenu.Items.Add(new MenuItem { Command = CommandRenderRaycastBitangentsCPUF32 });
                    submenu.Items.Add(new MenuItem { Command = CommandRenderRaycastBitangentsCPUF64 });
                    submenu.Items.Add(new MenuItem { Command = CommandRenderRaycastBitangentsCUDAF32 });
                    submenu.Items.Add(new MenuItem { Command = CommandRenderRaycastBitangentsCUDAF64 });
                    menu.Items.Add(submenu);
                }
                {
                    var submenu = new MenuItem { Header = "Raytrace" };
                    submenu.Items.Add(new MenuItem { Command = CommandRenderRaytraceCPUF32 });
                    submenu.Items.Add(new MenuItem { Command = CommandRenderRaytraceCPUF64 });
                    submenu.Items.Add(new MenuItem { Command = CommandRenderRaytraceCUDAF32 });
                    submenu.Items.Add(new MenuItem { Command = CommandRenderRaytraceCUDAF64 });
                    menu.Items.Add(submenu);
                }
                {
                    var submenu = new MenuItem { Header = "Ambient Occlusion" };
                    submenu.Items.Add(new MenuItem { Command = CommandRenderAmbientOcclusionCUDAF32 });
                    menu.Items.Add(submenu);
                }
                this.ContextMenu = menu;
            }
            Focusable = true;
        }
        enum RenderMode { Point, Wireframe, Raster, RaycastCPUF32, RaycastCPUF64, RaycastNormalsCPUF32, RaycastNormalsCPUF64, RaycastTangentsCPUF32, RaycastTangentsCPUF64, RaycastBitangentsCPUF32, RaycastBitangentsCPUF64, RaytraceCPUF32, RaytraceCPUF64, RaycastCUDAF32, RaycastCUDAF64, RaycastNormalsCUDAF32, RaycastNormalsCUDAF64, RaycastTangentsCUDAF32, RaycastTangentsCUDAF64, RaycastBitangentsCUDAF32, RaycastBitangentsCUDAF64, RaytraceCUDAF32, RaytraceCUDAF64, AmbientOcclusionCUDAF32, Direct3D9 }
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
                case RenderMode.RaycastCPUF32:
                    drawingContext.DrawImage(ImageHelp.CreateImage(Render.RaycastCPUF32, Scene, MVP, (int)Math.Ceiling(ActualWidth) / (ReduceQuality ? 2 : 1), (int)Math.Ceiling(ActualHeight) / (ReduceQuality ? 2 : 1)), new Rect(0, 0, ActualWidth, ActualHeight));
                    break;
                case RenderMode.RaycastCPUF64:
                    drawingContext.DrawImage(ImageHelp.CreateImage(Render.RaycastCPUF64, Scene, MVP, (int)Math.Ceiling(ActualWidth) / (ReduceQuality ? 2 : 1), (int)Math.Ceiling(ActualHeight) / (ReduceQuality ? 2 : 1)), new Rect(0, 0, ActualWidth, ActualHeight));
                    break;
                case RenderMode.RaycastCUDAF32:
                    if (Render.CUDAAvailable())
                    {
                        drawingContext.DrawImage(ImageHelp.CreateImage(Render.RaycastCUDAF32, Scene, MVP, (int)Math.Ceiling(ActualWidth) / (ReduceQuality ? 2 : 1), (int)Math.Ceiling(ActualHeight) / (ReduceQuality ? 2 : 1)), new Rect(0, 0, ActualWidth, ActualHeight));
                    }
                    else
                    {
                        drawingContext.DrawText(new FormattedText("CUDA is not available or was not built for this configuration.", CultureInfo.InvariantCulture, FlowDirection.LeftToRight, new Typeface("Arial"), 24, Brushes.Red), new Point(8, ActualHeight / 2));
                    }
                    break;
                case RenderMode.RaycastCUDAF64:
                    if (Render.CUDAAvailable())
                    {
                        drawingContext.DrawImage(ImageHelp.CreateImage(Render.RaycastCUDAF64, Scene, MVP, (int)Math.Ceiling(ActualWidth) / (ReduceQuality ? 2 : 1), (int)Math.Ceiling(ActualHeight) / (ReduceQuality ? 2 : 1)), new Rect(0, 0, ActualWidth, ActualHeight));
                    }
                    else
                    {
                        drawingContext.DrawText(new FormattedText("CUDA is not available or was not built for this configuration.", CultureInfo.InvariantCulture, FlowDirection.LeftToRight, new Typeface("Arial"), 24, Brushes.Red), new Point(8, ActualHeight / 2));
                    }
                    break;
                case RenderMode.RaycastNormalsCPUF32:
                    drawingContext.DrawImage(ImageHelp.CreateImage(Render.RaycastNormalsCPUF32, Scene, MVP, (int)Math.Ceiling(ActualWidth) / (ReduceQuality ? 2 : 1), (int)Math.Ceiling(ActualHeight) / (ReduceQuality ? 2 : 1)), new Rect(0, 0, ActualWidth, ActualHeight));
                    break;
                case RenderMode.RaycastNormalsCPUF64:
                    drawingContext.DrawImage(ImageHelp.CreateImage(Render.RaycastNormalsCPUF64, Scene, MVP, (int)Math.Ceiling(ActualWidth) / (ReduceQuality ? 2 : 1), (int)Math.Ceiling(ActualHeight) / (ReduceQuality ? 2 : 1)), new Rect(0, 0, ActualWidth, ActualHeight));
                    break;
                case RenderMode.RaycastNormalsCUDAF32:
                    if (Render.CUDAAvailable())
                    {
                        drawingContext.DrawImage(ImageHelp.CreateImage(Render.RaycastNormalsCUDAF32, Scene, MVP, (int)Math.Ceiling(ActualWidth) / (ReduceQuality ? 2 : 1), (int)Math.Ceiling(ActualHeight) / (ReduceQuality ? 2 : 1)), new Rect(0, 0, ActualWidth, ActualHeight));
                    }
                    else
                    {
                        drawingContext.DrawText(new FormattedText("CUDA is not available or was not built for this configuration.", CultureInfo.InvariantCulture, FlowDirection.LeftToRight, new Typeface("Arial"), 24, Brushes.Red), new Point(8, ActualHeight / 2));
                    }
                    break;
                case RenderMode.RaycastNormalsCUDAF64:
                    if (Render.CUDAAvailable())
                    {
                        drawingContext.DrawImage(ImageHelp.CreateImage(Render.RaycastNormalsCUDAF64, Scene, MVP, (int)Math.Ceiling(ActualWidth) / (ReduceQuality ? 2 : 1), (int)Math.Ceiling(ActualHeight) / (ReduceQuality ? 2 : 1)), new Rect(0, 0, ActualWidth, ActualHeight));
                    }
                    else
                    {
                        drawingContext.DrawText(new FormattedText("CUDA is not available or was not built for this configuration.", CultureInfo.InvariantCulture, FlowDirection.LeftToRight, new Typeface("Arial"), 24, Brushes.Red), new Point(8, ActualHeight / 2));
                    }
                    break;
                case RenderMode.RaycastTangentsCPUF32:
                    drawingContext.DrawImage(ImageHelp.CreateImage(Render.RaycastTangentsCPUF32, Scene, MVP, (int)Math.Ceiling(ActualWidth) / (ReduceQuality ? 2 : 1), (int)Math.Ceiling(ActualHeight) / (ReduceQuality ? 2 : 1)), new Rect(0, 0, ActualWidth, ActualHeight));
                    break;
                case RenderMode.RaycastTangentsCPUF64:
                    drawingContext.DrawImage(ImageHelp.CreateImage(Render.RaycastTangentsCPUF64, Scene, MVP, (int)Math.Ceiling(ActualWidth) / (ReduceQuality ? 2 : 1), (int)Math.Ceiling(ActualHeight) / (ReduceQuality ? 2 : 1)), new Rect(0, 0, ActualWidth, ActualHeight));
                    break;
                case RenderMode.RaycastTangentsCUDAF32:
                    if (Render.CUDAAvailable())
                    {
                        drawingContext.DrawImage(ImageHelp.CreateImage(Render.RaycastTangentsCUDAF32, Scene, MVP, (int)Math.Ceiling(ActualWidth) / (ReduceQuality ? 2 : 1), (int)Math.Ceiling(ActualHeight) / (ReduceQuality ? 2 : 1)), new Rect(0, 0, ActualWidth, ActualHeight));
                    }
                    else
                    {
                        drawingContext.DrawText(new FormattedText("CUDA is not available or was not built for this configuration.", CultureInfo.InvariantCulture, FlowDirection.LeftToRight, new Typeface("Arial"), 24, Brushes.Red), new Point(8, ActualHeight / 2));
                    }
                    break;
                case RenderMode.RaycastTangentsCUDAF64:
                    if (Render.CUDAAvailable())
                    {
                        drawingContext.DrawImage(ImageHelp.CreateImage(Render.RaycastTangentsCUDAF64, Scene, MVP, (int)Math.Ceiling(ActualWidth) / (ReduceQuality ? 2 : 1), (int)Math.Ceiling(ActualHeight) / (ReduceQuality ? 2 : 1)), new Rect(0, 0, ActualWidth, ActualHeight));
                    }
                    else
                    {
                        drawingContext.DrawText(new FormattedText("CUDA is not available or was not built for this configuration.", CultureInfo.InvariantCulture, FlowDirection.LeftToRight, new Typeface("Arial"), 24, Brushes.Red), new Point(8, ActualHeight / 2));
                    }
                    break;
                case RenderMode.RaycastBitangentsCPUF32:
                    drawingContext.DrawImage(ImageHelp.CreateImage(Render.RaycastBitangentsCPUF32, Scene, MVP, (int)Math.Ceiling(ActualWidth) / (ReduceQuality ? 2 : 1), (int)Math.Ceiling(ActualHeight) / (ReduceQuality ? 2 : 1)), new Rect(0, 0, ActualWidth, ActualHeight));
                    break;
                case RenderMode.RaycastBitangentsCPUF64:
                    drawingContext.DrawImage(ImageHelp.CreateImage(Render.RaycastBitangentsCPUF64, Scene, MVP, (int)Math.Ceiling(ActualWidth) / (ReduceQuality ? 2 : 1), (int)Math.Ceiling(ActualHeight) / (ReduceQuality ? 2 : 1)), new Rect(0, 0, ActualWidth, ActualHeight));
                    break;
                case RenderMode.RaycastBitangentsCUDAF32:
                    if (Render.CUDAAvailable())
                    {
                        drawingContext.DrawImage(ImageHelp.CreateImage(Render.RaycastBitangentsCUDAF32, Scene, MVP, (int)Math.Ceiling(ActualWidth) / (ReduceQuality ? 2 : 1), (int)Math.Ceiling(ActualHeight) / (ReduceQuality ? 2 : 1)), new Rect(0, 0, ActualWidth, ActualHeight));
                    }
                    else
                    {
                        drawingContext.DrawText(new FormattedText("CUDA is not available or was not built for this configuration.", CultureInfo.InvariantCulture, FlowDirection.LeftToRight, new Typeface("Arial"), 24, Brushes.Red), new Point(8, ActualHeight / 2));
                    }
                    break;
                case RenderMode.RaycastBitangentsCUDAF64:
                    if (Render.CUDAAvailable())
                    {
                        drawingContext.DrawImage(ImageHelp.CreateImage(Render.RaycastBitangentsCUDAF64, Scene, MVP, (int)Math.Ceiling(ActualWidth) / (ReduceQuality ? 2 : 1), (int)Math.Ceiling(ActualHeight) / (ReduceQuality ? 2 : 1)), new Rect(0, 0, ActualWidth, ActualHeight));
                    }
                    else
                    {
                        drawingContext.DrawText(new FormattedText("CUDA is not available or was not built for this configuration.", CultureInfo.InvariantCulture, FlowDirection.LeftToRight, new Typeface("Arial"), 24, Brushes.Red), new Point(8, ActualHeight / 2));
                    }
                    break;
                case RenderMode.RaytraceCPUF32:
                    drawingContext.DrawImage(ImageHelp.CreateImage(Render.RaytraceCPUF32, Scene, MVP, (int)Math.Ceiling(ActualWidth) / (ReduceQuality ? 2 : 1), (int)Math.Ceiling(ActualHeight) / (ReduceQuality ? 2 : 1)), new Rect(0, 0, ActualWidth, ActualHeight));
                    break;
                case RenderMode.RaytraceCPUF64:
                    drawingContext.DrawImage(ImageHelp.CreateImage(Render.RaytraceCPUF64, Scene, MVP, (int)Math.Ceiling(ActualWidth) / (ReduceQuality ? 2 : 1), (int)Math.Ceiling(ActualHeight) / (ReduceQuality ? 2 : 1)), new Rect(0, 0, ActualWidth, ActualHeight));
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
                case RenderMode.AmbientOcclusionCUDAF32:
                    if (Render.CUDAAvailable())
                    {
                        drawingContext.DrawImage(ImageHelp.CreateImage(Render.AmbientOcclusionCUDAF32, Scene, MVP, (int)Math.Ceiling(ActualWidth) / (ReduceQuality ? 2 : 1), (int)Math.Ceiling(ActualHeight) / (ReduceQuality ? 2 : 1)), new Rect(0, 0, ActualWidth, ActualHeight));
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