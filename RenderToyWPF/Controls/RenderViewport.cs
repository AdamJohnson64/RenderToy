﻿////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Runtime.InteropServices;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;

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
        static RoutedUICommand CommandRenderPreviewsToggle = new RoutedUICommand("Toggle Render Previews", "CommandRenderPreviewsToggle", typeof(RenderViewport));
        static RoutedUICommand CommandRenderWireframeToggle = new RoutedUICommand("Toggle Render Wireframe", "CommandRenderWireframeToggle", typeof(RenderViewport));
        public RenderViewport()
        {
            Focusable = true;
            CommandBindings.Add(new CommandBinding(CommandRenderPreviewsToggle, (s, e) => { renderPreviews = !renderPreviews; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            CommandBindings.Add(new CommandBinding(CommandRenderWireframeToggle, (s, e) => { renderWireframe = !renderWireframe; InvalidateVisual(); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            // Define input bindings for common display modes.
            InputBindings.Add(new KeyBinding(CommandRenderPreviewsToggle, Key.P, ModifierKeys.Control));
            InputBindings.Add(new KeyBinding(CommandRenderWireframeToggle, Key.W, ModifierKeys.Control));
            // Generate commands for render modes.
            foreach (var call in RenderCallCommands.Calls)
            {
                CommandBindings.Add(new CommandBinding(RenderCallCommands.Commands[call], (s, e) => { RenderMode = MultiPass.Create(call); e.Handled = true; }, (s, e) => { e.CanExecute = true; e.Handled = true; }));
            }
            // Generate context menu.
            var menu = new ContextMenu();
            foreach (var group in RenderCallCommands.Calls.GroupBy(x => RenderCall.GetDisplayNameBare(x.MethodInfo.Name)))
            {
                var menu_group = new MenuItem { Header = group.Key };
                foreach (var call in group)
                {
                    menu_group.Items.Add(new MenuItem { Command = RenderCallCommands.Commands[call] });
                }
                menu.Items.Add(menu_group);
            }
            this.ContextMenu = menu;
        }
        MultiPass RenderMode
        {
            set { renderMode = value; InvalidateVisual(); }
        }
        MultiPass renderMode = MultiPass.Create(RenderCallCommands.Calls[0]);
        bool renderPreviews = true;
        bool renderWireframe = false;
        protected override void OnMouseDown(MouseButtonEventArgs e)
        {
            base.OnMouseDown(e);
            Focus();
        }
        protected override void OnRenderToy(DrawingContext drawingContext)
        {
            if (renderMode != null)
            {
                int RENDER_WIDTH = (int)Math.Ceiling(ActualWidth) / (ReduceQuality ? 2 : 1);
                int RENDER_HEIGHT = (int)Math.Ceiling(ActualHeight) / (ReduceQuality ? 2 : 1);
                renderMode.SetScene(Scene);
                renderMode.SetCamera(MVP);
                renderMode.SetTarget(RENDER_WIDTH, RENDER_HEIGHT);
                WriteableBitmap bitmap = new WriteableBitmap(RENDER_WIDTH, RENDER_HEIGHT, 0, 0, PixelFormats.Bgra32, null);
                bitmap.Lock();
                renderMode.CopyTo(bitmap.BackBuffer, bitmap.PixelWidth, bitmap.PixelHeight, bitmap.BackBufferStride);
                bitmap.AddDirtyRect(new Int32Rect(0, 0, RENDER_WIDTH, RENDER_HEIGHT));
                bitmap.Unlock();
                drawingContext.DrawImage(bitmap, new Rect(0, 0, ActualWidth, ActualHeight));
            }
            if (renderWireframe)
            {
                drawingContext.PushOpacity(0.5);
                drawingContext.DrawImage(ImageHelp.CreateImage(RenderCS.WireframeCPUF64, Scene, MVP, ReduceQuality ? 256 : (int)Math.Ceiling(ActualWidth), ReduceQuality ? 256 : (int)Math.Ceiling(ActualHeight)), new Rect(0, 0, ActualWidth, ActualHeight));
                drawingContext.Pop();
            }
            if (renderPreviews)
            {
                Action<RenderCall.FillFunction, int, int, int> drawpreview = (fillwith, stacky, render_width, render_height) =>
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
                drawpreview(RenderCS.PointCPUF64, 0, ReduceQuality ? 32 : 64, ReduceQuality ? 32 : 64);
                drawpreview(RenderCS.WireframeCPUF64, 1, ReduceQuality ? 32 : 128, ReduceQuality ? 32 : 128);
                drawpreview(RenderCS.RasterCPUF64, 2, ReduceQuality ? 32 : 128, ReduceQuality ? 32 : 128);
            }
            if (renderMode != null)
            {
                drawingContext.DrawText(new FormattedText(renderMode.ToString(), CultureInfo.InvariantCulture, FlowDirection.LeftToRight, new Typeface("Arial"), 24, Brushes.LightGray), new Point(10, 10));
                drawingContext.DrawText(new FormattedText(renderMode.ToString(), CultureInfo.InvariantCulture, FlowDirection.LeftToRight, new Typeface("Arial"), 24, Brushes.DarkGray), new Point(8, 8));
            }
        }
    }
    class RenderCallCommands
    {
        static RenderCallCommands()
        {
            Calls = RenderCall.Generate(new[] { typeof(RenderCS), typeof(RenderD3D), typeof(RenderToyCLI), typeof(RenderToyInjections) }).ToArray();
            Commands = Calls.ToDictionary(x => x, y => new RoutedUICommand(RenderCall.GetDisplayNameFull(y.MethodInfo.Name), y.MethodInfo.Name, typeof(RenderCallCommands)));
        }
        public static readonly RenderCall[] Calls;
        public static readonly Dictionary<RenderCall, RoutedUICommand> Commands;
    }
    class RenderToyInjections
    {
        public static void AOCHaltonBiasedMPCUDAF32(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int render_width, int render_height, int bitmap_stride)
        {
            var hemisamples = MathHelp.HemiHaltonCosineBias(8192).ToArray();
            RenderToyCLI.AmbientOcclusionMPCUDAF32(SceneFormatter.CreateFlatMemoryF32(scene), SceneFormatter.CreateFlatMemoryF32(MathHelp.Invert(mvp)), bitmap_ptr, render_width, render_height, bitmap_stride, hemisamples.Length, SceneFormatter.CreateFlatMemoryF32(hemisamples));
        }
        public static void AOCHaltonBiasedFMPCUDAF32(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int render_width, int render_height, int bitmap_stride)
        {
            float[] acc = new float[4 * render_width * render_height];
            int accumulator_stride = sizeof(float) * 4 * render_width;
            var hemisamples = MathHelp.HemiHaltonCosineBias(32).ToArray();
            GCHandle handle_acc = GCHandle.Alloc(acc, GCHandleType.Pinned);
            RenderToyCLI.AmbientOcclusionFMPCUDAF32(SceneFormatter.CreateFlatMemoryF32(scene), SceneFormatter.CreateFlatMemoryF32(MathHelp.Invert(mvp)), handle_acc.AddrOfPinnedObject(), render_width, render_height, accumulator_stride, hemisamples.Length, SceneFormatter.CreateFlatMemoryF32(hemisamples));
            RenderToyCLI.ToneMap(handle_acc.AddrOfPinnedObject(), accumulator_stride, bitmap_ptr, render_width, render_height, bitmap_stride, 1.0f);
            handle_acc.Free();
        }
        public static void AOCHaltonBiasedCUDAF32(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int render_width, int render_height, int bitmap_stride)
        {
            var hemisamples = MathHelp.HemiHaltonCosineBias(512).ToArray();
            RenderToyCLI.AmbientOcclusionCUDAF32(SceneFormatter.CreateFlatMemoryF32(scene), SceneFormatter.CreateFlatMemoryF32(MathHelp.Invert(mvp)), bitmap_ptr, render_width, render_height, bitmap_stride, hemisamples.Length, SceneFormatter.CreateFlatMemoryF32(hemisamples));
        }
        public static void AOCHaltonUnbiasedCUDAF32(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int render_width, int render_height, int bitmap_stride)
        {
            var hemisamples = MathHelp.HemiHaltonUnbiased(512).ToArray();
            RenderToyCLI.AmbientOcclusionCUDAF32(SceneFormatter.CreateFlatMemoryF32(scene), SceneFormatter.CreateFlatMemoryF32(MathHelp.Invert(mvp)), bitmap_ptr, render_width, render_height, bitmap_stride, hemisamples.Length, SceneFormatter.CreateFlatMemoryF32(hemisamples));
        }
        public static void AOCHammerslyBiasedCUDAF32(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int render_width, int render_height, int bitmap_stride)
        {
            var hemisamples = MathHelp.HemiHammerslyCosineBias(512).ToArray();
            RenderToyCLI.AmbientOcclusionCUDAF32(SceneFormatter.CreateFlatMemoryF32(scene), SceneFormatter.CreateFlatMemoryF32(MathHelp.Invert(mvp)), bitmap_ptr, render_width, render_height, bitmap_stride, hemisamples.Length, SceneFormatter.CreateFlatMemoryF32(hemisamples));
        }
        public static void AOCHammerslyUnbiasedCUDAF32(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int render_width, int render_height, int bitmap_stride)
        {
            var hemisamples = MathHelp.HemiHammerslyUnbiased(512).ToArray();
            RenderToyCLI.AmbientOcclusionCUDAF32(SceneFormatter.CreateFlatMemoryF32(scene), SceneFormatter.CreateFlatMemoryF32(MathHelp.Invert(mvp)), bitmap_ptr, render_width, render_height, bitmap_stride, hemisamples.Length, SceneFormatter.CreateFlatMemoryF32(hemisamples));
        }
        public static void AOCRandomBiasedCUDAF32(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int render_width, int render_height, int bitmap_stride)
        {
            var hemisamples = MathHelp.HemiRandomCosineBias(512).ToArray();
            RenderToyCLI.AmbientOcclusionCUDAF32(SceneFormatter.CreateFlatMemoryF32(scene), SceneFormatter.CreateFlatMemoryF32(MathHelp.Invert(mvp)), bitmap_ptr, render_width, render_height, bitmap_stride, hemisamples.Length, SceneFormatter.CreateFlatMemoryF32(hemisamples));
        }
        public static void AOCRandomUnbiasedCUDAF32(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int render_width, int render_height, int bitmap_stride)
        {
            var hemisamples = MathHelp.HemiRandomUnbiased(512).ToArray();
            RenderToyCLI.AmbientOcclusionCUDAF32(SceneFormatter.CreateFlatMemoryF32(scene), SceneFormatter.CreateFlatMemoryF32(MathHelp.Invert(mvp)), bitmap_ptr, render_width, render_height, bitmap_stride, hemisamples.Length, SceneFormatter.CreateFlatMemoryF32(hemisamples));
        }
    }
}