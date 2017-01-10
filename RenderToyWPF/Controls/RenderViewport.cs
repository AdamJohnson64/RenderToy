////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System;
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
                DrawHelp.fnDrawLineViewport lineviewport = ControlUtil.CreateLineViewportFunction(renderer, ActualWidth, ActualHeight);
                DrawHelp.fnDrawLineWorld line = ControlUtil.CreateLineWorldFunction(lineviewport, MVP);
                renderer.WireframeBegin();
                renderer.WireframeColor(0.0, 1.0, 1.0);
                DrawHelp.DrawClipSpace(line, other);
                renderer.WireframeEnd();
            }
            DateTime timeEnd = DateTime.Now;
            // Try to maintain 30FPS by reducing quality.
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
        protected override void OnRenderToy(DrawingContext drawingContext)
        {
            ControlUtil.DrawWireframeGDI(Scene, MVP, (int)Math.Ceiling(ActualWidth), (int)Math.Ceiling(ActualHeight), drawingContext, ActualWidth, ActualHeight);
            Action<Func<Scene, Matrix3D, int, int, ImageSource>, int, int, int> drawpreview = (drawhelper, stacky, render_width, render_height) =>
            {
                double frame_l = ActualWidth - 192 - 8;
                double frame_t = 8 + (128 + 8) * stacky;
                double frame_w = 192;
                double frame_h = 128;
                double image_l = frame_l + 8;
                double image_t = frame_t + 8;
                double image_w = frame_w - 8 * 2;
                double image_h = frame_h - 8 * 2;
                drawingContext.DrawRoundedRectangle(Brushes.White, new Pen(Brushes.DarkGray, 2), new Rect(frame_l, frame_t, frame_w, frame_h), 8, 8);
                drawingContext.DrawImage(drawhelper(Scene, View * Projection * AspectCorrectFit(image_w, image_h), render_width, render_height), new Rect(image_l, image_t, image_w, image_h));
            };
            drawpreview(ControlUtil.ImagePoint, 0, ReduceQuality ? 32 : 64, ReduceQuality ? 32 : 64);
            drawpreview(ControlUtil.ImageRaster, 1, ReduceQuality ? 32 : 128, ReduceQuality ? 32 : 128);
            drawpreview(ControlUtil.ImageRaytrace, 2, ReduceQuality ? 32 : 128, ReduceQuality ? 32 : 128);
        }
    }
    class RenderViewportPoint : RenderViewportBase
    {
        protected override void OnRenderToy(DrawingContext drawingContext)
        {
            ControlUtil.DrawPoint(Scene, MVP, (int)Math.Ceiling(ActualWidth), (int)Math.Ceiling(ActualHeight), drawingContext, ActualWidth, ActualHeight);
        }
    }
    class RenderViewportPointGDI : RenderViewportBase
    {
        protected override void OnRenderToy(DrawingContext drawingContext)
        {
            ControlUtil.DrawPointGDI(Scene, MVP, (int)Math.Ceiling(ActualWidth), (int)Math.Ceiling(ActualHeight), drawingContext, ActualWidth, ActualHeight);
        }
    }
    class RenderViewportWireframeGDI : RenderViewportBase
    {
        protected override void OnRenderToy(DrawingContext drawingContext)
        {
            ControlUtil.DrawWireframeGDI(Scene, MVP, (int)Math.Ceiling(ActualWidth), (int)Math.Ceiling(ActualHeight), drawingContext, ActualWidth, ActualHeight);
        }
    }
    class RenderViewportWireframeWPF : RenderViewportBase
    {
        protected override void OnRenderToy(DrawingContext drawingContext)
        {
            ControlUtil.DrawWireframeWPF(Scene, MVP, drawingContext, ActualWidth, ActualHeight);
        }
    }
    class RenderViewportRaster : RenderViewportBase
    {
        protected override void OnRenderToy(DrawingContext drawingContext)
        {
            ControlUtil.DrawRaster(Scene, MVP, ReduceQuality ? 128 : 512, ReduceQuality ? 128 : 512, drawingContext, ActualWidth, ActualHeight);
        }
    }
    class RenderViewportRasterD3D : RenderViewportBase
    {
        protected override void OnRenderToy(DrawingContext drawingContext)
        {
            ControlUtil.DrawRasterD3D9(Scene, MVP, (int)Math.Ceiling(ActualWidth), (int)Math.Ceiling(ActualHeight), drawingContext, ActualWidth, ActualHeight);
        }
    }
    class RenderViewportRaytrace : RenderViewportBase
    {
        protected override void OnRenderToy(DrawingContext drawingContext)
        {
            ControlUtil.DrawRaytrace(Scene, MVP, ReduceQuality ? 128 : 512, ReduceQuality ? 128 : 512, drawingContext, ActualWidth, ActualHeight);
        }
    }
}