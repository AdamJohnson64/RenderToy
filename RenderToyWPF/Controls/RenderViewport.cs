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
        protected Matrix3D ProjectionWindow
        {
            get
            {
                // Aspect correct; We're using "at least" FOV correction so horizontal or vertical can be extended.
                {
                    double aspect = ActualWidth / ActualHeight;
                    if (aspect > 1)
                    {
                        return Projection * MathHelp.CreateScaleMatrix(1 / aspect, 1, 1);
                    }
                    else
                    {
                        return Projection * MathHelp.CreateScaleMatrix(1, aspect, 1);
                    }
                }
            }
        }
        TransformPosQuat Camera = new TransformPosQuat { Position = new Vector3D(0, 10, -20) };
        CameraPerspective CameraMat = new CameraPerspective();
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
                Matrix3D other = MathHelp.Invert(DrawExtra.View * DrawExtra.ProjectionWindow);
                IWireframeRenderer renderer = new WireframeWPF(drawingContext);
                DrawHelp.fnDrawLineViewport lineviewport = ControlUtil.CreateLineViewportFunction(renderer, ActualWidth, ActualHeight);
                DrawHelp.fnDrawLineWorld line = ControlUtil.CreateLineWorldFunction(lineviewport, View * ProjectionWindow);
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
            ControlUtil.DrawWireframeGDI(Scene, View * ProjectionWindow, (int)Math.Ceiling(ActualWidth), (int)Math.Ceiling(ActualHeight), drawingContext, ActualWidth, ActualHeight);
            ControlUtil.DrawPoint(Scene, View * ProjectionWindow, 256, 256, drawingContext, ActualWidth, ActualHeight);
            //ControlUtil.DrawRaster(Scene, View * ProjectionWindow, ReduceQuality ? 128 : 512, ReduceQuality ? 128 : 512, drawingContext, ActualWidth, ActualHeight);
            drawingContext.DrawImage(ControlUtil.ImageRaytrace(Scene, View * Projection, ReduceQuality ? 64 : 256, ReduceQuality ? 64 : 256), new Rect(ActualWidth - 256 - 8, 8, 256, 256));
        }
    }
    class RenderViewportPoint : RenderViewportBase
    {
        protected override void OnRenderToy(DrawingContext drawingContext)
        {
            ControlUtil.DrawPoint(Scene, View * ProjectionWindow, (int)Math.Ceiling(ActualWidth), (int)Math.Ceiling(ActualHeight), drawingContext, ActualWidth, ActualHeight);
        }
    }
    class RenderViewportPointGDI : RenderViewportBase
    {
        protected override void OnRenderToy(DrawingContext drawingContext)
        {
            ControlUtil.DrawPointGDI(Scene, View * ProjectionWindow, (int)Math.Ceiling(ActualWidth), (int)Math.Ceiling(ActualHeight), drawingContext, ActualWidth, ActualHeight);
        }
    }
    class RenderViewportWireframeGDI : RenderViewportBase
    {
        protected override void OnRenderToy(DrawingContext drawingContext)
        {
            ControlUtil.DrawWireframeGDI(Scene, View * ProjectionWindow, (int)Math.Ceiling(ActualWidth), (int)Math.Ceiling(ActualHeight), drawingContext, ActualWidth, ActualHeight);
        }
    }
    class RenderViewportWireframeWPF : RenderViewportBase
    {
        protected override void OnRenderToy(DrawingContext drawingContext)
        {
            ControlUtil.DrawWireframeWPF(Scene, View * ProjectionWindow, drawingContext, ActualWidth, ActualHeight);
        }
    }
    class RenderViewportRaster : RenderViewportBase
    {
        protected override void OnRenderToy(DrawingContext drawingContext)
        {
            ControlUtil.DrawRaster(Scene, View * ProjectionWindow, ReduceQuality ? 128 : 512, ReduceQuality ? 128 : 512, drawingContext, ActualWidth, ActualHeight);
        }
    }
    class RenderViewportRasterD3D : RenderViewportBase
    {
        protected override void OnRenderToy(DrawingContext drawingContext)
        {
            ControlUtil.DrawRasterD3D9(Scene, View * ProjectionWindow, (int)Math.Ceiling(ActualWidth), (int)Math.Ceiling(ActualHeight), drawingContext, ActualWidth, ActualHeight);
        }
    }
    class RenderViewportRaytrace : RenderViewportBase
    {
        protected override void OnRenderToy(DrawingContext drawingContext)
        {
            ControlUtil.DrawRaytrace(Scene, View * ProjectionWindow, ReduceQuality ? 128 : 512, ReduceQuality ? 128 : 512, drawingContext, ActualWidth, ActualHeight);
        }
    }
}