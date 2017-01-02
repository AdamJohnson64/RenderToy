using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Media3D;

namespace RenderToy
{
    public class RenderViewport : UserControl
    {
        public static DependencyProperty DrawExtraProperty = DependencyProperty.Register("DrawExtra", typeof(RenderViewport), typeof(RenderViewport));
        public RenderViewport DrawExtra { get { return (RenderViewport)GetValue(DrawExtraProperty); } set { SetValue(DrawExtraProperty, value);  } }
        #region - Section : Camera -
        private Matrix3D View
        {
            get
            {
                return MathHelp.Invert(Camera.Transform);
            }
        }
        private Matrix3D Projection
        {
            get
            {
                return CameraMat.Projection;
            }
        }
        private Matrix3D ProjectionWindow
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
        private TransformPosQuat Camera = new TransformPosQuat { Position = new Vector3D(0, 10, -20) };
        CameraPerspective CameraMat = new CameraPerspective();
        #endregion
        #region - Section : Coloring -
        private static Brush Brush_Background = Brushes.Black;
        private static Brush Brush_Frustum = Brushes.Cyan;
        private static Brush Brush_WorkingGrid = Brushes.LightGray;
        #endregion
        #region - Section : Input Handling -
        protected override void OnMouseLeftButtonDown(MouseButtonEventArgs e)
        {
            base.OnMouseLeftButtonDown(e);
            CaptureMouse();
            dragging = true;
            dragOrigin = e.GetPosition(this);
        }
        protected override void OnMouseLeftButtonUp(MouseButtonEventArgs e)
        {
            base.OnMouseLeftButtonUp(e);
            ReleaseMouseCapture();
            dragging = false;
        }
        protected override void OnMouseMove(MouseEventArgs e)
        {
            base.OnMouseMove(e);
            if (!dragging) return;
            Point dragTo = e.GetPosition(this);
            double dx = dragTo.X - dragOrigin.X;
            double dy = dragTo.Y - dragOrigin.Y;
            dragOrigin = dragTo;
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
        private bool dragging = false;
        private Point dragOrigin;
        #endregion
        #region - Section : Rendering -
        protected override void OnRender(DrawingContext drawingContext)
        {
            base.OnRender(drawingContext);
            drawingContext.DrawRectangle(Brush_Background, null, new Rect(0, 0, Math.Ceiling(ActualWidth), Math.Ceiling(ActualHeight)));
            // Compute the view matrix.
            Matrix3D transform_mvp = View * ProjectionWindow;
            // Stupid time; draw a grid representing the XZ plane.
            DrawHelp.fnDrawLineWorld drawline = CreateLineFunction(drawingContext, new Pen(Brush_WorkingGrid, -1), transform_mvp);
            DrawHelp.DrawPlane(drawline);
            DrawHelp.DrawParametricUV(drawline, new Sphere());
            if (DrawExtra != null)
            {
                // Draw the clip space of the Model-View-Projection.
                Matrix3D other = MathHelp.Invert(DrawExtra.View * DrawExtra.ProjectionWindow);
                DrawHelp.DrawClipSpace(CreateLineFunction(drawingContext, new Pen(Brush_Frustum, -1), transform_mvp), other);
            }
        }
        private DrawHelp.fnDrawLineViewport CreateLineViewportFunction(DrawingContext drawingContext, Pen pen)
        {
            double width = ActualWidth;
            double height = ActualHeight;
            return (p1, p2) =>
            {
                // Accept post-w-divided lines and render them to screen.
                drawingContext.DrawLine(pen,
                    new Point((p1.X + 1) * width / 2, (1 - p1.Y) * height / 2),
                    new Point((p2.X + 1) * width / 2, (1 - p2.Y) * height / 2));
            };
        }
        private DrawHelp.fnDrawLineWorld CreateLineFunction(DrawingContext drawingContext, Pen pen, Matrix3D mvp)
        {
            DrawHelp.fnDrawLineViewport drawviewport = CreateLineViewportFunction(drawingContext, pen);
            return (p1, p2) =>
            {
                DrawHelp.DrawLineWorld(drawviewport, mvp, new Point4D(p1.X, p1.Y, p1.Z, 1.0), new Point4D(p2.X, p2.Y, p2.Z, 1.0));
            };
        }
        #endregion
    }
}