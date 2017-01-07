using System;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Media3D;

namespace RenderToy
{
    public class RenderViewport : FrameworkElement
    {
        public static DependencyProperty DrawExtraProperty = DependencyProperty.Register("DrawExtra", typeof(RenderViewport), typeof(RenderViewport));
        public RenderViewport DrawExtra { get { return (RenderViewport)GetValue(DrawExtraProperty); } set { SetValue(DrawExtraProperty, value);  } }
        public Scene Scene = Scene.Default;
        public RenderViewport()
        {
            AllowDrop = true;
        }
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
            dragging = true;
            clickOrigin = System.Windows.Forms.Cursor.Position;
            dragOrigin = e.GetPosition(this);
        }
        protected override void OnMouseLeftButtonUp(MouseButtonEventArgs e)
        {
            base.OnMouseLeftButtonUp(e);
            Mouse.OverrideCursor = null;
            ReleaseMouseCapture();
            dragging = false;
            InvalidateVisual();
        }
        protected override void OnMouseMove(MouseEventArgs e)
        {
            base.OnMouseMove(e);
            if (!dragging) return;
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
        private bool dragging = false;
        private System.Drawing.Point clickOrigin;
        private System.Windows.Point dragOrigin;
        #endregion
        #region - Section : Rendering -
        protected override void OnRender(DrawingContext drawingContext)
        {
            base.OnRender(drawingContext);
            //ControlUtil.RenderRasterD3D9(drawingContext, ActualWidth, ActualHeight, Scene, View * ProjectionWindow, (int)Math.Ceiling(ActualWidth), (int)Math.Ceiling(ActualHeight));
            ControlUtil.RenderRaytrace(drawingContext, ActualWidth, ActualHeight, Scene, View * ProjectionWindow, dragging ? 128 : 512, dragging ? 128 : 512);
            ControlUtil.RenderWireframeGDI(drawingContext, ActualWidth, ActualHeight, Scene, View * ProjectionWindow, (int)Math.Ceiling(ActualWidth), (int)Math.Ceiling(ActualHeight));
            //ControlUtil.RenderWireframeWPF(drawingContext, ActualWidth, ActualHeight, Scene, View * ProjectionWindow);
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
        }
        #endregion
    }
}