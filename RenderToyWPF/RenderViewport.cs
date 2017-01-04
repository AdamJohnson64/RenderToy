using System;
using System.Windows;
using System.Windows.Controls;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Media.Media3D;

namespace RenderToy
{
    public class DragSourcePrimitive : FrameworkElement
    {
        public DragSourcePrimitive(IParametricUV primitive)
        {
            Width = 64;
            Height = 64;
            Primitive = primitive;
        }
        protected override void OnMouseLeftButtonDown(MouseButtonEventArgs e)
        {
            base.OnMouseLeftButtonDown(e);
            DragDrop.DoDragDrop(this, new Sphere(), DragDropEffects.Copy);
        }
        protected override void OnRender(DrawingContext drawingContext)
        {
            base.OnRender(drawingContext);
            IWireframeRenderer renderer = new WireframeWPF(drawingContext);
            drawingContext.DrawRectangle(Brushes.Transparent, null, new Rect(0, 0, Math.Ceiling(ActualWidth), Math.Ceiling(ActualHeight)));
            DrawHelp.fnDrawLineViewport linev = CreateLineViewportFunction(renderer);
            Matrix3D View = MathHelp.CreateTranslateMatrix(0, 0, 4);
            Matrix3D ProjectionWindow = CameraPerspective.CreateProjection(0.001, 100, 45.0 * Math.PI / 180.0, 45.0 * Math.PI / 180.0);
            DrawHelp.fnDrawLineWorld line = CreateLineWorldFunction(linev, View * ProjectionWindow);
            renderer.WireframeBegin();
            renderer.WireframeColor(0, 0, 0);
            DrawHelp.DrawParametricUV(line, Primitive);
            renderer.WireframeEnd();
        }
        private DrawHelp.fnDrawLineViewport CreateLineViewportFunction(IWireframeRenderer renderer)
        {
            double width = ActualWidth;
            double height = ActualHeight;
            return (p1, p2) =>
            {
                renderer.WireframeLine(
                    (p1.X + 1) * width / 2, (1 - p1.Y) * height / 2,
                    (p2.X + 1) * width / 2, (1 - p2.Y) * height / 2);
            };
        }
        private DrawHelp.fnDrawLineWorld CreateLineWorldFunction(DrawHelp.fnDrawLineViewport line, Matrix3D mvp)
        {
            return (p1, p2) =>
            {
                DrawHelp.DrawLineWorld(line, mvp, new Point4D(p1.X, p1.Y, p1.Z, 1.0), new Point4D(p2.X, p2.Y, p2.Z, 1.0));
            };
        }
        private IParametricUV Primitive;
    }
    public class DragSourceBezierPatch : DragSourcePrimitive
    {
        public DragSourceBezierPatch() : base(new BezierPatch()) { }
    }
    public class DragSourceCylinder : DragSourcePrimitive
    {
        public DragSourceCylinder() : base(new Cylinder()) { }
    }
    public class DragSourceSphere : DragSourcePrimitive
    {
        public DragSourceSphere() : base(new Sphere()) { }
    }
    public class RenderViewport : FrameworkElement
    {
        public static DependencyProperty DrawExtraProperty = DependencyProperty.Register("DrawExtra", typeof(RenderViewport), typeof(RenderViewport));
        public RenderViewport DrawExtra { get { return (RenderViewport)GetValue(DrawExtraProperty); } set { SetValue(DrawExtraProperty, value);  } }
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
        IWireframeRenderer renderer;
        protected override void OnRender(DrawingContext drawingContext)
        {
            base.OnRender(drawingContext);
            ////////////////////////////////////////////////////////////////////////////////
            // RAYTRACE OVERLAY
            // Raytrace the first layer so we have a reference image.
            {
                const int raytrace_width = 128;
                const int raytrace_height = 128;
                var bitmap = new WriteableBitmap(raytrace_width, raytrace_height, 0, 0, PixelFormats.Bgra32, null);
                Matrix3D inverse_mvp = MathHelp.Invert(View * ProjectionWindow);
                bitmap.Lock();
                Raytrace.DoRaytrace(inverse_mvp, bitmap.PixelWidth, bitmap.PixelHeight, bitmap.BackBuffer, bitmap.BackBufferStride);
                bitmap.AddDirtyRect(new Int32Rect(0, 0, bitmap.PixelWidth, bitmap.PixelHeight));
                bitmap.Unlock();
                drawingContext.DrawImage(bitmap, new Rect(0, 0, ActualWidth, ActualHeight));
            }
            ////////////////////////////////////////////////////////////////////////////////
            // WIREFRAME OVERLAY
            // Select an appropriate renderer.
            //renderer = new WireframeWPF(drawingContext);
            renderer = new WireframeGDIPlus(drawingContext, (int)Math.Ceiling(ActualWidth), (int)Math.Ceiling(ActualHeight));
            // Draw the scene.
            drawingContext.DrawRectangle(Brushes.Transparent, null, new Rect(0, 0, Math.Ceiling(ActualWidth), Math.Ceiling(ActualHeight)));
            DrawHelp.fnDrawLineViewport linev = CreateLineViewportFunction(renderer);
            DrawHelp.fnDrawLineWorld line = CreateLineWorldFunction(linev, View * ProjectionWindow);
            renderer.WireframeBegin();
            renderer.WireframeColor(0.75, 0.75, 0.75);
            DrawHelp.DrawPlane(line);
            // Draw something interesting.
            renderer.WireframeColor(0.0, 0.0, 0.0);
            DrawHelp.DrawParametricUV(line, new Sphere());
            //DrawHelp.DrawParametricUV(line, new BezierPatch());
            //DrawHelp.DrawTeapot(line);
            // If we're connected to another view camera then show it here.
            if (DrawExtra != null)
            {
                // Draw the clip space of the Model-View-Projection.
                Matrix3D other = MathHelp.Invert(DrawExtra.View * DrawExtra.ProjectionWindow);
                renderer.WireframeColor(0.0, 1.0, 1.0);
                DrawHelp.DrawClipSpace(line, other);
            }
            renderer.WireframeEnd();
        }
        private DrawHelp.fnDrawLineViewport CreateLineViewportFunction(IWireframeRenderer renderer)
        {
            double width = ActualWidth;
            double height = ActualHeight;
            return (p1, p2) =>
            {
                renderer.WireframeLine(
                    (p1.X + 1) * width / 2, (1 - p1.Y) * height / 2,
                    (p2.X + 1) * width / 2, (1 - p2.Y) * height / 2);
            };
        }
        private DrawHelp.fnDrawLineWorld CreateLineWorldFunction(DrawHelp.fnDrawLineViewport line, Matrix3D mvp)
        {
            return (p1, p2) =>
            {
                DrawHelp.DrawLineWorld(line, mvp, new Point4D(p1.X, p1.Y, p1.Z, 1.0), new Point4D(p2.X, p2.Y, p2.Z, 1.0));
            };
        }
        #endregion
    }
}