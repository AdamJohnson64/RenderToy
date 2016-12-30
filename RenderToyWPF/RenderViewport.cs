using System;
using System.Diagnostics;
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
        public RenderViewport()
        {
            ClipToBounds = true;
        }
        #region - Section : Camera -
        private Matrix3D View
        {
            get
            {
                return Camera.Transform;
            }
        }
        private Matrix3D Projection
        {
            get
            {
                return CameraMat.Projection;
            }
        }
        private TransformPosQuat Camera = new TransformPosQuat { Position = new Vector3D(0, 10, -20) };
        CameraPerspective CameraMat = new CameraPerspective();
        #endregion
        #region - Section : Coloring -
        private static Brush Brush_Background = Brushes.Black;
        private static Brush Brush_Frame = Brushes.DarkGray;
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
            drawingContext.DrawRectangle(Brush_Background, null, new Rect(0, 0, ActualWidth, ActualHeight));
            drawingContext.DrawRectangle(null, new Pen(Brush_Frame, -1), new Rect(4, 4, ActualWidth - 8, ActualHeight - 8));
            // Compute the view matrix.
            Matrix3D transform_camera = MathHelp.Invert(View);
            // Compute the projection matrix.
            Matrix3D transform_projection = Projection;
            // Aspect correct.
            // We're using "at least" FOV correction so horizontal or vertical can be extended.
            double a = ActualWidth / ActualHeight;
            if (a > 1)
            {
                Matrix3D scale = new Matrix3D(
                    1 / a, 0, 0, 0,
                    0, 1, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1);
                transform_projection = transform_projection * scale;
            }
            if (a < 1)
            {
                Matrix3D scale = new Matrix3D(
                    1, 0, 0, 0,
                    0, a, 0, 0,
                    0, 0, 1, 0,
                    0, 0, 0, 1);
                transform_projection = transform_projection * scale;
            }
            // Stupid time; draw a grid representing the XZ plane.
            Matrix3D transform_mvp = transform_camera * transform_projection;
            Pen pen = new Pen(Brush_WorkingGrid, -1);
            for (int i = 0; i <= 20; ++i)
            {
                // Draw an X line.
                float z = -10.0f + i;
                DrawLine3D(drawingContext, pen, new Point4D(-10, 0, z, 1), new Point4D(10, 0, z, 1), transform_mvp);
                // Draw a Z line.
                float x = -10.0f + i;
                DrawLine3D(drawingContext, pen, new Point4D(x, 0, -10, 1), new Point4D(x, 0, 10, 1), transform_mvp);
            }
            ////////////////////////////////////////////////////////////////////////////////
            // TODO: We're just going to draw a parametric here for testing.
            // Later we'll want to hoist this code out somewhere more meaningful.
            const int USEGMENTS = 9;
            const int VSEGMENTS = 9;
            IParametricUV uv = new Sphere();
            for (int i2 = 0; i2 <= VSEGMENTS; ++i2)
            {
                for (int i1 = 0; i1 < USEGMENTS; ++i1)
                {
                    // Draw U Lines.
                    {
                        Point3D p3u1 = uv.GetPointUV((i1 + 0.0) / USEGMENTS, (i2 + 0.0) / VSEGMENTS);
                        Point3D p3u2 = uv.GetPointUV((i1 + 1.0) / USEGMENTS, (i2 + 0.0) / VSEGMENTS);
                        Point4D p4u1 = new Point4D(p3u1.X, p3u1.Y, p3u1.Z, 1.0);
                        Point4D p4u2 = new Point4D(p3u2.X, p3u2.Y, p3u2.Z, 1.0);
                        DrawLine3D(drawingContext, pen, p4u1, p4u2, transform_mvp);
                    }
                    // Draw V Lines.
                    {
                        Point3D p3u1 = uv.GetPointUV((i2 + 0.0) / USEGMENTS, (i1 + 0.0) / VSEGMENTS);
                        Point3D p3u2 = uv.GetPointUV((i2 + 0.0) / USEGMENTS, (i1 + 1.0) / VSEGMENTS);
                        Point4D p4u1 = new Point4D(p3u1.X, p3u1.Y, p3u1.Z, 1.0);
                        Point4D p4u2 = new Point4D(p3u2.X, p3u2.Y, p3u2.Z, 1.0);
                        DrawLine3D(drawingContext, pen, p4u1, p4u2, transform_mvp);
                    }
                }
            }
            ////////////////////////////////////////////////////////////////////////////////
            if (DrawExtra != null)
            {
                // Compute the inverse of the MVP.
                Matrix3D other = MathHelp.Invert(DrawExtra.View) * Projection;
                other.Invert();
                // Start drawing the frustum.
                Pen pen_frustum = new Pen(Brush_Frustum, -1);
                {
                    // Transform the edges of homogeneous space into 3-space.
                    Point4D[] points = new Point4D[8];
                    for (int z = 0; z < 2; ++z)
                    {
                        for (int y = 0; y < 2; ++y)
                        {
                            for (int x = 0; x < 2; ++x)
                            {
                                Point4D p = other.Transform(new Point4D(-1 + x * 2, -1 + y * 2, z, 1));
                                // Homogeneous divide puts us back in real space.
                                p.X /= p.W; p.Y /= p.W; p.Z /= p.W; p.W = 1;
                                points[x + y * 2 + z * 4] = p;
                            }
                        }
                    }
                    // Draw the projection "rails" (the z-expanse lines from the four corners of the viewport).
                    DrawLine3D(drawingContext, pen_frustum, points[0], points[4], transform_mvp);
                    DrawLine3D(drawingContext, pen_frustum, points[1], points[5], transform_mvp);
                    DrawLine3D(drawingContext, pen_frustum, points[2], points[6], transform_mvp);
                    DrawLine3D(drawingContext, pen_frustum, points[3], points[7], transform_mvp);
                }
                {
                    // Draw several depth viewport frames at constant z spacing.
                    for (int z = 0; z <= 10; ++z)
                    {
                        Point4D[] frame = new Point4D[4];
                        for (int y = 0; y < 2; ++y)
                        {
                            for (int x = 0; x < 2; ++x)
                            {
                                Point4D homocoord = new Point4D(-1 + x * 2, -1 + y * 2, z / 10.0, 1);
                                Point4D p = other.Transform(homocoord);
                                // Homogeneous divide puts us back in real space.
                                p.X /= p.W; p.Y /= p.W; p.Z /= p.W; p.W = 1;
                                frame[x + y * 2] = p;
                            }
                        }
                        DrawLine3D(drawingContext, pen_frustum, frame[0], frame[1], transform_mvp);
                        DrawLine3D(drawingContext, pen_frustum, frame[0], frame[2], transform_mvp);
                        DrawLine3D(drawingContext, pen_frustum, frame[1], frame[3], transform_mvp);
                        DrawLine3D(drawingContext, pen_frustum, frame[2], frame[3], transform_mvp);
                    }
                }
            }
        }
        #endregion
        #region - Section : Primitive Handling -
        private void DrawLine3D(DrawingContext drawingContext, Pen pen, Point4D p1, Point4D p2, Matrix3D mvp)
        {
            TransformLine3D(mvp, ref p1, ref p2);
            if (!ClipLine3D(ref p1, ref p2)) return;
            DrawLine3DUnclipped(drawingContext, pen, p1, p2);
        }
        private void DrawLine3DUnclipped(DrawingContext drawingContext, Pen pen, Point4D p1, Point4D p2)
        {
            // Perform homogeneous divide.
            p1.X = p1.X / p1.W; p1.Y = p1.Y / p1.W; p1.Z = p1.Z / p1.W; p1.W = p1.W / p1.W;
            p2.X = p2.X / p2.W; p2.Y = p2.Y / p2.W; p2.Z = p2.Z / p2.W; p2.W = p2.W / p2.W;
            // Perform viewport transform.
            Point vp1 = new Point((p1.X + 1) * ActualWidth / 2, (1 - p1.Y) * ActualHeight / 2);
            Point vp2 = new Point((p2.X + 1) * ActualWidth / 2, (1 - p2.Y) * ActualHeight / 2);
            // Draw it!
            drawingContext.DrawLine(pen, vp1, vp2);
        }
        private static bool ClipLine3D(ref Point4D p1, ref Point4D p2)
        {
            // Clip to frustum edges.
            if (!ClipLine3D(ref p1, ref p2, new Point4D(0, 0, 1, 0))) return false;
            if (!ClipLine3D(ref p1, ref p2, new Point4D(0, 0, -1, 1))) return false;
            if (!ClipLine3D(ref p1, ref p2, new Point4D(-1, 0, 0, 1))) return false;
            if (!ClipLine3D(ref p1, ref p2, new Point4D(1, 0, 0, 1))) return false;
            if (!ClipLine3D(ref p1, ref p2, new Point4D(0, -1, 0, 1))) return false;
            if (!ClipLine3D(ref p1, ref p2, new Point4D(0, 1, 0, 1))) return false;
            return true;
        }
        private static bool ClipLine3D(ref Point4D p1, ref Point4D p2, Point4D plane)
        {
            // Determine which side of the plane these points reside.
            double side_p1 = MathHelp.Dot(p1, plane);
            double side_p2 = MathHelp.Dot(p2, plane);
            // If the line is completely behind the clip plane then reject it immediately.
            if (side_p1 <= 0 && side_p2 <= 0) return false;
            // If the line is completely in front of the clip plane then accept it immediately.
            if (side_p1 >= 0 && side_p2 >= 0) return true;
            // Otherwise the line straddles the clip plane; clip as appropriate.
            // Construct a line segment to clip.
            Point4D line_org = p1;
            Point4D line_dir = p2 - p1;
            // Compute the intersection with the clip plane.
            double lambda = -MathHelp.Dot(plane, line_org) / MathHelp.Dot(plane, line_dir);
            // If the intersection lies in the line segment then clip.
            if (lambda > 0 && lambda < 1)
            {
                // If P1 was behind the plane them move it to the intersection point.
                if (side_p1 <= 0) p1 = line_org + MathHelp.Scale(line_dir, lambda);
                // If P2 was behind the plane then move it to the intersection point.
                if (side_p2 <= 0) p2 = line_org + MathHelp.Scale(line_dir, lambda);
            }
            return true;
        }
        private static void TransformLine3D(Matrix3D mvp, ref Point4D p1, ref Point4D p2)
        {
            // Transform the supplied points into projection space.
            p1 = mvp.Transform(p1);
            p2 = mvp.Transform(p2);
        }
        #endregion
    }
}