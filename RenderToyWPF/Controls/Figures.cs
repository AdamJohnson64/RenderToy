using System;
using System.Collections.Generic;
using System.Globalization;
using System.Linq;
using System.Windows;
using System.Windows.Input;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using RenderToy.PipelineModel;
using RenderToy.SceneGraph;
using RenderToy.SceneGraph.Cameras;
using RenderToy.SceneGraph.Materials;
using RenderToy.SceneGraph.Primitives;
using RenderToy.SceneGraph.Transforms;

namespace RenderToy.WPF.Figures
{
    #region - Section : Document Figures -
    abstract class FigureBase : FrameworkElement
    {
        #region - Section : Abstract -
        protected abstract void RenderFigure(DrawingContext drawingContext);
        #endregion
        #region - Overrides : UIElement -
        protected override void OnRender(DrawingContext drawingContext)
        {
            drawingContext.DrawRectangle(Brushes.Transparent, null, new Rect(0, 0, ActualWidth, ActualHeight));
            RenderFigure(drawingContext);
        }
        #endregion
    }
    #endregion
    #region - Section : 3D Viewport Document Figures -
    abstract class Figure3DBase : FigureBase
    {
        #region - Section : Properties -
        public Matrix3D MVP
        {
            get
            {
                var mvp = MathHelp.Invert(Camera.Object.Transform);
                mvp = MathHelp.Multiply(mvp, Perspective.CreateProjection(0.01, 100.0, 60.0 * Math.PI / 180.0, 60.0 * Math.PI / 180.0));
                mvp = MathHelp.Multiply(mvp, Perspective.AspectCorrectFit(ActualWidth, ActualHeight));
                return mvp;
            }
        }
        public Scene Scene
        {
            get
            {
                var scene = new Scene();
                scene.AddChild(new Node("Plane", new TransformMatrix(MathHelp.CreateMatrixScale(10, 10, 10)), new Plane(), StockMaterials.Red, StockMaterials.PlasticRed));
                return scene;
            }
        }
        #endregion
        #region - Section : Dependency Properties -
        public static DependencyProperty CameraProperty = DependencyProperty.Register("Camera", typeof(Camera), typeof(Figure3DBase));
        public Camera Camera
        {
            get { return (Camera)GetValue(CameraProperty); }
            set { SetValue(CameraProperty, value); }
        }
        #endregion
        #region - Section : Construction -
        public Figure3DBase()
        {
            Camera = new Camera();
            ClipToBounds = true;
        }
        #endregion
        #region - Overrides : UIElement -
        protected override void OnMouseLeftButtonDown(MouseButtonEventArgs e)
        {
            base.OnMouseLeftButtonDown(e);
            Focus();
            CaptureMouse();
            Mouse.OverrideCursor = Cursors.None;
            isDragging = true;
            dragFrom = System.Windows.Forms.Cursor.Position;
            e.Handled = true;
        }
        protected override void OnMouseLeftButtonUp(MouseButtonEventArgs e)
        {
            base.OnMouseLeftButtonUp(e);
            Mouse.OverrideCursor = null;
            ReleaseMouseCapture();
            isDragging = false;
            e.Handled = true;
        }
        protected override void OnMouseMove(MouseEventArgs e)
        {
            base.OnMouseMove(e);
            if (!isDragging) return;
            System.Drawing.Point dragTo = System.Windows.Forms.Cursor.Position;
            double dx = dragTo.X - dragFrom.X;
            double dy = dragTo.Y - dragFrom.Y;
            System.Windows.Forms.Cursor.Position = dragFrom;
            // If there's no camera then there's nothing to update from here.
            if (Camera == null) return;
            // Detect modifier keys.
            bool isPressedLeftControl = Keyboard.IsKeyDown(Key.LeftCtrl);
            bool isPressedLeftShift = Keyboard.IsKeyDown(Key.LeftShift);
            // Process camera motion with modifier keys.
            if (isPressedLeftShift && isPressedLeftControl)
            {
                // Truck Mode (CTRL + SHIFT).
                Camera.Object.TranslatePost(new Vector3D(0, 0, dy * -0.05));
            }
            else if (!isPressedLeftShift && isPressedLeftControl)
            {
                // Rotate Mode (CTRL Only)
                Camera.Object.RotatePre(new Quaternion(new Vector3D(0, 1, 0), dx * 0.05));
                Camera.Object.RotatePost(new Quaternion(new Vector3D(1, 0, 0), dy * 0.05));
            }
            else if (!isPressedLeftShift && !isPressedLeftControl)
            {
                // Translation Mode (no modifier keys).
                Camera.Object.TranslatePost(new Vector3D(dx * -0.05, dy * 0.05, 0));
            }
            InvalidateVisual();
            e.Handled = true;
        }
        protected override void OnMouseWheel(MouseWheelEventArgs e)
        {
            base.OnMouseWheel(e);
            Camera.Object.TranslatePost(new Vector3D(0, 0, e.Delta * 0.01));
            e.Handled = true;
        }
        bool isDragging = false;
        System.Drawing.Point dragFrom;
        #endregion
        #region - Section : Static Helpers -
        public static void DrawPoints(DrawingContext drawingContext, IEnumerable<Vector4D> points)
        {
            var pen = new Pen(Brushes.Black, 1);
            foreach (var p in points)
            {
                drawingContext.DrawLine(pen, new Point(p.X - 2, p.Y - 2), new Point(p.X + 2, p.Y + 2));
                drawingContext.DrawLine(pen, new Point(p.X + 2, p.Y - 2), new Point(p.X - 2, p.Y + 2));
            }
        }
        public static void DrawWireframe(DrawingContext drawingContext, IEnumerable<Vector4D> lines)
        {
            DrawWireframe(drawingContext, lines, new Pen(Brushes.Black, 1));
        }
        public static void DrawWireframe(DrawingContext drawingContext, IEnumerable<Vector4D> lines, Pen pen)
        {
            var iter = lines.GetEnumerator();
            while (iter.MoveNext())
            {
                var P0 = iter.Current;
                if (!iter.MoveNext())
                {
                    break;
                }
                var P1 = iter.Current;
                drawingContext.DrawLine(pen, new Point(P0.X, P0.Y), new Point(P1.X, P1.Y));
            }
        }
        public static void DrawTriangles(DrawingContext drawingContext, IEnumerable<Vector4D> triangles)
        {
            DrawTriangles(drawingContext, triangles, new Pen(Brushes.Black, 1));
        }
        public static void DrawTriangles(DrawingContext drawingContext, IEnumerable<Vector4D> triangles, Pen pen)
        {
            var iter = triangles.GetEnumerator();
            while (iter.MoveNext())
            {
                var P0 = iter.Current;
                if (!iter.MoveNext())
                {
                    break;
                }
                var P1 = iter.Current;
                if (!iter.MoveNext())
                {
                    break;
                }
                var P2 = iter.Current;
                drawingContext.DrawLine(pen, new Point(P0.X, P0.Y), new Point(P1.X, P1.Y));
                drawingContext.DrawLine(pen, new Point(P1.X, P1.Y), new Point(P2.X, P2.Y));
                drawingContext.DrawLine(pen, new Point(P2.X, P2.Y), new Point(P0.X, P0.Y));
            }
        }
        public static void DrawBitmap(DrawingContext drawingContext, IEnumerable<PixelBgra32> pixels, double actualWidth, double actualHeight, int pixelWidth, int pixelHeight)
        {
            if (pixelWidth <= 0 || pixelHeight <= 0) return;
            WriteableBitmap bitmap = new WriteableBitmap(pixelWidth, pixelHeight, 0, 0, PixelFormats.Bgra32, null);
            bitmap.Lock();
            unsafe
            {
                int bufferstride = bitmap.BackBufferStride;
                byte* bufferptr = (byte*)bitmap.BackBuffer.ToPointer();
                foreach (var pixel in pixels)
                {
                    if (pixel.X < pixelWidth && pixel.Y < pixelHeight)
                    {
                        *(uint*)(bufferptr + 4 * pixel.X + bufferstride * pixel.Y) = pixel.Color;
                    }
                }
            }
            bitmap.AddDirtyRect(new Int32Rect(0, 0, pixelWidth, pixelHeight));
            bitmap.Unlock();
            drawingContext.DrawImage(bitmap, new Rect(0, 0, actualWidth, actualHeight));
        }
        public static void DrawGrid(DrawingContext drawingContext, double actualWidth, double actualHeight, int divisionsX, int divisionsY)
        {
            DrawGrid(drawingContext, actualWidth, actualHeight, divisionsX, divisionsY, new Pen(Brushes.Black, 1));
        }
        public static void DrawGrid(DrawingContext drawingContext, double actualWidth, double actualHeight, int divisionsX, int divisionsY, Pen pen)
        {
            for (int x = 0; x <= divisionsX; ++x)
            {
                drawingContext.DrawLine(pen, new Point(x * actualWidth / divisionsX, 0), new Point(x * actualWidth / divisionsX, actualHeight));
            }
            for (int y = 0; y <= divisionsY; ++y)
            {
                drawingContext.DrawLine(pen, new Point(0, y * actualHeight / divisionsY), new Point(actualWidth, y * actualHeight / divisionsY));
            }
        }
        #endregion
    }
    class FigurePointIntro : Figure3DBase
    {
        public FigurePointIntro()
        {
            Camera.Object.Transform = MathHelp.CreateMatrixLookAt(new Vector3D(10, 10, -20), new Vector3D(0, 0, 0), new Vector3D(0, 1, 0));
        }
        protected override void RenderFigure(DrawingContext drawingContext)
        {
            var vertexsource3 = Pipeline.SceneToPoints(Scene);
            var vertexsource4 = Pipeline.Vector3ToVector4(vertexsource3);
            var vertexclipspace = Pipeline.Transform(vertexsource4, MVP);
            var vertexclipped = Pipeline.ClipPoint(vertexclipspace);
            var vertexh = Pipeline.HomogeneousDivide(vertexclipped);
            var points = Pipeline.TransformToScreen(vertexh, ActualWidth, ActualHeight);
            Figure3DBase.DrawPoints(drawingContext, points);
        }
    }
    class FigurePointNegativeW : Figure3DBase
    {
        public FigurePointNegativeW()
        {
            Camera.Object.Transform = MathHelp.CreateMatrixLookAt(new Vector3D(2, 2, 0), new Vector3D(-10, 0, 10), new Vector3D(0, 1, 0));
        }
        protected override void RenderFigure(DrawingContext drawingContext)
        {
            var vertexsource3 = Pipeline.SceneToPoints(Scene);
            var vertexsource4 = Pipeline.Vector3ToVector4(vertexsource3);
            var vertexclipspace = Pipeline.Transform(vertexsource4, MVP);
            var vertexh = Pipeline.HomogeneousDivide(vertexclipspace);
            var points = Pipeline.TransformToScreen(vertexh, ActualWidth, ActualHeight);
            Figure3DBase.DrawPoints(drawingContext, points);
        }
    }
    class FigureWireframeIntro : Figure3DBase
    {
        public FigureWireframeIntro()
        {
            Camera.Object.Transform = MathHelp.CreateMatrixLookAt(new Vector3D(10, 10, -20), new Vector3D(0, 0, 0), new Vector3D(0, 1, 0));
        }
        protected override void RenderFigure(DrawingContext drawingContext)
        {
            var vertexsource3 = Pipeline.SceneToLines(Scene);
            var vertexsource4 = Pipeline.Vector3ToVector4(vertexsource3);
            var vertexclipspace = Pipeline.Transform(vertexsource4, MVP);
            var vertexclipped = Pipeline.ClipLine(vertexclipspace);
            var vertexh = Pipeline.HomogeneousDivide(vertexclipped);
            var lines = Pipeline.TransformToScreen(vertexh, ActualWidth, ActualHeight);
            Figure3DBase.DrawWireframe(drawingContext, lines);
        }
    }
    class FigureWireframeNegativeW : Figure3DBase
    {
        public FigureWireframeNegativeW()
        {
            Camera.Object.Transform = MathHelp.CreateMatrixLookAt(new Vector3D(2, 2, 0), new Vector3D(-10, 0, 10), new Vector3D(0, 1, 0));
        }
        protected override void RenderFigure(DrawingContext drawingContext)
        {
            var vertexsource3 = Pipeline.SceneToLines(Scene);
            var vertexsource4 = Pipeline.Vector3ToVector4(vertexsource3);
            var vertexclipspace = Pipeline.Transform(vertexsource4, MVP);
            var vertexh = Pipeline.HomogeneousDivide(vertexclipspace);
            var lines = Pipeline.TransformToScreen(vertexh, ActualWidth, ActualHeight);
            Figure3DBase.DrawWireframe(drawingContext, lines);
        }
    }
    class FigureWireframeClipped : Figure3DBase
    {
        public FigureWireframeClipped()
        {
            Camera.Object.Transform = MathHelp.CreateMatrixLookAt(new Vector3D(2, 2, 0), new Vector3D(-10, 0, 10), new Vector3D(0, 1, 0));
        }
        protected override void RenderFigure(DrawingContext drawingContext)
        {
            var mvp = MathHelp.Invert(Camera.Object.Transform);
            mvp = MathHelp.Multiply(mvp, Perspective.CreateProjection(0.01, 100.0, 60.0 * Math.PI / 180.0, 60.0 * Math.PI / 180.0));
            mvp = MathHelp.Multiply(mvp, Perspective.AspectCorrectFit(ActualWidth, ActualHeight));
            var vertexsource3 = Pipeline.SceneToLines(Scene);
            var vertexsource4 = Pipeline.Vector3ToVector4(vertexsource3);
            var vertexclipspace = Pipeline.Transform(vertexsource4, mvp);
            var vertexclipped = Pipeline.ClipLine(vertexclipspace);
            var vertexh = Pipeline.HomogeneousDivide(vertexclipped);
            var lines = Pipeline.TransformToScreen(vertexh, ActualWidth, ActualHeight);
            Figure3DBase.DrawWireframe(drawingContext, lines);
        }
    }
    class FigureTriangleIntro : Figure3DBase
    {
        public FigureTriangleIntro()
        {
            Camera.Object.Transform = MathHelp.CreateMatrixLookAt(new Vector3D(10, 10, -20), new Vector3D(0, 0, 0), new Vector3D(0, 1, 0));
        }
        protected override void RenderFigure(DrawingContext drawingContext)
        {
            var vertexsource3 = Pipeline.SceneToTriangles(Scene);
            var vertexsource4 = Pipeline.Vector3ToVector4(vertexsource3);
            var vertexclipspace = Pipeline.Transform(vertexsource4, MVP);
            var vertexclipped = Pipeline.ClipTriangle(vertexclipspace);
            var vertexh = Pipeline.HomogeneousDivide(vertexclipped);
            var triangles = Pipeline.TransformToScreen(vertexh, ActualWidth, ActualHeight);
            Figure3DBase.DrawTriangles(drawingContext, triangles);
        }
    }
    class FigureTriangleNegativeW : Figure3DBase
    {
        public FigureTriangleNegativeW()
        {
            Camera.Object.Transform = MathHelp.CreateMatrixLookAt(new Vector3D(2, 2, 0), new Vector3D(-10, 0, 10), new Vector3D(0, 1, 0));
        }
        protected override void RenderFigure(DrawingContext drawingContext)
        {
            var vertexsource3 = Pipeline.SceneToTriangles(Scene);
            var vertexsource4 = Pipeline.Vector3ToVector4(vertexsource3);
            var vertexclipspace = Pipeline.Transform(vertexsource4, MVP);
            var vertexh = Pipeline.HomogeneousDivide(vertexclipspace);
            var triangles = Pipeline.TransformToScreen(vertexh, ActualWidth, ActualHeight);
            Figure3DBase.DrawTriangles(drawingContext, triangles);
        }
    }
    class FigureTriangleClipped : Figure3DBase
    {
        public FigureTriangleClipped()
        {
            Camera.Object.Transform = MathHelp.CreateMatrixLookAt(new Vector3D(2, 2, 0), new Vector3D(-10, 0, 10), new Vector3D(0, 1, 0));
        }
        protected override void RenderFigure(DrawingContext drawingContext)
        {
            var vertexsource3 = Pipeline.SceneToTriangles(Scene);
            var vertexsource4 = Pipeline.Vector3ToVector4(vertexsource3);
            var vertexclipspace = Pipeline.Transform(vertexsource4, MVP);
            var vertexclipped = Pipeline.ClipTriangle(vertexclipspace);
            var vertexh = Pipeline.HomogeneousDivide(vertexclipped);
            var triangles = Pipeline.TransformToScreen(vertexh, ActualWidth, ActualHeight);
            Figure3DBase.DrawTriangles(drawingContext, triangles);
        }
    }
    #endregion
    #region - Section : Drag Handle Figures -
    abstract class FigureDragShapeBase : FigureBase
    {
        protected Vector4D[] FigurePoints
        {
            get; set;
        }
        #region - Overrides : UIElement -
        protected override void OnMouseLeftButtonDown(MouseButtonEventArgs e)
        {
            base.OnMouseLeftButtonDown(e);
            var mousepos = e.GetPosition(this);
            double mx = mousepos.X / ActualWidth;
            double my = mousepos.Y / ActualHeight;
            var hittest = FigurePoints
                .Select((point, index) => new { Index = index, Distance = MathHelp.Length(new Vector2D(point.X - mx, point.Y - my)) })
                .OrderBy(x => x.Distance)
                .Where(x => x.Distance < 4)
                .ToArray();
            if (hittest.Length == 0) return;
            Focus();
            CaptureMouse();
            Mouse.OverrideCursor = Cursors.None;
            isDragging = true;
            dragPoint = hittest[0].Index;
            e.Handled = true;
            InvalidateVisual();
        }
        protected override void OnMouseLeftButtonUp(MouseButtonEventArgs e)
        {
            base.OnMouseLeftButtonUp(e);
            ReleaseMouseCapture();
            Mouse.OverrideCursor = null;
            isDragging = false;
            e.Handled = true;
            InvalidateVisual();
        }
        protected override void OnMouseMove(MouseEventArgs e)
        {
            base.OnMouseMove(e);
            if (!isDragging) return;
            var current = FigurePoints[dragPoint];
            var mousepos = e.GetPosition(this);
            FigurePoints[dragPoint] = new Vector4D(mousepos.X / ActualWidth, mousepos.Y / ActualHeight, current.Z, current.W);
            InvalidateVisual();
        }
        protected override void OnRender(DrawingContext drawingContext)
        {
            drawingContext.DrawRectangle(Brushes.Transparent, null, new Rect(0, 0, ActualWidth, ActualHeight));
            RenderFigure(drawingContext);
            var pen = new Pen(Brushes.Black, 1);
            foreach (var point in FigurePoints)
            {
                drawingContext.DrawRectangle(Brushes.LightGreen, pen, new Rect(point.X * ActualWidth - 4, point.Y * ActualHeight - 4, 8, 8));
            }
            if (isDragging)
            {
                var point = FigurePoints[dragPoint];
                drawingContext.DrawRectangle(null, pen, new Rect(point.X * ActualWidth - 8, point.Y * ActualHeight - 8, 16, 16));

            }
        }
        bool isDragging = false;
        int dragPoint = 0;
        #endregion
    }
    #endregion
    #region - Section : Clipping Figures -
    abstract class FigureTriangleClipping : Figure3DBase
    {
        protected override void RenderFigure(DrawingContext drawingContext)
        {
            var unclipped = GetTriangle();
            var mvp = Matrix3D.Identity;
            mvp = MathHelp.Multiply(mvp, Perspective.AspectCorrectFit(ActualWidth, ActualHeight));
            mvp = MathHelp.Multiply(mvp, MathHelp.CreateMatrixScale(0.5, 0.5, 0.5));
            {
                var transformed = Pipeline.Transform(unclipped, mvp);
                var primitives = Pipeline.TransformToScreen(transformed, ActualWidth, ActualHeight);
                Figure3DBase.DrawTriangles(drawingContext, primitives, new Pen(Brushes.LightGray, 2));
            }
            {
                var clipframe = new Vector4D[]
                {
                    new Vector4D(-1, 1, 0.5, 1), new Vector4D(1, 1, 0.5, 1),
                    new Vector4D(1, 1, 0.5, 1), new Vector4D(1, -1, 0.5, 1),
                    new Vector4D(1, -1, 0.5, 1), new Vector4D(-1, -1, 0.5, 1),
                    new Vector4D(-1, -1, 0.5, 1), new Vector4D(-1, 1, 0.5, 1),
                };
                var transformed = Pipeline.Transform(clipframe, mvp);
                var primitives = Pipeline.TransformToScreen(transformed, ActualWidth, ActualHeight);
                Figure3DBase.DrawWireframe(drawingContext, primitives, new Pen(Brushes.LightGray, 1));
            }
            {
                var clipped = Pipeline.ClipTriangle(unclipped);
                var transformed = Pipeline.Transform(clipped, mvp);
                var primitives = Pipeline.TransformToScreen(transformed, ActualWidth, ActualHeight);
                Figure3DBase.DrawTriangles(drawingContext, primitives, new Pen(Brushes.Black, 1));
            }
        }
        protected abstract Vector4D[] GetTriangle();
    }
    class FigureTriangleClippingNone : FigureTriangleClipping
    {
        protected override Vector4D[] GetTriangle()
        {
            return new Vector4D[] { new Vector4D(0, 0.9, 0.5, 1), new Vector4D(-0.9, -0.9, 0.5, 1), new Vector4D(0.9, -0.9, 0.5, 1) };
        }
    }
    class FigureTriangleClipping0 : FigureTriangleClipping
    {
        protected override Vector4D[] GetTriangle()
        {
            return new Vector4D[] { new Vector4D(1.5, 0.9, 0.5, 1), new Vector4D(1.75, -0.9, 0.5, 1), new Vector4D(1.5, -0.9, 0.5, 1) };
        }
    }
    class FigureTriangleClipping1 : FigureTriangleClipping
    {
        protected override Vector4D[] GetTriangle()
        {
            return new Vector4D[] { new Vector4D(0, 0.9, 0.5, 1), new Vector4D(-0.9, -1.25, 0.5, 1), new Vector4D(0.9, -1.5, 0.5, 1) };
        }
    }
    class FigureTriangleClipping2 : FigureTriangleClipping
    {
        protected override Vector4D[] GetTriangle()
        {
            return new Vector4D[] { new Vector4D(-0.9, 0.25, 0.5, 1), new Vector4D(0.9, 0.5, 0.5, 1), new Vector4D(0, -1.5, 0.5, 1) };
        }
    }
    class FigureTriangleClippingMany : FigureTriangleClipping
    {
        protected override Vector4D[] GetTriangle()
        {
            return new Vector4D[] { new Vector4D(0, 0.9, 0.5, 1), new Vector4D(1.5, -1.25, 0.5, 1), new Vector4D(-1.5, -0.5, 0.5, 1) };
        }
    }
    #endregion
    #region - Section : Rasterization Figures -
    abstract class FigureRasterBase : FigureDragShapeBase
    {
        public FigureRasterBase()
        {
            RenderOptions.SetBitmapScalingMode(this, BitmapScalingMode.NearestNeighbor);
        }
        protected const int pixelWidth = 16;
        protected const int pixelHeight = 16;
    }
    class FigureRasterPoints : FigureRasterBase
    {
        public FigureRasterPoints()
        {
            FigurePoints = new Vector4D[]
            {
                new Vector4D(1.4 / pixelWidth, 1.1 / pixelHeight, 0.5, 1),
                new Vector4D((pixelWidth - 1.2) / pixelWidth, (pixelHeight - 2.2) / pixelHeight, 1.5, 1)
            };
        }
        protected override void RenderFigure(DrawingContext drawingContext)
        {
            var points = new Vector4D[] { FigurePoints[0], FigurePoints[1] };
            var scaled = Pipeline.Transform(points, MathHelp.CreateMatrixScale(pixelWidth, pixelHeight, 1));
            var pixels = Pipeline.RasterizePoint(scaled);
            Figure3DBase.DrawBitmap(drawingContext, pixels, ActualWidth, ActualHeight, pixelWidth, pixelHeight);
            Figure3DBase.DrawGrid(drawingContext, ActualWidth, ActualHeight, pixelWidth, pixelHeight, new Pen(Brushes.LightGray, 1));
        }
    }
    class FigureRasterLines : FigureRasterBase
    {
        public FigureRasterLines()
        {
            FigurePoints = new Vector4D[]
            {
                new Vector4D(0.5 / pixelWidth, 0.9 / pixelHeight, 0.5, 1),
                new Vector4D((pixelWidth - 2.1) / pixelWidth, (pixelHeight - 3.2) / pixelHeight, 0.5, 1),
                new Vector4D((pixelWidth - 1.4) / pixelWidth, 0.2 / pixelHeight, 0.5, 1),
                new Vector4D(0.1 / pixelWidth, (pixelHeight - 1.2) / pixelHeight, 0.5, 1)
            };
        }
        protected override void RenderFigure(DrawingContext drawingContext)
        {
            var lines = new Vector4D[] { FigurePoints[0], FigurePoints[1], FigurePoints[2], FigurePoints[3] };
            var scaled = Pipeline.Transform(lines, MathHelp.CreateMatrixScale(pixelWidth, pixelHeight, 1));
            var pixels = Pipeline.RasterizeLine(scaled);
            Figure3DBase.DrawBitmap(drawingContext, pixels, ActualWidth, ActualHeight, pixelWidth, pixelHeight);
            Figure3DBase.DrawGrid(drawingContext, ActualWidth, ActualHeight, pixelWidth, pixelHeight, new Pen(Brushes.LightGray, 1));
            var pen = new Pen(Brushes.Black, 2);
            var iter = Array.AsReadOnly<Vector4D>(lines).GetEnumerator();
            while (iter.MoveNext())
            {
                var P0 = iter.Current;
                if (!iter.MoveNext())
                {
                    break;
                }
                var P1 = iter.Current;
                drawingContext.DrawLine(pen, new Point(P0.X * ActualWidth, P0.Y * ActualHeight), new Point(P1.X * ActualWidth, P1.Y * ActualHeight));
            }
        }
    }
    class FigureRasterTriangles : FigureRasterBase
    {
        public FigureRasterTriangles()
        {
            FigurePoints = new Vector4D[]
            {
                new Vector4D((pixelWidth / 2 - 0.5) / pixelWidth, 1.2 / pixelHeight, 0.5, 1),
                new Vector4D((pixelWidth - 1.8) / pixelWidth, (pixelHeight - 3.2) / pixelHeight, 0.5, 1),
                new Vector4D(1.3 / pixelWidth, (pixelHeight - 1.4) / pixelHeight, 0.5, 1)
            };
        }
        protected override void RenderFigure(DrawingContext drawingContext)
        {
            var triangles = FigurePoints;
            var scaled = Pipeline.Transform(triangles, MathHelp.CreateMatrixScale(pixelWidth, pixelHeight, 1));
            var pixels = Pipeline.RasterizeTriangle(scaled);
            Figure3DBase.DrawBitmap(drawingContext, pixels, ActualWidth, ActualHeight, pixelWidth, pixelHeight);
            Figure3DBase.DrawGrid(drawingContext, ActualWidth, ActualHeight, pixelWidth, pixelHeight, new Pen(Brushes.LightGray, 1));
            var pen = new Pen(Brushes.Black, 2);
            var iter = Array.AsReadOnly<Vector4D>(triangles).GetEnumerator();
            while (iter.MoveNext())
            {
                var P0 = iter.Current;
                if (!iter.MoveNext())
                {
                    break;
                }
                var P1 = iter.Current;
                if (!iter.MoveNext())
                {
                    break;
                }
                var P2 = iter.Current;
                drawingContext.DrawLine(pen, new Point(P0.X * ActualWidth, P0.Y * ActualHeight), new Point(P1.X * ActualWidth, P1.Y * ActualHeight));
                drawingContext.DrawLine(pen, new Point(P1.X * ActualWidth, P1.Y * ActualHeight), new Point(P2.X * ActualWidth, P2.Y * ActualHeight));
                drawingContext.DrawLine(pen, new Point(P2.X * ActualWidth, P2.Y * ActualHeight), new Point(P0.X * ActualWidth, P0.Y * ActualHeight));
            }
        }
    }
    class FigureRasterScene : Figure3DBase
    {
        public FigureRasterScene()
        {
            Camera.Object.Transform = MathHelp.CreateMatrixLookAt(new Vector3D(10, 10, -20), new Vector3D(0, 0, 0), new Vector3D(0, 1, 0));
            RenderOptions.SetBitmapScalingMode(this, BitmapScalingMode.NearestNeighbor);
        }
        protected override void RenderFigure(DrawingContext drawingContext)
        {
            const int pixelWidth = 32;
            const int pixelHeight = 32;
            var vertexsource3 = Pipeline.SceneToTriangles(Scene);
            var vertexsource4 = Pipeline.Vector3ToVector4(vertexsource3);
            var vertexclipspace = Pipeline.Transform(vertexsource4, MVP);
            var vertexclipped = Pipeline.ClipTriangle(vertexclipspace);
            var vertexh = Pipeline.HomogeneousDivide(vertexclipped);
            Figure3DBase.DrawGrid(drawingContext, ActualWidth, ActualHeight, pixelWidth, pixelHeight, new Pen(Brushes.LightGray, 1));
            {
                var triangles = Pipeline.TransformToScreen(vertexh, pixelWidth, pixelHeight);
                var pixels = Pipeline.RasterizeTriangle(triangles);
                Figure3DBase.DrawBitmap(drawingContext, pixels, ActualWidth, ActualHeight, pixelWidth, pixelHeight);
            }
            {
                var triangles = Pipeline.TransformToScreen(vertexh, ActualWidth, ActualHeight);
                Figure3DBase.DrawTriangles(drawingContext, triangles);
            }
        }
    }
    #endregion
    #region - Section : Barycentric Figures -
    class FigureBarycentric : FigureDragShapeBase
    {
        public FigureBarycentric()
        {
            FigurePoints = new Vector4D[]
            {
                new Vector4D(0.5, 0, 0.5, 1.0),
                new Vector4D(1.0, 1, 0.5, 1.0),
                new Vector4D(0.0, 1, 0.5, 1.0),
                new Vector4D(0.5, 0.5, 0.5, 1.0),
            };
        }
        protected override void RenderFigure(DrawingContext drawingContext)
        {
            // Compute all values.
            var P0v4 = FigurePoints[0];
            var P1v4 = FigurePoints[1];
            var P2v4 = FigurePoints[2];
            var Bav4 = FigurePoints[3];
            var P0 = new Vector3D(P0v4.X, P0v4.Y, P0v4.Z);
            var P1 = new Vector3D(P1v4.X, P1v4.Y, P1v4.Z);
            var P2 = new Vector3D(P2v4.X, P2v4.Y, P2v4.Z);
            var Ba = new Vector3D(Bav4.X, Bav4.Y, Bav4.Z);
            // Compute all values.
            Func<double, double> SIGN = v => v < 0 ? -1 : (v > 0 ? 1 : 0);
            var edge1 = P1 - P0;
            var edge2 = P2 - P1;
            var area = MathHelp.Length(MathHelp.Cross(edge1, edge2)) / 2;
            edge1 = P1 - Ba;
            edge2 = P2 - P1;
            var alphanorm = MathHelp.Cross(edge1, edge2);
            var alphaarea = SIGN(alphanorm.Z) * MathHelp.Length(alphanorm) / 2;
            var alpha = alphaarea / area;
            edge1 = P2 - Ba;
            edge2 = P0 - P2;
            var betanorm = MathHelp.Cross(edge1, edge2);
            var betaarea = SIGN(betanorm.Z) * MathHelp.Length(betanorm) / 2;
            var beta = betaarea / area;
            var gamma = 1 - alpha - beta;
            // Draw everything.
            var penEdge = new Pen(Brushes.DarkGray, 2);
            var penSpoke = new Pen(Brushes.LightGray, 1);
            // Draw triangle edges.
            drawingContext.DrawLine(penEdge, new Point(P0.X * ActualWidth, P0.Y * ActualHeight), new Point(P1.X * ActualWidth, P1.Y * ActualHeight));
            drawingContext.DrawLine(penEdge, new Point(P1.X * ActualWidth, P1.Y * ActualHeight), new Point(P2.X * ActualWidth, P2.Y * ActualHeight));
            drawingContext.DrawLine(penEdge, new Point(P2.X * ActualWidth, P2.Y * ActualHeight), new Point(P0.X * ActualWidth, P0.Y * ActualHeight));
            // Draw barycentric "Spokes"
            drawingContext.DrawLine(penSpoke, new Point(Ba.X * ActualWidth, Ba.Y * ActualHeight), new Point(P0.X * ActualWidth, P0.Y * ActualHeight));
            drawingContext.DrawLine(penSpoke, new Point(Ba.X * ActualWidth, Ba.Y * ActualHeight), new Point(P1.X * ActualWidth, P1.Y * ActualHeight));
            drawingContext.DrawLine(penSpoke, new Point(Ba.X * ActualWidth, Ba.Y * ActualHeight), new Point(P2.X * ActualWidth, P2.Y * ActualHeight));
            var typeface = new Typeface(new FontFamily("Arial"), FontStyles.Normal, FontWeights.Normal, FontStretches.Normal);
            // Print total triangle area.
            var formattedtext = new FormattedText("Total Area = " + area.ToString("N3") + "\nɑ = " + alpha.ToString("N3") + "\nβ = " + beta.ToString("N3") + "\nɣ = " + gamma.ToString("N3"), CultureInfo.InvariantCulture, FlowDirection.LeftToRight, typeface, 12, Brushes.Black);
            drawingContext.DrawText(formattedtext, new Point(0, 0));
            // Print ɣ area.
            formattedtext = new FormattedText("ɣ = " + gamma.ToString("N3"), CultureInfo.InvariantCulture, FlowDirection.LeftToRight, typeface, 12, Brushes.Black);
            drawingContext.DrawText(formattedtext, new Point((Ba.X + P0.X + P1.X) * ActualWidth / 3 - formattedtext.Width / 2, (Ba.Y + P0.Y + P1.Y) * ActualHeight / 3 - formattedtext.Height / 2));
            // Print ɑ area.
            formattedtext = new FormattedText("Area = " + alphaarea.ToString("N3") + "\nɑ = " + alpha.ToString("N3"), CultureInfo.InvariantCulture, FlowDirection.LeftToRight, typeface, 12, Brushes.Black);
            drawingContext.DrawText(formattedtext, new Point((Ba.X + P1.X + P2.X) * ActualWidth / 3 - formattedtext.Width / 2, (Ba.Y + P1.Y + P2.Y) * ActualHeight / 3 - formattedtext.Height / 2));
            // Print β area.
            formattedtext = new FormattedText("Area = " + betaarea.ToString("N3") + "\nβ = " + beta.ToString("N3"), CultureInfo.InvariantCulture, FlowDirection.LeftToRight, typeface, 12, Brushes.Black);
            drawingContext.DrawText(formattedtext, new Point((Ba.X + P2.X + P0.X) * ActualWidth / 3 - formattedtext.Width / 2, (Ba.Y + P2.Y + P0.Y) * ActualHeight / 3 - formattedtext.Height / 2));
        }
    }
    class FigureBarycentric2Coeff : FigureDragShapeBase
    {
        public FigureBarycentric2Coeff()
        {
            FigurePoints = new Vector4D[]
            {
                new Vector4D(0.5, 0, 0.5, 1.0),
                new Vector4D(1.0, 1, 0.5, 1.0),
                new Vector4D(0.0, 1, 0.5, 1.0),
                new Vector4D(0.5, 0.5, 0.5, 1.0),
            };
        }
        protected override void RenderFigure(DrawingContext drawingContext)
        {
            // Compute all values.
            var P0v4 = FigurePoints[0];
            var P1v4 = FigurePoints[1];
            var P2v4 = FigurePoints[2];
            var Bav4 = FigurePoints[3];
            var P0 = new Vector3D(P0v4.X, P0v4.Y, P0v4.Z);
            var P1 = new Vector3D(P1v4.X, P1v4.Y, P1v4.Z);
            var P2 = new Vector3D(P2v4.X, P2v4.Y, P2v4.Z);
            var Ba = new Vector3D(Bav4.X, Bav4.Y, Bav4.Z);
            Func<double, double> SIGN = v => v < 0 ? -1 : (v > 0 ? 1 : 0);
            var edgeAlpha = P1 - P0;
            var edgeBeta = P2 - P0;
            var edgeSpoke = Ba - P0;
            var triangleNormal = MathHelp.Cross(edgeAlpha, edgeBeta);
            var triangleArea = MathHelp.Length(triangleNormal);
            var alphaNormal = MathHelp.Cross(edgeSpoke, edgeBeta);
            var alphaArea = SIGN(alphaNormal.Z) * MathHelp.Length(alphaNormal);
            var alphaValue = alphaArea / triangleArea;
            var betaNormal = MathHelp.Cross(edgeAlpha, edgeSpoke);
            var betaArea = SIGN(betaNormal.Z) * MathHelp.Length(betaNormal);
            var betaValue = betaArea / triangleArea;
            // Draw everything.
            var penEdge = new Pen(Brushes.DarkGray, 2);
            var penSpoke = new Pen(Brushes.LightGray, 1);
            var penAlpha = new Pen(Brushes.Red, 2);
            var penBeta = new Pen(Brushes.Green, 2);
            var typeface = new Typeface(new FontFamily("Arial"), FontStyles.Normal, FontWeights.Normal, FontStretches.Normal);
            // Draw triangle edges.
            drawingContext.DrawLine(penEdge, new Point(P0.X * ActualWidth, P0.Y * ActualHeight), new Point(P1.X * ActualWidth, P1.Y * ActualHeight));
            drawingContext.DrawLine(penEdge, new Point(P1.X * ActualWidth, P1.Y * ActualHeight), new Point(P2.X * ActualWidth, P2.Y * ActualHeight));
            drawingContext.DrawLine(penEdge, new Point(P2.X * ActualWidth, P2.Y * ActualHeight), new Point(P0.X * ActualWidth, P0.Y * ActualHeight));
            // Draw a vector walk across ɑ and β.
            var A = P0;
            var B = A + edgeAlpha * alphaValue;
            var C = B + edgeBeta * betaValue;
            drawingContext.DrawLine(penAlpha, new Point(A.X * ActualWidth, A.Y * ActualHeight), new Point(B.X * ActualWidth, B.Y * ActualHeight));
            drawingContext.DrawLine(penBeta, new Point(B.X * ActualWidth, B.Y * ActualHeight), new Point(C.X * ActualWidth, C.Y * ActualHeight));
            var formattedtext = new FormattedText("ɑ = " + alphaValue.ToString("N3"), CultureInfo.InvariantCulture, FlowDirection.LeftToRight, typeface, 16, Brushes.Red);
            drawingContext.DrawText(formattedtext, new Point((A.X + B.X) * ActualWidth / 2, (A.Y + B.Y) * ActualHeight / 2 - formattedtext.Height));
            formattedtext = new FormattedText("β = " + betaValue.ToString("N3"), CultureInfo.InvariantCulture, FlowDirection.LeftToRight, typeface, 16, Brushes.Green);
            drawingContext.DrawText(formattedtext, new Point((B.X + C.X) * ActualWidth / 2, (B.Y + C.Y) * ActualHeight / 2));
        }
    }
    class FigureBarycentricInterpolation : FigureDragShapeBase
    {
        const int pixelWidth = 64;
        const int pixelHeight = 64;
        public FigureBarycentricInterpolation()
        {
            RenderOptions.SetBitmapScalingMode(this, BitmapScalingMode.NearestNeighbor);
            FigurePoints = new Vector4D[]
            {
                new Vector4D(0.5, 0, 0.5, 1.0),
                new Vector4D(1.0, 1, 0.5, 1.0),
                new Vector4D(0.0, 1, 0.5, 1.0)
            };
        }
        protected override void RenderFigure(DrawingContext drawingContext)
        {
            var P0v4 = FigurePoints[0];
            var P1v4 = FigurePoints[1];
            var P2v4 = FigurePoints[2];
            var P0 = new Vector3D(P0v4.X, P0v4.Y, P0v4.Z);
            var P1 = new Vector3D(P1v4.X, P1v4.Y, P1v4.Z);
            var P2 = new Vector3D(P2v4.X, P2v4.Y, P2v4.Z);
            Figure3DBase.DrawBitmap(drawingContext, Rasterize(), ActualWidth, ActualHeight, pixelWidth, pixelHeight);
            var penEdge = new Pen(Brushes.DarkGray, 2);
            drawingContext.DrawLine(penEdge, new Point(P0.X * ActualWidth, P0.Y * ActualHeight), new Point(P1.X * ActualWidth, P1.Y * ActualHeight));
            drawingContext.DrawLine(penEdge, new Point(P1.X * ActualWidth, P1.Y * ActualHeight), new Point(P2.X * ActualWidth, P2.Y * ActualHeight));
            drawingContext.DrawLine(penEdge, new Point(P2.X * ActualWidth, P2.Y * ActualHeight), new Point(P0.X * ActualWidth, P0.Y * ActualHeight));
        }
        IEnumerable<PixelBgra32> Rasterize()
        {
            var P0v4 = FigurePoints[0];
            var P1v4 = FigurePoints[1];
            var P2v4 = FigurePoints[2];
            var P0 = new Vector3D(P0v4.X, P0v4.Y, 0);
            var P1 = new Vector3D(P1v4.X, P1v4.Y, 0);
            var P2 = new Vector3D(P2v4.X, P2v4.Y, 0);
            var edgeAlpha = P1 - P0;
            var edgeBeta = P2 - P0;
            var triangleNormal = MathHelp.Cross(edgeAlpha, edgeBeta);
            var triangleArea = MathHelp.Length(triangleNormal);
            var N0 = MathHelp.Cross(triangleNormal, P1 - P0);
            var N1 = MathHelp.Cross(triangleNormal, P2 - P1);
            var N2 = MathHelp.Cross(triangleNormal, P0 - P2);
            var D0 = MathHelp.Dot(P0, N0);
            var D1 = MathHelp.Dot(P1, N1);
            var D2 = MathHelp.Dot(P2, N2);
            for (ushort y = 0; y < pixelHeight; ++y)
            {
                for (ushort x = 0; x < pixelWidth; ++x)
                {
                    var Ba = new Vector3D((x + 0.5) / pixelWidth, (y + 0.5) / pixelHeight, 0);
                    var Dist0 = MathHelp.Dot(Ba, N0) - D0;
                    var Dist1 = MathHelp.Dot(Ba, N1) - D1;
                    var Dist2 = MathHelp.Dot(Ba, N2) - D2;
                    if (Dist0 >= 0 && Dist1 >= 0 && Dist2 >= 0)
                    {
                        var edgeSpoke = Ba - P0;
                        var alphaNormal = MathHelp.Cross(edgeSpoke, edgeBeta);
                        var alphaArea = MathHelp.Length(alphaNormal);
                        var alphaValue = alphaArea / triangleArea;
                        var betaNormal = MathHelp.Cross(edgeAlpha, edgeSpoke);
                        var betaArea = MathHelp.Length(betaNormal);
                        var betaValue = betaArea / triangleArea;
                        double r = 1 - alphaValue - betaValue;
                        double g = alphaValue;
                        double b = betaValue;
                        uint color = ((uint)(r * 255) << 16) | ((uint)(g * 255) << 8) | ((uint)(b * 255) << 0) | 0xFF000000;
                        yield return new PixelBgra32 { X = x, Y = y, Color = color };
                    }
                }
            }
        }
    }
    #endregion
    #region - Section : Homogeneous Rasterization Figures -
    class FigureHomogeneousRasterization : FigureDragShapeBase
    {
        const int pixelWidth = 64;
        const int pixelHeight = 64;
        public FigureHomogeneousRasterization()
        {
            RenderOptions.SetBitmapScalingMode(this, BitmapScalingMode.NearestNeighbor);
            FigurePoints = new Vector4D[]
            {
                new Vector4D(0.5, 0, 0.5, 1.0),
                new Vector4D(1.0, 1, 0.5, 1.0),
                new Vector4D(0.0, 1, 0.5, 1.0)
            };
        }
        protected override void RenderFigure(DrawingContext drawingContext)
        {
            var P0v4 = FigurePoints[0];
            var P1v4 = FigurePoints[1];
            var P2v4 = FigurePoints[2];
            var P0 = new Vector3D(P0v4.X, P0v4.Y, P0v4.Z);
            var P1 = new Vector3D(P1v4.X, P1v4.Y, P1v4.Z);
            var P2 = new Vector3D(P2v4.X, P2v4.Y, P2v4.Z);
            Figure3DBase.DrawBitmap(drawingContext, Rasterize(), ActualWidth, ActualHeight, pixelWidth, pixelHeight);
            var penEdge = new Pen(Brushes.DarkGray, 2);
            drawingContext.DrawLine(penEdge, new Point(P0.X * ActualWidth, P0.Y * ActualHeight), new Point(P1.X * ActualWidth, P1.Y * ActualHeight));
            drawingContext.DrawLine(penEdge, new Point(P1.X * ActualWidth, P1.Y * ActualHeight), new Point(P2.X * ActualWidth, P2.Y * ActualHeight));
            drawingContext.DrawLine(penEdge, new Point(P2.X * ActualWidth, P2.Y * ActualHeight), new Point(P0.X * ActualWidth, P0.Y * ActualHeight));
        }
        IEnumerable<PixelBgra32> Rasterize()
        {
            /*
            var triangles = new Triangle<Vector4D>[]
            {
                new Triangle<Vector4D> { P0 = FigurePoints[0], P1 = FigurePoints[1], P2 = FigurePoints[2], Color = 0xFF00FFFF }
            };
            return Pipeline.RasterizeHomogeneous(triangles, pixelWidth, pixelHeight);
            */
            var P0v4 = FigurePoints[0];
            var P1v4 = FigurePoints[1];
            var P2v4 = FigurePoints[2];
            Matrix3D Minv = MathHelp.Invert(new Matrix3D(P0v4.X, P1v4.X, P2v4.X, 0, P0v4.Y, P1v4.Y, P2v4.Y, 0, P0v4.W, P1v4.W, P2v4.W, 0, 0, 0, 0, 1));
            Vector3D interp = MathHelp.TransformVector(Minv, new Vector3D(1, 1, 1));
            for (ushort y = 0; y < pixelHeight; ++y)
            {
                for (ushort x = 0; x < pixelWidth; ++x)
                {
                    double px = (x + 0.5) / pixelWidth;
                    double py = (y + 0.5) / pixelHeight;
                    double w = interp.X * px + interp.Y * py + interp.Z;
                    double a = Minv.M11 * px + Minv.M12 * py + Minv.M13;
                    double b = Minv.M21 * px + Minv.M22 * py + Minv.M23;
                    double c = Minv.M31 * px + Minv.M32 * py + Minv.M33;
                    if (a > 0 && b > 0 && c > 0)
                    {
                        uint color = ((uint)(a * 255) << 16) | ((uint)(b * 255) << 8) | ((uint)(c * 255) << 0) | 0xFF000000;
                        yield return new PixelBgra32 { X = x, Y = y, Color = color };
                    }
                }
            }
        }
    }
    class FigureRasterSceneHomogeneous : Figure3DBase
    {
        public FigureRasterSceneHomogeneous()
        {
            Camera.Object.Transform = MathHelp.CreateMatrixLookAt(new Vector3D(10, 10, -20), new Vector3D(0, 0, 0), new Vector3D(0, 1, 0));
            RenderOptions.SetBitmapScalingMode(this, BitmapScalingMode.NearestNeighbor);
        }
        protected override void RenderFigure(DrawingContext drawingContext)
        {
            const int pixelWidth = 64;
            const int pixelHeight = 64;
            {
                var A = Pipeline.SceneToTriangles(Scene);
                var B = Pipeline.Vector3ToVector4(A);
                var C = Pipeline.Transform(B, MVP);
                var D = Pipeline.RasterizeHomogeneous(C, pixelWidth, pixelHeight);
                Figure3DBase.DrawBitmap(drawingContext, D, ActualWidth, ActualHeight, pixelWidth, pixelHeight);
            }
            {
                var A = Pipeline.SceneToLines(Scene);
                var B = Pipeline.Vector3ToVector4(A);
                var C = Pipeline.Transform(B, MVP);
                var D = Pipeline.ClipLine(C);
                var E = Pipeline.HomogeneousDivide(D);
                var F = Pipeline.TransformToScreen(E, ActualWidth, ActualHeight);
                Figure3DBase.DrawWireframe(drawingContext, F);
            }
        }
    }
    #endregion
}