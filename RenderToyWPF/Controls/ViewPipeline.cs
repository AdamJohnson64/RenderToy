using System;
using System.Collections.Generic;
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
        public static void DrawPoints(DrawingContext drawingContext, IEnumerable<PipelineModel.Vertex<Vector4D>> points)
        {
            var pen = new Pen(Brushes.Black, 1);
            foreach (var p in points)
            {
                drawingContext.DrawLine(pen, new Point(p.Position.X - 2, p.Position.Y - 2), new Point(p.Position.X + 2, p.Position.Y + 2));
                drawingContext.DrawLine(pen, new Point(p.Position.X + 2, p.Position.Y - 2), new Point(p.Position.X - 2, p.Position.Y + 2));
            }
        }
        public static void DrawWireframe(DrawingContext drawingContext, IEnumerable<PipelineModel.Line<Vector4D>> lines)
        {
            DrawWireframe(drawingContext, lines, new Pen(Brushes.Black, 1));
        }
        public static void DrawWireframe(DrawingContext drawingContext, IEnumerable<PipelineModel.Line<Vector4D>> lines, Pen pen)
        {
            foreach (var l in lines)
            {
                drawingContext.DrawLine(pen, new Point(l.P0.X, l.P0.Y), new Point(l.P1.X, l.P1.Y));
            }
        }
        public static void DrawTriangles(DrawingContext drawingContext, IEnumerable<PipelineModel.Triangle<Vector4D>> triangles)
        {
            DrawTriangles(drawingContext, triangles, new Pen(Brushes.Black, 1));
        }
        public static void DrawTriangles(DrawingContext drawingContext, IEnumerable<PipelineModel.Triangle<Vector4D>> triangles, Pen pen)
        {
            foreach (var t in triangles)
            {
                drawingContext.DrawLine(pen, new Point(t.P0.X, t.P0.Y), new Point(t.P1.X, t.P1.Y));
                drawingContext.DrawLine(pen, new Point(t.P1.X, t.P1.Y), new Point(t.P2.X, t.P2.Y));
                drawingContext.DrawLine(pen, new Point(t.P2.X, t.P2.Y), new Point(t.P0.X, t.P0.Y));
            }
        }
        public static void DrawBitmap(DrawingContext drawingContext, IEnumerable<PipelineModel.PixelBgra32> pixels, double actualWidth, double actualHeight, int pixelWidth, int pixelHeight)
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
            var vertexsource3 = Pipeline.SceneToVertices(Scene);
            var vertexsource4 = Pipeline.Vector3ToVector4(vertexsource3);
            var vertexclipspace = Pipeline.Transform(vertexsource4, MVP);
            var vertexclipped = Pipeline.Clip(vertexclipspace);
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
            var vertexsource3 = Pipeline.SceneToVertices(Scene);
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
            var vertexclipped = Pipeline.Clip(vertexclipspace);
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
            var vertexclipped = Pipeline.Clip(vertexclipspace);
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
            var vertexclipped = Pipeline.Clip(vertexclipspace);
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
            var vertexclipped = Pipeline.Clip(vertexclipspace);
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
            var unclipped = new Triangle<Vector4D>[] { GetTriangle() };
            var mvp = Matrix3D.Identity;
            mvp = MathHelp.Multiply(mvp, Perspective.AspectCorrectFit(ActualWidth, ActualHeight));
            mvp = MathHelp.Multiply(mvp, MathHelp.CreateMatrixScale(0.5, 0.5, 0.5));
            {
                var transformed = Pipeline.Transform(unclipped, mvp);
                var primitives = Pipeline.TransformToScreen(transformed, ActualWidth, ActualHeight);
                Figure3DBase.DrawTriangles(drawingContext, primitives, new Pen(Brushes.LightGray, 2));
            }
            {
                var clipframe = new Line<Vector4D>[]
                {
                    new Line<Vector4D> { P0 = new Vector4D(-1, 1, 0.5, 1), P1 = new Vector4D(1, 1, 0.5, 1) },
                    new Line<Vector4D> { P0 = new Vector4D(1, 1, 0.5, 1), P1 = new Vector4D(1, -1, 0.5, 1) },
                    new Line<Vector4D> { P0 = new Vector4D(1, -1, 0.5, 1), P1 = new Vector4D(-1, -1, 0.5, 1) },
                    new Line<Vector4D> { P0 = new Vector4D(-1, -1, 0.5, 1), P1 = new Vector4D(-1, 1, 0.5, 1) },
                };
                var transformed = Pipeline.Transform(clipframe, mvp);
                var primitives = Pipeline.TransformToScreen(transformed, ActualWidth, ActualHeight);
                Figure3DBase.DrawWireframe(drawingContext, primitives, new Pen(Brushes.LightGray, 1));
            }
            {
                var clipped = Pipeline.Clip(unclipped);
                var transformed = Pipeline.Transform(clipped, mvp);
                var primitives = Pipeline.TransformToScreen(transformed, ActualWidth, ActualHeight);
                Figure3DBase.DrawTriangles(drawingContext, primitives, new Pen(Brushes.Black, 1));
            }
        }
        protected abstract Triangle<Vector4D> GetTriangle();
    }
    class FigureTriangleClippingNone : FigureTriangleClipping
    {
        protected override Triangle<Vector4D> GetTriangle()
        {
            return new Triangle<Vector4D> { P0 = new Vector4D(0, 0.9, 0.5, 1), P1 = new Vector4D(-0.9, -0.9, 0.5, 1), P2 = new Vector4D(0.9, -0.9, 0.5, 1) };
        }
    }
    class FigureTriangleClipping0 : FigureTriangleClipping
    {
        protected override Triangle<Vector4D> GetTriangle()
        {
            return new Triangle<Vector4D> { P0 = new Vector4D(1.5, 0.9, 0.5, 1), P1 = new Vector4D(1.75, -0.9, 0.5, 1), P2 = new Vector4D(1.5, -0.9, 0.5, 1) };
        }
    }
    class FigureTriangleClipping1 : FigureTriangleClipping
    {
        protected override Triangle<Vector4D> GetTriangle()
        {
            return new Triangle<Vector4D> { P0 = new Vector4D(0, 0.9, 0.5, 1), P1 = new Vector4D(-0.9, -1.25, 0.5, 1), P2 = new Vector4D(0.9, -1.5, 0.5, 1) };
        }
    }
    class FigureTriangleClipping2 : FigureTriangleClipping
    {
        protected override Triangle<Vector4D> GetTriangle()
        {
            return new Triangle<Vector4D> { P0 = new Vector4D(-0.9, 0.25, 0.5, 1), P1 = new Vector4D(0.9, 0.5, 0.5, 1), P2 = new Vector4D(0, -1.5, 0.5, 1) };
        }
    }
    class FigureTriangleClippingMany : FigureTriangleClipping
    {
        protected override Triangle<Vector4D> GetTriangle()
        {
            return new Triangle<Vector4D> { P0 = new Vector4D(0, 0.9, 0.5, 1), P1 = new Vector4D(1.5, -1.25, 0.5, 1), P2 = new Vector4D(-1.5, -0.5, 0.5, 1) };
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
            var points = new PipelineModel.Vertex<Vector4D>[]
            {
                new Vertex<Vector4D> { Position = FigurePoints[0], Color = 0xFF00FFFF },
                new Vertex<Vector4D> { Position = FigurePoints[1], Color = 0xFF0000FF }
            };
            var scaled = PipelineModel.Pipeline.Transform(points, MathHelp.CreateMatrixScale(pixelWidth, pixelHeight, 1));
            var pixels = PipelineModel.Pipeline.Rasterize(scaled);
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
            var lines = new PipelineModel.Line<Vector4D>[]
            {
                new Line<Vector4D> { P0 = FigurePoints[0], P1 = FigurePoints[1], Color = 0xFF00FFFF },
                new Line<Vector4D> { P0 = FigurePoints[2], P1 = FigurePoints[3], Color = 0xFF0000FF }
            };
            var scaled = PipelineModel.Pipeline.Transform(lines, MathHelp.CreateMatrixScale(pixelWidth, pixelHeight, 1));
            var pixels = PipelineModel.Pipeline.Rasterize(scaled);
            Figure3DBase.DrawBitmap(drawingContext, pixels, ActualWidth, ActualHeight, pixelWidth, pixelHeight);
            Figure3DBase.DrawGrid(drawingContext, ActualWidth, ActualHeight, pixelWidth, pixelHeight, new Pen(Brushes.LightGray, 1));
            var pen = new Pen(Brushes.Black, 2);
            foreach (var line in lines)
            {
                drawingContext.DrawLine(pen, new Point(line.P0.X * ActualWidth, line.P0.Y * ActualHeight), new Point(line.P1.X * ActualWidth, line.P1.Y * ActualHeight));
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
            var triangles = new Triangle<Vector4D>[]
            {
                new Triangle<Vector4D> { P0 = FigurePoints[0], P1 = FigurePoints[1], P2 = FigurePoints[2], Color = 0xFF00FFFF }
            };
            var scaled = PipelineModel.Pipeline.Transform(triangles, MathHelp.CreateMatrixScale(pixelWidth, pixelHeight, 1));
            var pixels = PipelineModel.Pipeline.Rasterize(scaled);
            Figure3DBase.DrawBitmap(drawingContext, pixels, ActualWidth, ActualHeight, pixelWidth, pixelHeight);
            Figure3DBase.DrawGrid(drawingContext, ActualWidth, ActualHeight, pixelWidth, pixelHeight, new Pen(Brushes.LightGray, 1));
            var pen = new Pen(Brushes.Black, 2);
            foreach (var t in triangles)
            {
                drawingContext.DrawLine(pen, new Point(t.P0.X * ActualWidth, t.P0.Y * ActualHeight), new Point(t.P1.X * ActualWidth, t.P1.Y * ActualHeight));
                drawingContext.DrawLine(pen, new Point(t.P1.X * ActualWidth, t.P1.Y * ActualHeight), new Point(t.P2.X * ActualWidth, t.P2.Y * ActualHeight));
                drawingContext.DrawLine(pen, new Point(t.P2.X * ActualWidth, t.P2.Y * ActualHeight), new Point(t.P0.X * ActualWidth, t.P0.Y * ActualHeight));
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
            var vertexclipped = Pipeline.Clip(vertexclipspace);
            var vertexh = Pipeline.HomogeneousDivide(vertexclipped);
            Figure3DBase.DrawGrid(drawingContext, ActualWidth, ActualHeight, pixelWidth, pixelHeight, new Pen(Brushes.LightGray, 1));
            {
                var triangles = Pipeline.TransformToScreen(vertexh, pixelWidth, pixelHeight);
                var pixels = Pipeline.Rasterize(triangles);
                Figure3DBase.DrawBitmap(drawingContext, pixels, ActualWidth, ActualHeight, pixelWidth, pixelHeight);
            }
            {
                var triangles = Pipeline.TransformToScreen(vertexh, ActualWidth, ActualHeight);
                Figure3DBase.DrawTriangles(drawingContext, triangles);
            }
        }
    }
    #endregion
}