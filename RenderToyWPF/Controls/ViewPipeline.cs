using System;
using System.Collections.Generic;
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
    abstract class FigureBase : FrameworkElement
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
                scene.AddChild(new Node("Plane", new TransformMatrix(MathHelp.CreateMatrixScale(10, 10, 10)), new Plane(), StockMaterials.Black, StockMaterials.PlasticRed));
                return scene;
            }
        }
        #endregion
        #region - Section : Dependency Properties -
        public static DependencyProperty CameraProperty = DependencyProperty.Register("Camera", typeof(Camera), typeof(FigureBase));
        public Camera Camera
        {
            get { return (Camera)GetValue(CameraProperty); }
            set { SetValue(CameraProperty, value); }
        }
        #endregion
        #region - Section : Construction -
        public FigureBase()
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
    class FigurePointIntro : FigureBase
    {
        public FigurePointIntro()
        {
            Camera.Object.Transform = MathHelp.CreateMatrixLookAt(new Vector3D(10, 10, -20), new Vector3D(0, 0, 0), new Vector3D(0, 1, 0));
        }
        protected override void OnRender(DrawingContext drawingContext)
        {
            var vertexsource3 = Pipeline.SceneToVertices(Scene);
            var vertexsource4 = Pipeline.Vector3ToVector4(vertexsource3);
            var vertexclipspace = Pipeline.Transform(vertexsource4, MVP);
            var vertexclipped = Pipeline.Clip(vertexclipspace);
            var vertexh = Pipeline.HomogeneousDivide(vertexclipped);
            var points = Pipeline.TransformToScreen(vertexh, ActualWidth, ActualHeight);
            drawingContext.DrawRectangle(Brushes.Transparent, null, new Rect(0, 0, ActualWidth, ActualHeight));
            FigureBase.DrawPoints(drawingContext, points);
        }
    }
    class FigurePointNegativeW : FigureBase
    {
        public FigurePointNegativeW()
        {
            Camera.Object.Transform = MathHelp.CreateMatrixLookAt(new Vector3D(2, 2, 0), new Vector3D(-10, 0, 10), new Vector3D(0, 1, 0));
        }
        protected override void OnRender(DrawingContext drawingContext)
        {
            var vertexsource3 = Pipeline.SceneToVertices(Scene);
            var vertexsource4 = Pipeline.Vector3ToVector4(vertexsource3);
            var vertexclipspace = Pipeline.Transform(vertexsource4, MVP);
            var vertexh = Pipeline.HomogeneousDivide(vertexclipspace);
            var points = Pipeline.TransformToScreen(vertexh, ActualWidth, ActualHeight);
            drawingContext.DrawRectangle(Brushes.Transparent, null, new Rect(0, 0, ActualWidth, ActualHeight));
            FigureBase.DrawPoints(drawingContext, points);
        }
    }
    class FigureWireframeIntro : FigureBase
    {
        public FigureWireframeIntro()
        {
            Camera.Object.Transform = MathHelp.CreateMatrixLookAt(new Vector3D(10, 10, -20), new Vector3D(0, 0, 0), new Vector3D(0, 1, 0));
        }
        protected override void OnRender(DrawingContext drawingContext)
        {
            var vertexsource3 = Pipeline.SceneToLines(Scene);
            var vertexsource4 = Pipeline.Vector3ToVector4(vertexsource3);
            var vertexclipspace = Pipeline.Transform(vertexsource4, MVP);
            var vertexclipped = Pipeline.Clip(vertexclipspace);
            var vertexh = Pipeline.HomogeneousDivide(vertexclipped);
            var lines = Pipeline.TransformToScreen(vertexh, ActualWidth, ActualHeight);
            drawingContext.DrawRectangle(Brushes.Transparent, null, new Rect(0, 0, ActualWidth, ActualHeight));
            FigureBase.DrawWireframe(drawingContext, lines);
        }
    }
    class FigureWireframeNegativeW : FigureBase
    {
        public FigureWireframeNegativeW()
        {
            Camera.Object.Transform = MathHelp.CreateMatrixLookAt(new Vector3D(2, 2, 0), new Vector3D(-10, 0, 10), new Vector3D(0, 1, 0));
        }
        protected override void OnRender(DrawingContext drawingContext)
        {
            var vertexsource3 = Pipeline.SceneToLines(Scene);
            var vertexsource4 = Pipeline.Vector3ToVector4(vertexsource3);
            var vertexclipspace = Pipeline.Transform(vertexsource4, MVP);
            var vertexh = Pipeline.HomogeneousDivide(vertexclipspace);
            var lines = Pipeline.TransformToScreen(vertexh, ActualWidth, ActualHeight);
            drawingContext.DrawRectangle(Brushes.Transparent, null, new Rect(0, 0, ActualWidth, ActualHeight));
            FigureBase.DrawWireframe(drawingContext, lines);
        }
    }
    class FigureWireframeClipped : FigureBase
    {
        public FigureWireframeClipped()
        {
            Camera.Object.Transform = MathHelp.CreateMatrixLookAt(new Vector3D(2, 2, 0), new Vector3D(-10, 0, 10), new Vector3D(0, 1, 0));
        }
        protected override void OnRender(DrawingContext drawingContext)
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
            drawingContext.DrawRectangle(Brushes.Transparent, null, new Rect(0, 0, ActualWidth, ActualHeight));
            FigureBase.DrawWireframe(drawingContext, lines);
        }
    }
    class FigureTriangleIntro : FigureBase
    {
        public FigureTriangleIntro()
        {
            Camera.Object.Transform = MathHelp.CreateMatrixLookAt(new Vector3D(10, 10, -20), new Vector3D(0, 0, 0), new Vector3D(0, 1, 0));
        }
        protected override void OnRender(DrawingContext drawingContext)
        {
            var vertexsource3 = Pipeline.SceneToTriangles(Scene);
            var vertexsource4 = Pipeline.Vector3ToVector4(vertexsource3);
            var vertexclipspace = Pipeline.Transform(vertexsource4, MVP);
            var vertexclipped = Pipeline.Clip(vertexclipspace);
            var vertexh = Pipeline.HomogeneousDivide(vertexclipped);
            var triangles = Pipeline.TransformToScreen(vertexh, ActualWidth, ActualHeight);
            drawingContext.DrawRectangle(Brushes.Transparent, null, new Rect(0, 0, ActualWidth, ActualHeight));
            FigureBase.DrawTriangles(drawingContext, triangles);
        }
    }
    class FigureTriangleNegativeW : FigureBase
    {
        public FigureTriangleNegativeW()
        {
            Camera.Object.Transform = MathHelp.CreateMatrixLookAt(new Vector3D(2, 2, 0), new Vector3D(-10, 0, 10), new Vector3D(0, 1, 0));
        }
        protected override void OnRender(DrawingContext drawingContext)
        {
            var vertexsource3 = Pipeline.SceneToTriangles(Scene);
            var vertexsource4 = Pipeline.Vector3ToVector4(vertexsource3);
            var vertexclipspace = Pipeline.Transform(vertexsource4, MVP);
            var vertexh = Pipeline.HomogeneousDivide(vertexclipspace);
            var triangles = Pipeline.TransformToScreen(vertexh, ActualWidth, ActualHeight);
            drawingContext.DrawRectangle(Brushes.Transparent, null, new Rect(0, 0, ActualWidth, ActualHeight));
            FigureBase.DrawTriangles(drawingContext, triangles);
        }
    }
    abstract class FigureTriangleClipping : FigureBase
    {
        protected override void OnRender(DrawingContext drawingContext)
        {
            var unclipped = new Triangle<Vector4D>[] { GetTriangle() };
            var mvp = Matrix3D.Identity;
            mvp = MathHelp.Multiply(mvp, Perspective.AspectCorrectFit(ActualWidth, ActualHeight));
            mvp = MathHelp.Multiply(mvp, MathHelp.CreateMatrixScale(0.5, 0.5, 0.5));
            {
                var transformed = Pipeline.Transform(unclipped, mvp);
                var primitives = Pipeline.TransformToScreen(transformed, ActualWidth, ActualHeight);
                FigureBase.DrawTriangles(drawingContext, primitives, new Pen(Brushes.LightGray, 2));
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
                FigureBase.DrawWireframe(drawingContext, primitives, new Pen(Brushes.LightGray, 1));
            }
            {
                var clipped = Pipeline.Clip(unclipped);
                var transformed = Pipeline.Transform(clipped, mvp);
                var primitives = Pipeline.TransformToScreen(transformed, ActualWidth, ActualHeight);
                FigureBase.DrawTriangles(drawingContext, primitives, new Pen(Brushes.Black, 1));
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
    class FigureTriangleClipped : FigureBase
    {
        public FigureTriangleClipped()
        {
            Camera.Object.Transform = MathHelp.CreateMatrixLookAt(new Vector3D(2, 2, 0), new Vector3D(-10, 0, 10), new Vector3D(0, 1, 0));
        }
        protected override void OnRender(DrawingContext drawingContext)
        {
            var vertexsource3 = Pipeline.SceneToTriangles(Scene);
            var vertexsource4 = Pipeline.Vector3ToVector4(vertexsource3);
            var vertexclipspace = Pipeline.Transform(vertexsource4, MVP);
            var vertexclipped = Pipeline.Clip(vertexclipspace);
            var vertexh = Pipeline.HomogeneousDivide(vertexclipped);
            var triangles = Pipeline.TransformToScreen(vertexh, ActualWidth, ActualHeight);
            drawingContext.DrawRectangle(Brushes.Transparent, null, new Rect(0, 0, ActualWidth, ActualHeight));
            FigureBase.DrawTriangles(drawingContext, triangles);
        }
    }
    class FigureRasterBase : FigureBase
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
        protected override void OnRender(DrawingContext drawingContext)
        {
            var points = new PipelineModel.Vertex<Vector4D>[]
            {
                new Vertex<Vector4D> { Position = new Vector4D(1.4, 1.1, 0.5, 1), Color = 0xFF00FFFF },
                new Vertex<Vector4D> { Position = new Vector4D(pixelWidth - 1.2, pixelHeight - 2.2, 1.5, 1), Color = 0xFF0000FF }
            };
            var pixels = PipelineModel.Pipeline.Rasterize(points);
            FigureBase.DrawBitmap(drawingContext, pixels, ActualWidth, ActualHeight, pixelWidth, pixelHeight);
            FigureBase.DrawGrid(drawingContext, ActualWidth, ActualHeight, pixelWidth, pixelHeight, new Pen(Brushes.LightGray, 1));
            var pen = new Pen(Brushes.Black, 2);
            foreach (var point in points)
            {
                double x = point.Position.X * ActualWidth / pixelWidth;
                double y = point.Position.Y * ActualHeight / pixelHeight;
                drawingContext.DrawLine(pen, new Point(x - 4, y - 4), new Point(x + 4, y + 4));
                drawingContext.DrawLine(pen, new Point(x + 4, y - 4), new Point(x - 4, y + 4));
            }
        }
    }
    class FigureRasterLines : FigureRasterBase
    {
        protected override void OnRender(DrawingContext drawingContext)
        {
            var lines = new PipelineModel.Line<Vector4D>[]
            {
                new Line<Vector4D> { P0 = new Vector4D(0.5, 0.9, 0.5, 1), P1 = new Vector4D(pixelWidth - 2.1, pixelHeight - 3.2, 0.5, 1), Color = 0xFF00FFFF },
                new Line<Vector4D> { P0 = new Vector4D(pixelWidth - 1.4, 0.2, 0.5, 1), P1 = new Vector4D(0.1, pixelHeight - 1.2, 0.5, 1), Color = 0xFF0000FF }
            };
            var pixels = PipelineModel.Pipeline.Rasterize(lines);
            FigureBase.DrawBitmap(drawingContext, pixels, ActualWidth, ActualHeight, pixelWidth, pixelHeight);
            FigureBase.DrawGrid(drawingContext, ActualWidth, ActualHeight, pixelWidth, pixelHeight, new Pen(Brushes.LightGray, 1));
            var pen = new Pen(Brushes.Black, 2);
            foreach (var line in lines)
            {
                drawingContext.DrawLine(pen, new Point(line.P0.X * ActualWidth / pixelWidth, line.P0.Y * ActualHeight / pixelHeight), new Point(line.P1.X * ActualWidth / pixelWidth, line.P1.Y * ActualHeight / pixelHeight));
            }
        }
    }
    class FigureRasterTriangles : FigureRasterBase
    {
        protected override void OnRender(DrawingContext drawingContext)
        {
            var triangles = new Triangle<Vector4D>[]
            {
                new Triangle<Vector4D> { P0 = new Vector4D(pixelWidth / 2 - 0.5, 1.2, 0.5, 1), P1 = new Vector4D(pixelWidth - 1.8, pixelHeight - 3.2, 0.5, 1), P2 = new Vector4D(1.3, pixelHeight - 1.4, 0.5, 1), Color = 0xFF00FFFF }
            };
            var pixels = PipelineModel.Pipeline.Rasterize(triangles);
            FigureBase.DrawBitmap(drawingContext, pixels, ActualWidth, ActualHeight, pixelWidth, pixelHeight);
            FigureBase.DrawGrid(drawingContext, ActualWidth, ActualHeight, pixelWidth, pixelHeight, new Pen(Brushes.LightGray, 1));
            var pen = new Pen(Brushes.Black, 2);
            foreach (var t in triangles)
            {
                drawingContext.DrawLine(pen, new Point(t.P0.X * ActualWidth / pixelWidth, t.P0.Y * ActualHeight / pixelHeight), new Point(t.P1.X * ActualWidth / pixelWidth, t.P1.Y * ActualHeight / pixelHeight));
                drawingContext.DrawLine(pen, new Point(t.P1.X * ActualWidth / pixelWidth, t.P1.Y * ActualHeight / pixelHeight), new Point(t.P2.X * ActualWidth / pixelWidth, t.P2.Y * ActualHeight / pixelHeight));
                drawingContext.DrawLine(pen, new Point(t.P2.X * ActualWidth / pixelWidth, t.P2.Y * ActualHeight / pixelHeight), new Point(t.P0.X * ActualWidth / pixelWidth, t.P0.Y * ActualHeight / pixelHeight));
            }
        }
    }
    class FigureRasterScene : FigureBase
    {
        public FigureRasterScene()
        {
            RenderOptions.SetBitmapScalingMode(this, BitmapScalingMode.NearestNeighbor);
        }
        protected override void OnRender(DrawingContext drawingContext)
        {
            const int pixelWidth = 32;
            const int pixelHeight = 32;
            var scene = new Scene();
            scene.AddChild(new Node("Plane", new TransformMatrix(MathHelp.CreateMatrixScale(10, 10, 10)), new Plane(), StockMaterials.Red, StockMaterials.PlasticRed));
            var mvp = Matrix3D.Identity;
            mvp = MathHelp.Multiply(mvp, MathHelp.Invert(MathHelp.CreateMatrixLookAt(new Vector3D(10, 10, -20), new Vector3D(0, 0, 0), new Vector3D(0, 1, 0))));
            mvp = MathHelp.Multiply(mvp, Perspective.CreateProjection(0.01, 100.0, 60.0 * Math.PI / 180.0, 60.0 * Math.PI / 180.0));
            mvp = MathHelp.Multiply(mvp, Perspective.AspectCorrectFit(ActualWidth, ActualHeight));
            var vertexsource3 = Pipeline.SceneToTriangles(scene);
            var vertexsource4 = Pipeline.Vector3ToVector4(vertexsource3);
            var vertexclipspace = Pipeline.Transform(vertexsource4, mvp);
            var vertexclipped = Pipeline.Clip(vertexclipspace);
            var vertexh = Pipeline.HomogeneousDivide(vertexclipped);
            FigureBase.DrawGrid(drawingContext, ActualWidth, ActualHeight, pixelWidth, pixelHeight, new Pen(Brushes.LightGray, 1));
            {
                var triangles = Pipeline.TransformToScreen(vertexh, pixelWidth, pixelHeight);
                var pixels = Pipeline.Rasterize(triangles);
                FigureBase.DrawBitmap(drawingContext, pixels, ActualWidth, ActualHeight, pixelWidth, pixelHeight);
            }
            {
                var triangles = Pipeline.TransformToScreen(vertexh, ActualWidth, ActualHeight);
                FigureBase.DrawTriangles(drawingContext, triangles);
            }
        }
    }
}