using System.Windows;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Media.Media3D;

namespace RenderToy
{
    public static class ControlUtil
    {
        public static void RenderRaytrace(DrawingContext drawingContext, double width, double height, Scene scene, Matrix3D mvp, int render_width, int render_height)
        {
            var bitmap = new WriteableBitmap(render_width, render_height, 0, 0, PixelFormats.Bgra32, null);
            bitmap.Lock();
            Raytrace.DoRaytrace(scene, MathHelp.Invert(mvp), bitmap.PixelWidth, bitmap.PixelHeight, bitmap.BackBuffer, bitmap.BackBufferStride);
            bitmap.AddDirtyRect(new Int32Rect(0, 0, bitmap.PixelWidth, bitmap.PixelHeight));
            bitmap.Unlock();
            drawingContext.DrawImage(bitmap, new Rect(0, 0, width, height));
        }
        public static void RenderWireframeGDI(DrawingContext drawingContext, double width, double height, Scene scene, Matrix3D mvp, int render_width, int render_height)
        {
            RenderWireframe(new WireframeGDIPlus(drawingContext, render_width, render_height), width, height, scene, mvp);
        }
        public static void RenderWireframeWPF(DrawingContext drawingContext, double width, double height, Scene scene, Matrix3D mvp)
        {
            RenderWireframe(new WireframeWPF(drawingContext), width, height, scene, mvp);
        }
        private static void RenderWireframe(IWireframeRenderer renderer, double width, double height, Scene scene, Matrix3D mvp)
        {
            DrawHelp.fnDrawLineViewport lineviewport = CreateLineViewportFunction(renderer, width, height);
            renderer.WireframeBegin();
            // Draw something interesting.
            renderer.WireframeColor(0.0, 0.0, 0.0);
            foreach (var transformedobject in TransformedObject.Enumerate(scene))
            {
                IParametricUV uv = transformedobject.Node.Primitive as IParametricUV;
                if (uv == null) continue;
                DrawHelp.fnDrawLineWorld line = CreateLineWorldFunction(lineviewport, transformedobject.Transform * mvp);
                Color color = transformedobject.Node.WireColor;
                renderer.WireframeColor(color.R / 255.0 / 2, color.G / 255.0 / 2, color.B / 255.0 / 2);
                DrawHelp.DrawParametricUV(line, uv);
            }
            renderer.WireframeEnd();
        }
        public static DrawHelp.fnDrawLineViewport CreateLineViewportFunction(IWireframeRenderer renderer, double width, double height)
        {
            return (p1, p2) =>
            {
                renderer.WireframeLine(
                    (p1.X + 1) * width / 2, (1 - p1.Y) * height / 2,
                    (p2.X + 1) * width / 2, (1 - p2.Y) * height / 2);
            };
        }
        public static DrawHelp.fnDrawLineWorld CreateLineWorldFunction(DrawHelp.fnDrawLineViewport line, Matrix3D mvp)
        {
            return (p1, p2) =>
            {
                DrawHelp.DrawLineWorld(line, mvp, new Point4D(p1.X, p1.Y, p1.Z, 1.0), new Point4D(p2.X, p2.Y, p2.Z, 1.0));
            };
        }
    }
}