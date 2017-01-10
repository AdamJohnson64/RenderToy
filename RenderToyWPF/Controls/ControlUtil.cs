////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System;
using System.Linq;
using System.Windows;
using System.Windows.Interop;
using System.Windows.Media;
using System.Windows.Media.Imaging;
using System.Windows.Media.Media3D;

namespace RenderToy
{
    public static class ControlUtil
    {
        #region - Section : General -
        public static uint ColorToUInt32(Color color)
        {
            return
                ((uint)color.A << 24) |
                ((uint)color.R << 16) |
                ((uint)color.G << 8) |
                ((uint)color.B << 0);
        }
        #endregion
        #region - Section : Phase 1 - Point Rendering (Reference) -
        public static void DrawPoint(Scene scene, Matrix3D mvp, int render_width, int render_height, DrawingContext drawingContext, double width, double height)
        {
            var bitmap = ImagePoint(scene, mvp, render_width, render_height);
            drawingContext.DrawImage(bitmap, new Rect(0, 0, width, height));
        }
        public static ImageSource ImagePoint(Scene scene, Matrix3D mvp, int render_width, int render_height)
        {
            WriteableBitmap bitmap = new WriteableBitmap(render_width, render_height, 0, 0, PixelFormats.Bgra32, null);
            bitmap.Lock();
            unsafe
            {
                Action<Point3D, uint> plot = (p, color) => {
                    int x = (int)p.X;
                    int y = (int)p.Y;
                    // Discard pixels outside the framebuffer.
                    if (!(x >= 0 && x < render_width && y >= 0 && y < render_height)) return;
                    byte* pRaster = (byte*)bitmap.BackBuffer + bitmap.BackBufferStride * y;
                    byte* pPixel = pRaster + 4 * x;
                    *(uint*)pPixel = color;
                };
                foreach (var transformedobject in TransformedObject.Enumerate(scene))
                {
                    Matrix3D model_mvp = transformedobject.Transform * mvp;
                    IParametricUV uv = transformedobject.Node.Primitive as IParametricUV;
                    if (uv == null) continue;
                    for (int v = 0; v <= 20; ++v)
                    {
                        for (int u = 0; u <= 20; ++u)
                        {
                            var v3 = new Point3D[]
                            {
                                uv.GetPointUV((u + 0.0) / 20, (v + 0.0) / 20),
                            };
                            var v3t = v3
                                .Select(p => new Point4D(p.X, p.Y, p.Z, 1))
                                .Select(p => model_mvp.Transform(p))
                                .Where(p => p.W > 0)
                                .Select(p => new Point3D(p.X / p.W, p.Y / p.W, p.Z / p.W))
                                .Select(p => new Point3D((1 + p.X) * render_width / 2, (1 - p.Y) * render_height / 2, p.Z));
                            foreach (var vtx in v3t)
                            {
                                plot(vtx, ColorToUInt32(transformedobject.Node.WireColor));
                            }
                        }
                    }
                }
            }
            bitmap.AddDirtyRect(new Int32Rect(0, 0, render_width, render_height));
            bitmap.Unlock();
            return bitmap;
        }
        #endregion
        #region - Section : Phase 1 - Point Rendering (GDI+) -
        public static void DrawPointGDI(Scene scene, Matrix3D mvp, int render_width, int render_height, DrawingContext drawingContext, double width, double height)
        {
            DrawPointALR(scene, mvp, new WireframeGDIPlus(drawingContext, render_width, render_height), width, height);
        }
        #endregion
        #region - Section : Phase 1 - Point Rendering (WPF) -
        public static void DrawPointWPF(Scene scene, Matrix3D mvp, DrawingContext drawingContext, double width, double height)
        {
            DrawPointALR(scene, mvp, new WireframeWPF(drawingContext), width, height);
        }
        #endregion
        #region - Section : Phase 1 - Point Rendering (Abstract Line Renderer) -
        public static void DrawPointALR(Scene scene, Matrix3D mvp, IWireframeRenderer renderer, double width, double height)
        {
            DrawHelp.fnDrawLineViewport lineviewport = CreateLineViewportFunction(renderer, width, height);
            renderer.WireframeBegin();
            // Draw something interesting.
            renderer.WireframeColor(0.0, 0.0, 0.0);
            foreach (var transformedobject in TransformedObject.Enumerate(scene))
            {
                IParametricUV uv = transformedobject.Node.Primitive as IParametricUV;
                if (uv == null) continue;
                DrawHelp.fnDrawPointWorld line = CreatePointWorldFunction(lineviewport, transformedobject.Transform * mvp);
                Color color = transformedobject.Node.WireColor;
                renderer.WireframeColor(color.R / 255.0 / 2, color.G / 255.0 / 2, color.B / 255.0 / 2);
                DrawHelp.DrawParametricUV(line, uv);
            }
            renderer.WireframeEnd();
        }
        public static DrawHelp.fnDrawPointWorld CreatePointWorldFunction(DrawHelp.fnDrawLineViewport line, Matrix3D mvp)
        {
            const double s = 0.01;
            return (p) =>
            {
                DrawHelp.DrawLineWorld(line, mvp, new Point4D(p.X - s, p.Y, p.Z, 1.0), new Point4D(p.X + s, p.Y, p.Z, 1.0));
                DrawHelp.DrawLineWorld(line, mvp, new Point4D(p.X, p.Y - s, p.Z, 1.0), new Point4D(p.X, p.Y + s, p.Z, 1.0));
                DrawHelp.DrawLineWorld(line, mvp, new Point4D(p.X, p.Y, p.Z - s, 1.0), new Point4D(p.X, p.Y, p.Z + s, 1.0));
            };
        }
        #endregion
        #region - Section : Phase 2 - Wireframe Rendering (GDI+) -
        public static void DrawWireframeGDI(Scene scene, Matrix3D mvp, int render_width, int render_height, DrawingContext drawingContext, double width, double height)
        {
            DrawWireframeCommon(scene, mvp, new WireframeGDIPlus(drawingContext, render_width, render_height), width, height);
        }
        #endregion
        #region - Section : Phase 2 - Wireframe Rendering (WPF) -
        public static void DrawWireframeWPF(Scene scene, Matrix3D mvp, DrawingContext drawingContext, double width, double height)
        {
            DrawWireframeCommon(scene, mvp, new WireframeWPF(drawingContext), width, height);
        }
        #endregion
        #region - Section : Phase 2 - Wireframe Rendering (Abstract Line Renderer) -
        private static void DrawWireframeCommon(Scene scene, Matrix3D mvp, IWireframeRenderer renderer, double width, double height)
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
        #endregion
        #region - Section : Phase 3 - Rasterized Rendering (Reference) -
        public static void DrawRaster(Scene scene, Matrix3D mvp, int render_width, int render_height, DrawingContext drawingContext, double width, double height)
        {
            var bitmap = ImageRaster(scene, mvp, render_width, render_height);
            drawingContext.DrawImage(bitmap, new Rect(0, 0, width, height));
        }
        public static ImageSource ImageRaster(Scene scene, Matrix3D mvp, int render_width, int render_height)
        {
            WriteableBitmap bitmap = new WriteableBitmap(render_width, render_height, 0, 0, PixelFormats.Bgra32, null);
            bitmap.Lock();
            unsafe
            {
                // Fill one scanline.
                Action<int, int, int, uint> fillscan = (y, x1, x2, color) =>
                {
                    if (y < 0 || y >= render_height) return;
                    x1 = Math.Max(0, Math.Min(render_width, x1));
                    x2 = Math.Max(0, Math.Min(render_width, x2));
                    byte* pRaster = (byte*)bitmap.BackBuffer + bitmap.BackBufferStride * y;
                    for (int scanx = x1; scanx < x2; ++scanx)
                    {
                        byte* pPixel = pRaster + 4 * scanx;
                        *(uint*)pPixel = color;
                    }
                    if (x1 >= 0 && x1 < render_width)
                    {
                        *(uint*)(pRaster + 4 * x1) = 0xffff0000U;
                    }
                };
                // Fill a triangle defined by 3 points.
                Action<Point3D, Point3D, Point3D, uint> filltri = (p1, p2, p3, color) =>
                {
                    // Define a triangle.
                    Point3D[] tpoints_unordered = { p1, p2, p3 };
                    // Order the points by ascending Y coordinate.
                    Point3D[] tpoints = tpoints_unordered.OrderBy(x => x.Y).ToArray();
                    // Section 1 - t[0] to t[1] vertical scan; top part of triangle.
                    // Define the left and right slopes (we don't know the order yet.
                    {
                        double[] slope = {
                            (tpoints[1].X - tpoints[0].X) / (tpoints[1].Y - tpoints[0].Y),
                            (tpoints[2].X - tpoints[0].X) / (tpoints[2].Y - tpoints[0].Y),
                        };
                        // Order the slopes to determine the left and right sides.
                        double[] slope_by_x = slope.OrderBy(x => x).ToArray();
                        // Scan Y and fill lines over the interpolated slopes.
                        for (int y = 0; y < tpoints[1].Y - tpoints[0].Y; ++y)
                        {
                            fillscan(
                                (int)(tpoints[0].Y + y),
                                (int)(tpoints[0].X + slope_by_x[0] * y),
                                (int)(tpoints[0].X + slope_by_x[1] * y),
                                color);
                        }
                    }
                    // Section 2 - t[2] to t[1] inverted vertical scan; bottom part of triangle.
                    {
                        double[] slope_unordered = {
                            (tpoints[0].X - tpoints[2].X) / (tpoints[2].Y - tpoints[0].Y),
                            (tpoints[1].X - tpoints[2].X) / (tpoints[2].Y - tpoints[1].Y),
                        };
                        // Order the slopes to determine the left and right sides.
                        double[] slope = slope_unordered.OrderBy(x => x).ToArray();
                        // Scan Y and fill lines over the interpolated slopes.
                        for (int y = 0; y < tpoints[2].Y - tpoints[1].Y; ++y)
                        {
                            fillscan(
                                (int)(tpoints[2].Y - y),
                                (int)(tpoints[2].X + slope[0] * y),
                                (int)(tpoints[2].X + slope[1] * y),
                                color);
                        }
                    }
                };
                foreach (var transformedobject in TransformedObject.Enumerate(scene))
                {
                    Matrix3D model_mvp = transformedobject.Transform * mvp;
                    IParametricUV uv = transformedobject.Node.Primitive as IParametricUV;
                    if (uv == null) continue;
                    for (int v = 0; v < 10; ++v)
                    {
                        for (int u = 0; u < 10; ++u)
                        {
                            Point3D[] v3 = new Point3D[]
                            {
                                uv.GetPointUV((u + 0.0) / 10, (v + 0.0) / 10),
                                uv.GetPointUV((u + 1.0) / 10, (v + 0.0) / 10),
                                uv.GetPointUV((u + 0.0) / 10, (v + 1.0) / 10),
                                uv.GetPointUV((u + 1.0) / 10, (v + 1.0) / 10),
                            };
                            Point3D[] v3t = v3
                                .Select(p => new Point4D(p.X, p.Y, p.Z, 1))
                                .Select(p => model_mvp.Transform(p))
                                .Select(p => new Point3D(p.X / p.W, p.Y / p.W, p.Z / p.W))
                                .Select(p => new Point3D((1 + p.X) * render_width / 2, (1 - p.Y) * render_height / 2, p.Z))
                                .ToArray();
                            filltri(v3t[0], v3t[1], v3t[3], ColorToUInt32(transformedobject.Node.WireColor));
                            filltri(v3t[3], v3t[2], v3t[0], ColorToUInt32(transformedobject.Node.WireColor));
                        }
                    }
                }
            }
            bitmap.AddDirtyRect(new Int32Rect(0, 0, render_width, render_height));
            bitmap.Unlock();
            return bitmap;
        }
        #endregion
        #region - Section : Phase 3 - Rasterized Rendering (Direct3D 9) -
        public static void DrawRasterD3D9(Scene scene, Matrix3D mvp, int render_width, int render_height, DrawingContext drawingContext, double width, double height)
        {
            var bitmap = ImageRasterD3D9(scene, mvp, render_width, render_height);
            drawingContext.DrawImage(bitmap, new Rect(0, 0, width, height));
        }
        public static ImageSource ImageRasterD3D9(Scene scene, Matrix3D mvp, int render_width, int render_height)
        {
            D3D9Surface d3dsurface = d3dsurface = new D3D9Surface();
            D3DImage d3dimage = new D3DImage();
            d3dimage.Lock();
            d3dimage.SetBackBuffer(D3DResourceType.IDirect3DSurface9, d3dsurface.SurfacePtr, true);
            d3dimage.AddDirtyRect(new Int32Rect(0, 0, 256, 256));
            d3dimage.Unlock();
            return d3dimage;
        }
        #endregion
        #region - Section : Phase 4 - Raytrace Rendering (Reference) -
        public static void DrawRaytrace(Scene scene, Matrix3D mvp, int render_width, int render_height, DrawingContext drawingContext, double width, double height)
        {
            var bitmap = ImageRaytrace(scene, mvp, render_width, render_height);
            drawingContext.DrawImage(bitmap, new Rect(0, 0, width, height));
        }
        public static ImageSource ImageRaytrace(Scene scene, Matrix3D mvp, int render_width, int render_height)
        {
            var bitmap = new WriteableBitmap(render_width, render_height, 0, 0, PixelFormats.Bgra32, null);
            bitmap.Lock();
            Raytrace.DoRaytrace(scene, MathHelp.Invert(mvp), bitmap.PixelWidth, bitmap.PixelHeight, bitmap.BackBuffer, bitmap.BackBufferStride);
            bitmap.AddDirtyRect(new Int32Rect(0, 0, bitmap.PixelWidth, bitmap.PixelHeight));
            bitmap.Unlock();
            return bitmap;
        }
        #endregion
    }
}