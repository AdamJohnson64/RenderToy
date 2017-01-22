////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using System;
using System.Linq;
using System.Windows.Media;
using System.Windows.Media.Media3D;

namespace RenderToy
{
    public static partial class Render
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
        public static void Point(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            foreach (var transformedobject in TransformedObject.Enumerate(scene))
            {
                Matrix3D model_mvp = transformedobject.Transform * mvp;
                uint color = ColorToUInt32(transformedobject.Node.WireColor);
                // Draw a single pixel to the framebuffer (safety function).
                Action<int, int> drawpixel2d = (x, y) =>
                {
                    if (!(x >= 0 && x < bitmap_width && y >= 0 && y < bitmap_height)) return;
                    unsafe
                    {
                        byte* pRaster = (byte*)bitmap_ptr + bitmap_stride * y;
                        byte* pPixel = pRaster + 4 * x;
                        *(uint*)pPixel = color;
                    }
                };
                // Transform, clip and render a 3D point at P.
                DrawHelp.fnDrawPointWorld drawpoint = (p) =>
                {
                    Point4D v4 = new Point4D(p.X, p.Y, p.Z, 1);
                    v4 = model_mvp.Transform(v4);
                    if (v4.W <= 0) return;
                    drawpixel2d((int)((1 + v4.X / v4.W) * bitmap_width / 2), (int)((1 - v4.Y / v4.W) * bitmap_height / 2));
                };
                IParametricUV uv = transformedobject.Node.Primitive as IParametricUV;
                if (uv != null)
                {
                    DrawHelp.DrawParametricUV(drawpoint, uv.GetPointUV);
                }
                IParametricUVW uvw = transformedobject.Node.Primitive as IParametricUVW;
                if (uvw != null)
                {
                    DrawHelp.DrawParametricUVW(drawpoint, uvw.GetPointUVW);
                }
            }
        }
        #endregion
        #region - Section : Phase 2 - Wireframe Rendering (Reference) -
        public static void Wireframe(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            foreach (var transformedobject in TransformedObject.Enumerate(scene))
            {
                Matrix3D model_mvp = transformedobject.Transform * mvp;
                uint color = ColorToUInt32(transformedobject.Node.WireColor);
                // Draw a single pixel to the framebuffer (safety function).
                Action<int, int> drawpixel2d = (x, y) =>
                {
                    if (!(x >= 0 && x < bitmap_width && y >= 0 && y < bitmap_height)) return;
                    unsafe
                    {
                        byte* pRaster = (byte*)bitmap_ptr + bitmap_stride * y;
                        byte* pPixel = pRaster + 4 * x;
                        *(uint*)pPixel = color;
                    }
                };
                // Draw a line to the framebuffer.
                DrawHelp.fnDrawLineWorld drawline2d = (p1, p2) =>
                {
                    if (Math.Abs(p2.X - p1.X) > Math.Abs(p2.Y - p1.Y))
                    {
                        // X spanning line; this line is longer in the X axis.
                        // Scan in the X direction plotting Y points.
                        if (p1.X > p2.X)
                        {
                            Point3D t = p1;
                            p1 = p2;
                            p2 = t;
                        }
                        for (int x = 0; x < p2.X - p1.X; ++x)
                        {
                            drawpixel2d((int)(p1.X + x), (int)(p1.Y + (p2.Y - p1.Y) * x / (p2.X - p1.X)));
                        }
                    }
                    else
                    {
                        // Y spanning line; this line is longer in the Y axis.
                        // Scan in the Y direction plotting X points.
                        if (p1.Y > p2.Y)
                        {
                            Point3D t = p1;
                            p1 = p2;
                            p2 = t;
                        }
                        for (int y = 0; y < p2.Y - p1.Y; ++y)
                        {
                            drawpixel2d((int)(p1.X + (p2.X - p1.X) * y / (p2.Y - p1.Y)), (int)(p1.Y + y));
                        }
                    }
                };
                // Transform, clip and render a 3D line between P1 and P2.
                DrawHelp.fnDrawLineWorld drawline3d = (p1, p2) => {
                    Point4D v41 = new Point4D(p1.X, p1.Y, p1.Z, 1);
                    Point4D v42 = new Point4D(p2.X, p2.Y, p2.Z, 1);
                    if (!ClipHelp.TransformAndClipLine(ref v41, ref v42, model_mvp)) return;
                    drawline2d(
                        new Point3D((1 + v41.X / v41.W) * bitmap_width / 2, (1 - v41.Y / v41.W) * bitmap_height / 2, v41.Z / v41.W),
                        new Point3D((1 + v42.X / v42.W) * bitmap_width / 2, (1 - v42.Y / v42.W) * bitmap_height / 2, v42.Z / v42.W));
                };
                IParametricUV uv = transformedobject.Node.Primitive as IParametricUV;
                if (uv != null)
                {
                    DrawHelp.DrawParametricUV(drawline3d, uv.GetPointUV);
                }
                IParametricUVW uvw = transformedobject.Node.Primitive as IParametricUVW;
                if (uvw != null)
                {
                    DrawHelp.DrawParametricUVW(drawline3d, uvw.GetPointUVW);
                }
            }
        }
        #endregion
        #region - Section : Phase 3 - Rasterized Rendering (Reference) -
        public static void Raster(Scene scene, Matrix3D mvp, IntPtr bitmap_ptr, int bitmap_width, int bitmap_height, int bitmap_stride)
        {
            float[] depthbuffer = new float[bitmap_width * bitmap_height];
            for (int i = 0; i < depthbuffer.Length; ++i) depthbuffer[i] = 1;
            foreach (var transformedobject in TransformedObject.Enumerate(scene))
            {
                Matrix3D model_mvp = transformedobject.Transform * mvp;
                IParametricUV uv = transformedobject.Node.Primitive as IParametricUV;
                if (uv == null) continue;
                uint color = ColorToUInt32(transformedobject.Node.WireColor);
                // Fill one scanline.
                Action<int, Point3D, Point3D> fillscan = (y, x1, x2) =>
                {
                    if (y < 0 || y >= bitmap_height) return;
                    int sx1 = (int)Math.Max(0, Math.Min(bitmap_width, x1.X));
                    int sx2 = (int)Math.Max(0, Math.Min(bitmap_width, x2.X));
                    unsafe
                    {
                        byte* pRaster = (byte*)bitmap_ptr + bitmap_stride * y;
                        for (int scanx = sx1; scanx < sx2; ++scanx)
                        {
                            // Compute pixel depth.
                            float depth = (float)(x1.Z + (x2.Z - x1.Z) * (scanx - sx1) / (sx2 - sx1));
                            // Reject this pixel if it's further away than what we already have.
                            if (depth > depthbuffer[y * bitmap_width + scanx]) continue;
                            // Otherwise set the depth and render the pixel.
                            depthbuffer[y * bitmap_width + scanx] = depth;
                            byte* pPixel = pRaster + 4 * scanx;
                            *(uint*)pPixel = color;
                        }
                    }
                };
                // Fill a triangle defined by 3 points.
                Action<Point3D, Point3D, Point3D> filltri_viewspace = (p1, p2, p3) =>
                {
                    double ymin = Math.Min(p1.Y, Math.Min(p2.Y, p3.Y));
                    double ymax = Math.Max(p1.Y, Math.Max(p2.Y, p3.Y));
                    int yscanmin = Math.Max(0, (int)ymin);
                    int yscanmax = Math.Min(bitmap_height, (int)ymax);
                    // Calculate edge lines.
                    var e1 = new { O = p1, D = p2 - p1 };
                    var e2 = new { O = p2, D = p3 - p2 };
                    var e3 = new { O = p3, D = p1 - p3 };
                    var edges = new[] { e1, e2, e3 };
                    Point3D o1 = p1, o2 = p2, o3 = p3;
                    Vector3D d1 = p2 - p1, d2 = p3 - p2, d3 = p1 - p3;
                    // Scan in the range of the triangle.
                    for (int y = yscanmin; y <= yscanmax; ++y)
                    {
                        double sy = y;
                        var allx = edges
                            .Select(e => new { E = e, l = (sy - e.O.Y) / e.D.Y })
                            .Where(e => e.l >= 0 && e.l <= 1)
                            .Select(e => e.E.O + e.l * e.E.D)
                            .OrderBy(x => x.X)
                            .ToArray();
                        if (allx.Length == 0) continue;
                        fillscan(y, allx.First(), allx.Last());
                    }
                };
                Action<Point4D, Point4D, Point4D> filltri_clipspace = (p1, p2, p3) =>
                {
                    foreach (var tri in ClipHelp.ClipTriangle3D(new ClipHelp.Triangle { p1 = p1, p2 = p2, p3 = p3 }))
                    {
                        Point4D[] v3 = { tri.p1, tri.p2, tri.p3 };
                        Point3D[] v3t = v3
                            .Select(p => new Point3D(p.X / p.W, p.Y / p.W, p.Z / p.W))
                            .Select(p => new Point3D((1 + p.X) * bitmap_width / 2, (1 - p.Y) * bitmap_height / 2, p.Z))
                            .ToArray();
                        filltri_viewspace(v3t[0], v3t[1], v3t[2]);
                    }
                };
                for (int v = 0; v < 10; ++v)
                {
                    for (int u = 0; u < 10; ++u)
                    {
                        Point3D[] v3 =
                        {
                            uv.GetPointUV((u + 0.0) / 10, (v + 0.0) / 10),
                            uv.GetPointUV((u + 1.0) / 10, (v + 0.0) / 10),
                            uv.GetPointUV((u + 0.0) / 10, (v + 1.0) / 10),
                            uv.GetPointUV((u + 1.0) / 10, (v + 1.0) / 10),
                        };
                        // Transform all points into clip space.
                        Point4D[] v3t = v3
                            .Select(p => new Point4D(p.X, p.Y, p.Z, 1))
                            .Select(p => model_mvp.Transform(p))
                            .ToArray();
                        // Fill the clip space triangle.
                        filltri_clipspace(v3t[0], v3t[1], v3t[3]);
                        filltri_clipspace(v3t[3], v3t[2], v3t[0]);
                    }
                }
            }
        }
        #endregion
    }
}