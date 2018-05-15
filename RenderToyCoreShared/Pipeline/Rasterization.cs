////////////////////////////////////////////////////////////////////////////////
// RenderToy - A bit of history that's now a bit of silicon...
// Copyright (C) Adam Johnson 2017
////////////////////////////////////////////////////////////////////////////////

using RenderToy.SceneGraph;
using RenderToy.Utility;
using System;
using System.Collections.Generic;
using System.Linq;

namespace RenderToy.PipelineModel
{
    static partial class Rasterization
    {
        /// <summary>
        /// Convert a Vector4D color to a Bgra32 uint color.
        /// </summary>
        /// <param name="color">The vector form of the color.</param>
        /// <returns>A 32-bit Bgra32 color.</returns>
        public static uint ColorToUInt32(Vector4D color)
        {
            return
                ((uint)(MathHelp.Saturate(color.W) * 255) << 24) |
                ((uint)(MathHelp.Saturate(color.X) * 255) << 16) |
                ((uint)(MathHelp.Saturate(color.Y) * 255) << 8) |
                ((uint)(MathHelp.Saturate(color.Z) * 255) << 0);
        }
        /// <summary>
        /// Write a stream of pixels into an unmanaged Bgra32 bitmap.
        /// </summary>
        /// <param name="pixels">The stream of pixels to be written.</param>
        /// <param name="bitmap_ptr">The base pointer of the bitmap.</param>
        /// <param name="render_width">The pixel width of the bitmap.</param>
        /// <param name="render_height">The pixel height of the bitmap.</param>
        /// <param name="bitmap_stride">The count of bytes between rasters of the bitmap.</param>
        public static void FillBitmap(IEnumerable<PixelBgra32> pixels, IntPtr bitmap_ptr, int render_width, int render_height, int bitmap_stride)
        {
            foreach (var pixel in pixels)
            {
                if (!(pixel.X >= 0 && pixel.X < render_width && pixel.Y >= 0 && pixel.Y < render_height))
                {
                    continue;
                }
                unsafe
                {
                    byte* pRaster = (byte*)bitmap_ptr + bitmap_stride * pixel.Y;
                    byte* pPixel = pRaster + 4 * pixel.X;
                    *(uint*)pPixel = pixel.Color;
                }
            }
        }
        /// <summary>
        /// Rasterize a single screen-space points into colored pixels.
        /// </summary>
        /// <param name="point">The point to be rasterized.</param>
        /// <returns>A stream of pixels to write to the framebuffer.</returns>
        public static IEnumerable<PixelBgra32> RasterizePoint(Vector4D point, uint color)
        {
            yield return new PixelBgra32 { X = (ushort)point.X, Y = (ushort)point.Y, Color = color };
        }
        /// <summary>
        /// Rasterize a stream of screen-space points and emit pixels for all.
        /// </summary>
        /// <param name="vertices">The vertex source to be transformed.</param>
        /// <returns>A stream of screen transformed pixels.</returns>
        public static IEnumerable<PixelBgra32> RasterizePoint(IEnumerable<Vector4D> vertices, uint color)
        {
            return vertices.SelectMany(p => RasterizePoint(p, color));
        }
        /// <summary>
        /// Rasterize a stream of points to a stream of pixels.
        /// </summary>
        /// <param name="points">The points to rasterize.</param>
        /// <param name="mvp">The model-view-projection matrix into clip space.</param>
        /// <param name="color">The color to render pixels with.</param>
        /// <param name="render_width">The pixel width of the target bitmap.</param>
        /// <param name="render_height">The pixel height of the target bitmap.</param>
        /// <returns>A stream of pixels representing all points.</returns>
        public static IEnumerable<PixelBgra32> RasterizePoint(IEnumerable<Vector3D> points, Matrix3D mvp, uint color, int render_width, int render_height)
        {
            var v3tov4 = Transformation.Vector3ToVector4(points);
            var clipspace = Transformation.Transform(v3tov4, mvp);
            var clipped = Clipping.ClipPoint(clipspace);
            var hdiv = Transformation.HomogeneousDivide(clipped);
            var screenspace = Transformation.TransformToScreen(hdiv, render_width, render_height);
            return RasterizePoint(screenspace, color);
        }
        /// <summary>
        /// Rasterize a transformed scene of colored objects as points into a stream of pixels.
        /// </summary>
        /// <param name="scene">The scene to be rasterized.</param>
        /// <param name="mvp">The model-view-projection matrix into clip space.</param>
        /// <param name="render_width">The pixel width of the target bitmap.</param>
        /// <param name="render_height">The pixel height of the target bitmap.</param>
        /// <returns>A stream of pixels representing the complete scene as points.</returns>
        public static IEnumerable<PixelBgra32> RasterizePoint(Scene scene, Matrix3D mvp, int render_width, int render_height)
        {
            return TransformedObject.Enumerate(scene).SelectMany(x => RasterizePoint(PrimitiveAssembly.CreatePoints(x.Node.GetPrimitive()), x.Transform * mvp, Rasterization.ColorToUInt32(x.Node.GetWireColor()), render_width, render_height));
        }
        /// <summary>
        /// Rasterize a stream of clip-space points and emit pixels.
        /// </summary>
        /// <param name="vertices">The vertex source to be transformed.</param>
        /// <param name="width">The width of the screen in pixels.</param>
        /// <param name="height">The height of the screen in pixels.</param>
        /// <returns>A stream of pixels to be written to the framebuffer.</returns>
        public static IEnumerable<PixelBgra32> RasterizePoint(IEnumerable<Vector4D> vertices, ushort width, ushort height)
        {
            return vertices.Select(v => new PixelBgra32 { X = (ushort)((v.X + 1) * width / 2), Y = (ushort)((1 - v.Y) * height / 2), Color = 0xFF808080 });
        }
        /// <summary>
        /// Rasterize a line in sceen space and emit pixels.
        /// </summary>
        /// <param name="line"></param>
        /// <returns>A stream of pixels to be written to the framebuffer.</returns>
        public static IEnumerable<PixelBgra32> RasterizeLine(Vector4D P0, Vector4D P1, uint color)
        {
            if (Math.Abs(P1.X - P0.X) > Math.Abs(P1.Y - P0.Y))
            {
                // X spanning line; this line is longer in the X axis.
                // Scan in the X direction plotting Y points.
                var p0 = new Vector4D();
                var p1 = new Vector4D();
                if (P0.X < P1.X)
                {
                    p0 = P0;
                    p1 = P1;
                }
                else
                {
                    p0 = P1;
                    p1 = P0;
                }
                for (int x = 0; x <= p1.X - p0.X; ++x)
                {
                    yield return new PixelBgra32 { X = (ushort)(p0.X + x), Y = (ushort)(p0.Y + (p1.Y - p0.Y) * x / (p1.X - p0.X)), Color = color };
                }
            }
            else
            {
                // Y spanning line; this line is longer in the Y axis.
                // Scan in the Y direction plotting X points.
                var p0 = new Vector4D();
                var p1 = new Vector4D();
                if (P0.Y < P1.Y)
                {
                    p0 = P0;
                    p1 = P1;
                }
                else
                {
                    p0 = P1;
                    p1 = P0;
                }
                for (int y = 0; y <= p1.Y - p0.Y; ++y)
                {
                    yield return new PixelBgra32 { X = (ushort)(p0.X + (p1.X - p0.X) * y / (p1.Y - p0.Y)), Y = (ushort)(p0.Y + y), Color = color };
                }
            }
        }
        /// <summary>
        /// Rasterize a stream of lines into pixels.
        /// </summary>
        /// <param name="lines">The lines to be rasterized.</param>
        /// <returns>A stream of pixels to be written to the framebuffer.</returns>
        public static IEnumerable<PixelBgra32> RasterizeLine(IEnumerable<Vector4D> lines, uint color)
        {
            IEnumerator<Vector4D> iter = lines.GetEnumerator();
            while (iter.MoveNext())
            {
                var P0 = iter.Current;
                if (!iter.MoveNext()) yield break;
                var P1 = iter.Current;
                foreach (var pixel in RasterizeLine(P0, P1, color))
                {
                    yield return pixel;
                }
            }
        }
        /// <summary>
        /// Rasterize a stream of lines to a stream of pixels.
        /// </summary>
        /// <param name="lines">The lines to rasterize.</param>
        /// <param name="mvp">The model-view-projection matrix into clip space.</param>
        /// <param name="color">The color to render pixels with.</param>
        /// <param name="render_width">The pixel width of the target bitmap.</param>
        /// <param name="render_height">The pixel height of the target bitmap.</param>
        /// <returns>A stream of pixels representing all lines.</returns>
        public static IEnumerable<PixelBgra32> RasterizeLine(IEnumerable<Vector3D> lines, Matrix3D mvp, uint color, int render_width, int render_height)
        {
            var v3tov4 = Transformation.Vector3ToVector4(lines);
            var clipspace = Transformation.Transform(v3tov4, mvp);
            var clipped = Clipping.ClipLine(clipspace);
            var hdiv = Transformation.HomogeneousDivide(clipped);
            var screenspace = Transformation.TransformToScreen(hdiv, render_width, render_height);
            return RasterizeLine(screenspace, color);
        }
        /// <summary>
        /// Rasterize a transformed scene of colored objects as lines into a stream of pixels.
        /// </summary>
        /// <param name="scene">The scene to be rasterized.</param>
        /// <param name="mvp">The model-view-projection matrix into clip space.</param>
        /// <param name="render_width">The pixel width of the target bitmap.</param>
        /// <param name="render_height">The pixel height of the target bitmap.</param>
        /// <returns>A stream of pixels representing the complete scene as lines.</returns>
        public static IEnumerable<PixelBgra32> RasterizeLine(Scene scene, Matrix3D mvp, int render_width, int render_height)
        {
            return TransformedObject.Enumerate(scene).SelectMany(x => RasterizeLine(PrimitiveAssembly.CreateLines(x.Node.GetPrimitive()), x.Transform * mvp, Rasterization.ColorToUInt32(x.Node.GetWireColor()), render_width, render_height));
        }
        /// <summary>
        /// Rasterize a triangle in screen-space and emit pixels.
        /// </summary>
        /// <param name="triangle">The triangle to be rasterized.</param>
        /// <returns>A stream of pixels to be written to the framebuffer.</returns>
        public static IEnumerable<PixelBgra32> RasterizeTriangle(Vector4D P0, Vector4D P1, Vector4D P2, uint color)
        {
            // Calculate edge lines.
            var edges = new[]
            {
                new { Org = P0, Dir = P1 - P0 },
                new { Org = P1, Dir = P2 - P1 },
                new { Org = P2, Dir = P0 - P2 }
            };
            // Scan in the range of the triangle.
            int yscanmin = (int)Math.Floor(edges.Min(p => p.Org.Y));
            int yscanmax = (int)Math.Ceiling(edges.Max(p => p.Org.Y));
            for (int y = yscanmin; y <= yscanmax; ++y)
            {
                double yline = y + 0.5;
                var allx = edges
                    .Select(edge => new { Edge = edge, Lambda = (yline - edge.Org.Y) / edge.Dir.Y })
                    .Where(edge => edge.Lambda >= 0 && edge.Lambda <= 1)
                    .Select(edge => edge.Edge.Org + edge.Lambda * edge.Edge.Dir)
                    .OrderBy(x => x.X)
                    .ToArray();
                if (allx.Count() == 0) continue;
                int xmin = (int)(allx.First().X + 0.5);
                int xmax = (int)(allx.Last().X + 0.5);
                for (int x = xmin; x < xmax; ++x)
                {
                    double xline = x + 0.5;
                    double alpha = MathHelp.Dot(new Vector2D(xline - P0.X, yline - P0.Y), new Vector2D(P1.X - P0.X, P1.Y - P0.Y));
                    double beta = MathHelp.Dot(new Vector2D(xline - P0.X, yline - P0.Y), new Vector2D(P2.X - P0.X, P2.Y - P0.Y));
                    yield return new PixelBgra32 { X = (ushort)x, Y = (ushort)y, Color = color };
                }
            }
        }
        /// <summary>
        /// Rasterize a stream of triangles into pixels.
        /// </summary>
        /// <param name="triangles">The triangles to be rasterized.</param>
        /// <returns>A stream of pixels to be written to the framebuffer.</returns>
        public static IEnumerable<PixelBgra32> RasterizeTriangle(IEnumerable<Vector4D> triangles, uint color)
        {
            var iter = triangles.GetEnumerator();
            while (iter.MoveNext())
            {
                var P0 = iter.Current;
                if (!iter.MoveNext())
                {
                    yield break;
                }
                var P1 = iter.Current;
                if (!iter.MoveNext())
                {
                    yield break;
                }
                var P2 = iter.Current;
                foreach (var pixel in RasterizeTriangle(P0, P1, P2, color))
                {
                    yield return pixel;
                }
            }
        }
        /// <summary>
        /// Rasterize a stream of triangles to a stream of pixels.
        /// </summary>
        /// <param name="triangles">The triangles to rasterize.</param>
        /// <param name="mvp">The model-view-projection matrix into clip space.</param>
        /// <param name="color">The color to render pixels with.</param>
        /// <param name="render_width">The pixel width of the target bitmap.</param>
        /// <param name="render_height">The pixel height of the target bitmap.</param>
        /// <returns>A stream of pixels representing all triangles.</returns>
        public static IEnumerable<PixelBgra32> RasterizeTriangle(IEnumerable<Vector3D> triangles, Matrix3D mvp, uint color, int render_width, int render_height)
        {
            var v3tov4 = Transformation.Vector3ToVector4(triangles);
            var clipspace = Transformation.Transform(v3tov4, mvp);
            var clipped = Clipping.ClipTriangle(clipspace);
            var hdiv = Transformation.HomogeneousDivide(clipped);
            var screenspace = Transformation.TransformToScreen(hdiv, render_width, render_height);
            return RasterizeTriangle(screenspace, color);
        }
        /// <summary>
        /// Rasterize a transformed scene of colored objects as triangles into a stream of pixels.
        /// </summary>
        /// <param name="scene">The scene to be rasterized.</param>
        /// <param name="mvp">The model-view-projection matrix into clip space.</param>
        /// <param name="render_width">The pixel width of the target bitmap.</param>
        /// <param name="render_height">The pixel height of the target bitmap.</param>
        /// <returns>A stream of pixels representing the complete scene as triangles.</returns>
        public static IEnumerable<PixelBgra32> RasterizeTriangle(Scene scene, Matrix3D mvp, int render_width, int render_height)
        {
            return TransformedObject.Enumerate(scene).SelectMany(x => RasterizeTriangle(PrimitiveAssembly.CreateTriangles(x.Node.GetPrimitive()), x.Transform * mvp, Rasterization.ColorToUInt32(x.Node.GetWireColor()), render_width, render_height));
        }
        /// <summary>
        /// Rasterize a homogeneous triangle directly without homogeneous divide.
        /// </summary>
        /// <param name="P0">The first point of the triangle.</param>
        /// <param name="P1">The second point of the triangle.</param>
        /// <param name="P2">The third point of the triangle.</param>
        /// <param name="pixelWidth">The pixel width of the target bitmap.</param>
        /// <param name="pixelHeight">The pixel height of the target bitmap.</param>
        /// <returns>A stream of pixels forming the rasterized triangle.</returns>
        public static IEnumerable<PixelBgra32> RasterizeHomogeneous(Vector4D P0, Vector4D P1, Vector4D P2, ushort pixelWidth, ushort pixelHeight)
        {
            // Early out if everything is behind the camera.
            if (P0.W <= 0 && P1.W <= 0 && P2.W <= 0)
            {
                yield break;
            }
            // Compute a conservative bounding box.
            int minx = int.MaxValue;
            int maxx = int.MinValue;
            int miny = int.MaxValue;
            int maxy = int.MinValue;
            foreach (Vector4D v in new Vector4D[] { P0, P1, P2 })
            {
                if (v.W > 0)
                {
                    Vector4D project = v * (1 / v.W);
                    double px = (project.X + 1) / 2 * pixelWidth;
                    double py = (1 - project.Y) / 2 * pixelHeight;
                    minx = Math.Min(minx, (int)px - 1);
                    maxx = Math.Max(maxx, (int)px + 1);
                    miny = Math.Min(miny, (int)py - 1);
                    maxy = Math.Max(maxy, (int)py + 1);
                }
                else
                {
                    if (v.X < 0) minx = int.MinValue;
                    if (v.X > 0) maxx = int.MaxValue;
                    if (v.Y > 0) miny = int.MinValue;
                    if (v.Y < 0) maxy = int.MaxValue;
                }
            }
            minx = Math.Max(minx, 0);
            maxx = Math.Min(maxx, pixelWidth);
            miny = Math.Max(miny, 0);
            maxy = Math.Min(maxy, pixelHeight);
            // Perform the rasterization within these bounds.
            Matrix3D Minv = MathHelp.Invert(
                new Matrix3D(
                    P0.X, P1.X, P2.X, 0,
                    P0.Y, P1.Y, P2.Y, 0,
                    P0.W, P1.W, P2.W, 0,
                    0, 0, 0, 1));
            Vector3D interp = MathHelp.TransformVector(Minv, new Vector3D(1, 1, 1));
            for (int y = miny; y < maxy; ++y)
            {
                for (int x = minx; x < maxx; ++x)
                {
                    double px = (x + 0.5) / pixelWidth * 2 - 1;
                    double py = 1 - (y + 0.5) / pixelHeight * 2;
                    double w = interp.X * px + interp.Y * py + interp.Z;
                    double a = Minv.M11 * px + Minv.M12 * py + Minv.M13;
                    double b = Minv.M21 * px + Minv.M22 * py + Minv.M23;
                    double c = Minv.M31 * px + Minv.M32 * py + Minv.M33;
                    if (a > 0 && b > 0 && c > 0)
                    {
                        double R = a / w;
                        double G = b / w;
                        double B = c / w;
                        uint color = ((uint)(R * 255) << 16) | ((uint)(G * 255) << 8) | ((uint)(B * 255) << 0) | 0xFF000000;
                        yield return new PixelBgra32 { X = (ushort)x, Y = (ushort)y, Color = color };
                    }
                }
            }
        }
        /// <summary>
        /// Rasterize a stream of homogeneous triangles directly without homogeneous divide.
        /// </summary>
        /// <param name="triangles">The input stream of triangles.</param>
        /// <param name="pixelWidth">The pixel width of the target bitmap.</param>
        /// <param name="pixelHeight">The pixel height of the target bitmap.</param>
        /// <returns>A stream of pixels forming all rasterized triangles.</returns>
        public static IEnumerable<PixelBgra32> RasterizeHomogeneous(IEnumerable<Vector4D> triangles, ushort pixelWidth, ushort pixelHeight)
        {
            var iter = triangles.GetEnumerator();
            while (iter.MoveNext())
            {
                var P0 = iter.Current;
                if (!iter.MoveNext())
                {
                    yield break;
                }
                var P1 = iter.Current;
                if (!iter.MoveNext())
                {
                    yield break;
                }
                var P2 = iter.Current;
                foreach (var pixel in RasterizeHomogeneous(P0, P1, P2, pixelWidth, pixelHeight))
                {
                    yield return pixel;
                }
            }
        }
        /// <summary>
        /// Rasterize a stream of triangles to a stream of pixels via homogeneous rasterization.
        /// </summary>
        /// <param name="triangles">The triangles to rasterize.</param>
        /// <param name="mvp">The model-view-projection matrix into clip space.</param>
        /// <param name="color">The color to render pixels with.</param>
        /// <param name="render_width">The pixel width of the target bitmap.</param>
        /// <param name="render_height">The pixel height of the target bitmap.</param>
        /// <returns>A stream of pixels representing all triangles.</returns>
        public static IEnumerable<PixelBgra32> RasterizeHomogeneous(IEnumerable<Vector3D> triangles, Matrix3D mvp, uint color, ushort render_width, ushort render_height)
        {
            var v3tov4 = Transformation.Vector3ToVector4(triangles);
            var clipspace = Transformation.Transform(v3tov4, mvp);
            return RasterizeHomogeneous(clipspace, render_width, render_height);
        }
        /// <summary>
        /// Rasterize a transformed scene of colored objects as triangles into a stream of pixels.
        /// </summary>
        /// <param name="scene">The scene to be rasterized.</param>
        /// <param name="mvp">The model-view-projection matrix into clip space.</param>
        /// <param name="render_width">The pixel width of the target bitmap.</param>
        /// <param name="render_height">The pixel height of the target bitmap.</param>
        /// <returns>A stream of pixels representing the complete scene as triangles.</returns>
        public static IEnumerable<PixelBgra32> RasterizeHomogeneous(Scene scene, Matrix3D mvp, ushort render_width, ushort render_height)
        {
            return TransformedObject.Enumerate(scene).SelectMany(x => RasterizeHomogeneous(PrimitiveAssembly.CreateTriangles(x.Node.GetPrimitive()), x.Transform * mvp, Rasterization.ColorToUInt32(x.Node.GetWireColor()), render_width, render_height));
        }
    }
}