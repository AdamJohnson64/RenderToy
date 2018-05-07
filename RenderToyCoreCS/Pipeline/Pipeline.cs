// The pipeline is our ticket to fully configurable render process from source
// data to display.
//
// By modelling the render pipeline as functional components we can combine
// them in many ways to define (e.g.):
//     - Empty pipelines.
//     - Degenerate pipelines.
//     - Simplified geometry.
//     - Simplified interpolators.
//
// We also provide a way to visualize these pipelines as a means to inspect how
// the code is producing output and also to facilitate interrogation and debug.
// Further, we may use this decomposition of the rendering pipeline to enable
// parallelism in the future.
//
// A well defined pipeline must, at a minimum, consist of a pixel output.

using System;
using System.Collections.Generic;
using System.Linq;
using RenderToy.SceneGraph;
using RenderToy.SceneGraph.Meshes;
using RenderToy.SceneGraph.Primitives;

namespace RenderToy.PipelineModel
{
    /// <summary>
    /// Representation of a single colored pixel.
    /// 
    /// Coordinates are limited to 16-bit precision (to 65535 pixels).
    /// Colors are represented as 32bpp BGRA DWORDs.
    /// </summary>
    public struct PixelBgra32
    {
        public ushort X, Y;
        public uint Color;
    }
    /// <summary>
    /// Pipeline functions for assembling render pipelines.
    /// 
    /// These functions can be chained arbitrarily to assemble complete or
    /// partial pipelines. Components can be omitted to test cases such as
    /// (e.g.) a missing clipper unit or different transform behaviors.
    /// </summary>
    class Pipeline
    {
        /// <summary>
        /// Clip a list of vertices.
        /// 
        /// Internally this simply omits points for which w≤0.
        /// </summary>
        /// <param name="vertices">The vertex source to be clipped.</param>
        /// <returns>A filtered list of vertices for which no vertex has w≤0.</returns>
        public static IEnumerable<Vector4D> ClipPoint(IEnumerable<Vector4D> vertices)
        {
            return vertices.Where(v => v.W > 0);
        }
        /// <summary>
        /// Clip a list of lines.
        /// </summary>
        /// <param name="lines">The line source to be clipped.</param>
        /// <returns>A stream of line segments completely clipped by and contained in clip space.</returns>
        public static IEnumerable<Vector4D> ClipLine(IEnumerable<Vector4D> lines)
        {
            var iter = lines.GetEnumerator();
            while (iter.MoveNext())
            {
                var P0 = iter.Current;
                if (!iter.MoveNext())
                {
                    yield break;
                }
                var P1 = iter.Current;
                if (ClipHelp.ClipLine3D(ref P0, ref P1))
                {
                    yield return P0;
                    yield return P1;
                }
            }
        }
        /// <summary>
        /// Clip a list of triangles.
        /// </summary>
        /// <param name="triangles">The triangle source to be clipped.</param>
        /// <returns>A stream of triangles completely clipped by and contained in clip space.</returns>
        public static IEnumerable<Vector4D> ClipTriangle(IEnumerable<Vector4D> triangles)
        {
            return ClipHelp.ClipTriangle4D(triangles);
        }
        /// <summary>
        /// Perform a homogeneous divide on a 4D vector (for readability only).
        /// </summary>
        /// <param name="vertex">The input vector.</param>
        /// <returns>A vector of the form [x/w,y/y,z/w,1].</returns>
        public static Vector4D HomogeneousDivide(Vector4D vertex)
        {
            return MathHelp.Multiply(1.0 / vertex.W, vertex);
        }
        /// <summary>
        /// Perform a homogeneous divide on a vertex stream.
        /// 
        /// [x,y,z,w] will be transformed to [x/w,y/w,z/w,1] for all vertices.
        /// </summary>
        /// <param name="vertices">The vertex source to be transformed</param>
        /// <returns>A stream of post-homogeneous-divide vertices.</returns>
        public static IEnumerable<Vector4D> HomogeneousDivide(IEnumerable<Vector4D> vertices)
        {
            return vertices.Select(v => HomogeneousDivide(v));
        }
        /// <summary>
        /// Transform a stream of vertices by an arbitrary 4D matrix.
        /// </summary>
        /// <param name="vertices">The vertex source to be transformed.</param>
        /// <param name="transform">The transform to apply to each vertex.</param>
        /// <returns>A stream of transformed vertices.</returns>
        public static IEnumerable<Vector4D> Transform(IEnumerable<Vector4D> vertices, Matrix3D transform)
        {
            return vertices.Select(v => MathHelp.Transform(transform, v));
        }
        /// <summary>
        /// Transform a homogeneous vertex into screen space with the supplied dimensions.
        /// </summary>
        /// <param name="vertex">The vertex to be transformed.</param>
        /// <param name="width">The width of the screen area in pixels.</param>
        /// <param name="height">The height of the screen area in pixels.</param>
        /// <returns>The vertex transformed into screen space.</returns>
        public static Vector4D TransformToScreen(Vector4D vertex, double width, double height)
        {
            return new Vector4D((vertex.X + 1) * width / 2, (1 - vertex.Y) * height / 2, vertex.Z, vertex.W);
        }
        /// <summary>
        /// Transform a list of vertices into screen space.
        /// </summary>
        /// <param name="vertices">The vertices to be transformed.</param>
        /// <param name="width">The width of the screen area in pixels.</param>
        /// <param name="height">The height of the screen area in pixels.</param>
        /// <returns>A stream of screen-space transformed vertices.</returns>
        public static IEnumerable<Vector4D> TransformToScreen(IEnumerable<Vector4D> vertices, double width, double height)
        {
            return vertices.Select(v => TransformToScreen(v, width, height));
        }
        /// <summary>
        /// Rasterize a single screen-space points into colored pixels.
        /// </summary>
        /// <param name="point">The point to be rasterized.</param>
        /// <returns>A stream of pixels to write to the framebuffer.</returns>
        public static IEnumerable<PixelBgra32> RasterizePoint(Vector4D point)
        {
           yield return new PixelBgra32 { X = (ushort)point.X, Y = (ushort)point.Y, Color = 0xFF808080 };
        }
        /// <summary>
        /// Rasterize a stream of screen-space points and emit pixels for all.
        /// </summary>
        /// <param name="vertices">The vertex source to be transformed.</param>
        /// <returns>A stream of screen transformed pixels.</returns>
        public static IEnumerable<PixelBgra32> RasterizePoint(IEnumerable<Vector4D> vertices)
        {
            return vertices.SelectMany(p => RasterizePoint(p));
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
        public static IEnumerable<PixelBgra32> RasterizeLine(Vector4D P0, Vector4D P1)
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
                    yield return new PixelBgra32 { X = (ushort)(p0.X + x), Y = (ushort)(p0.Y + (p1.Y - p0.Y) * x / (p1.X - p0.X)), Color = 0xFF808080 };
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
                    yield return new PixelBgra32 { X = (ushort)(p0.X + (p1.X - p0.X) * y / (p1.Y - p0.Y)), Y = (ushort)(p0.Y + y), Color = 0xFF808080 };
                }
            }
        }
        /// <summary>
        /// Rasterize a stream of lines into pixels.
        /// </summary>
        /// <param name="lines">The lines to be rasterized.</param>
        /// <returns>A stream of pixels to be written to the framebuffer.</returns>
        public static IEnumerable<PixelBgra32> RasterizeLine(IEnumerable<Vector4D> lines)
        {
            IEnumerator<Vector4D> iter = lines.GetEnumerator();
            while (iter.MoveNext())
            {
                var P0 = iter.Current;
                if (!iter.MoveNext()) yield break;
                var P1 = iter.Current;
                foreach (var pixel in RasterizeLine(P0, P1))
                {
                    yield return pixel;
                }
            }
        }
        /// <summary>
        /// Rasterize a triangle in screen-space and emit pixels.
        /// </summary>
        /// <param name="triangle">The triangle to be rasterized.</param>
        /// <returns>A stream of pixels to be written to the framebuffer.</returns>
        public static IEnumerable<PixelBgra32> RasterizeTriangle(Vector4D P0, Vector4D P1, Vector4D P2)
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
                    yield return new PixelBgra32 { X = (ushort)x, Y = (ushort)y, Color = 0xFF808080 };
                }
            }
        }
        /// <summary>
        /// Rasterize a stream of triangles into pixels.
        /// </summary>
        /// <param name="triangles">The triangles to be rasterized.</param>
        /// <returns>A stream of pixels to be written to the framebuffer.</returns>
        public static IEnumerable<PixelBgra32> RasterizeTriangle(IEnumerable<Vector4D> triangles)
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
                foreach (var pixel in RasterizeTriangle(P0, P1, P2))
                {
                    yield return pixel;
                }
            }
        }
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
        /// Convert an input scene into a point list.
        /// </summary>
        /// <param name="scene">The source scene.</param>
        /// <returns>A stream of colorer vertices.</returns>
        public static IEnumerable<Vector3D> SceneToPoints(Scene scene)
        {
            foreach (var transformedobject in TransformedObject.Enumerate(scene))
            {
                Matrix3D model_mvp = transformedobject.Transform;
                IParametricUV uv = transformedobject.Node.Primitive as IParametricUV;
                if (uv != null)
                {
                    int USEGMENTS = 20;
                    int VSEGMENTS = 20;
                    // Simply move some number of steps across u and v and draw the points in space.
                    for (int u = 0; u <= USEGMENTS; ++u)
                    {
                        for (int v = 0; v <= VSEGMENTS; ++v)
                        {
                            // Determine the point and draw it; easy.
                            yield return MathHelp.TransformPoint(model_mvp, uv.GetPointUV((double)u / USEGMENTS, (double)v / VSEGMENTS));
                        }
                    }
                    continue;
                }
                IParametricUVW uvw = transformedobject.Node.Primitive as IParametricUVW;
                if (uvw != null)
                {
                    int USEGMENTS = 20;
                    int VSEGMENTS = 20;
                    int WSEGMENTS = 20;
                    // Simply move some number of steps across u and v and draw the points in space.
                    for (int u = 0; u <= USEGMENTS; ++u)
                    {
                        for (int v = 0; v <= VSEGMENTS; ++v)
                        {
                            for (int w = 0; w <= VSEGMENTS; ++w)
                            {
                                // Determine the point and draw it; easy.
                                yield return MathHelp.TransformPoint(model_mvp, uvw.GetPointUVW((double)u / USEGMENTS, (double)v / VSEGMENTS, (double)w / WSEGMENTS));
                            }
                        }
                    }
                }
                Mesh mesh = transformedobject.Node.Primitive as Mesh;
                if (mesh != null)
                {
                    foreach (var p in mesh.Vertices)
                    {
                        yield return p;
                    }
                    continue;
                }
            }
        }
        /// <summary>
        /// Convert an input scene into a wireframe line list.
        /// </summary>
        /// <param name="scene">The source scene.</param>
        /// <returns>A stream of colored line segments.</returns>
        public static IEnumerable<Vector3D> SceneToLines(Scene scene)
        {
            foreach (var transformedobject in TransformedObject.Enumerate(scene))
            {
                Matrix3D model_mvp = transformedobject.Transform;
                uint color = DrawHelp.ColorToUInt32(transformedobject.Node.WireColor);
                IParametricUV uv = transformedobject.Node.Primitive as IParametricUV;
                if (uv != null)
                {
                    int USEGMENTS = 5;
                    int VSEGMENTS = 5;
                    for (int u = 0; u <= USEGMENTS; ++u)
                    {
                        for (int v = 0; v < VSEGMENTS; ++v)
                        {
                            // Draw U Lines.
                            {
                                yield return MathHelp.TransformPoint(model_mvp, uv.GetPointUV((u + 0.0) / USEGMENTS, (v + 0.0) / VSEGMENTS));
                                yield return MathHelp.TransformPoint(model_mvp, uv.GetPointUV((u + 0.0) / USEGMENTS, (v + 1.0) / VSEGMENTS));
                            }
                            // Draw V Lines.
                            {
                                yield return MathHelp.TransformPoint(model_mvp, uv.GetPointUV((v + 0.0) / VSEGMENTS, (u + 0.0) / USEGMENTS));
                                yield return MathHelp.TransformPoint(model_mvp, uv.GetPointUV((v + 1.0) / VSEGMENTS, (u + 0.0) / USEGMENTS));
                            }
                        }
                    }
                    continue;
                }
                IParametricUVW uvw = transformedobject.Node.Primitive as IParametricUVW;
                if (uvw != null)
                {
                    int USEGMENTS = 20;
                    int VSEGMENTS = 20;
                    int WSEGMENTS = 20;
                    // Simply move some number of steps across u and v and draw the points in space.
                    for (int u = 0; u <= USEGMENTS; ++u)
                    {
                        for (int v = 0; v <= VSEGMENTS; ++v)
                        {
                            for (int w = 0; w <= WSEGMENTS; ++w)
                            {
                                // Determine the point and draw it; easy.
                                //Vector3D p = MathHelp.TransformPoint(model_mvp, uvw.GetPointUVW((double)u / USEGMENTS, (double)v / VSEGMENTS, (double)w / WSEGMENTS));
                                //yield return new Vertex<Vector3D> { Position = p, Color = color };
                            }
                        }
                    }
                }
                Mesh mesh = transformedobject.Node.Primitive as Mesh;
                if (mesh != null)
                {
                    foreach (var p in mesh.Vertices)
                    {
                        //yield return new Vertex<Vector3D> { Position = p, Color = color };
                    }
                    continue;
                }
            }
        }
        public static IEnumerable<Vector3D> SceneToTriangles(Scene scene)
        {
            foreach (var transformedobject in TransformedObject.Enumerate(scene))
            {
                Matrix3D model_mvp = transformedobject.Transform;
                uint color = DrawHelp.ColorToUInt32(transformedobject.Node.WireColor);
                IParametricUV uv = transformedobject.Node.Primitive as IParametricUV;
                if (uv != null)
                {
                    int USEGMENTS = 5;
                    int VSEGMENTS = 5;
                    for (int u = 0; u < USEGMENTS; ++u)
                    {
                        for (int v = 0; v < VSEGMENTS; ++v)
                        {
                            Vector3D p300 = MathHelp.TransformPoint(model_mvp, uv.GetPointUV((u + 0.0) / USEGMENTS, (v + 0.0) / VSEGMENTS));
                            Vector3D p310 = MathHelp.TransformPoint(model_mvp, uv.GetPointUV((u + 1.0) / USEGMENTS, (v + 0.0) / VSEGMENTS));
                            Vector3D p301 = MathHelp.TransformPoint(model_mvp, uv.GetPointUV((u + 0.0) / USEGMENTS, (v + 1.0) / VSEGMENTS));
                            Vector3D p311 = MathHelp.TransformPoint(model_mvp, uv.GetPointUV((u + 1.0) / USEGMENTS, (v + 1.0) / VSEGMENTS));
                            yield return p300; yield return p310; yield return p311;
                            yield return p311; yield return p301; yield return p300;
                        }
                    }
                    continue;
                }
            }
        }
        /// <summary>
        /// Up-cast a stream of 3D vectors to a stream of 4D vectors with w=1.
        /// </summary>
        /// <param name="vertices">A stream of 3D vectors.</param>
        /// <returns>A stream of 4D vectors with w=1.</returns>
        public static Vector4D Vector3ToVector4(Vector3D vertex)
        {
            return new Vector4D { X = vertex.X, Y = vertex.Y, Z = vertex.Z, W = 1 };
        }
        /// <summary>
        /// Cast a sequence of Vector3 points to their homogeneous representation [x,y,z,1].
        /// </summary>
        /// <param name="vertices">The vertices to cast.</param>
        /// <returns>A stream of homogeneous vertices expanded as [x,y,z,1].</returns>
        public static IEnumerable<Vector4D> Vector3ToVector4(IEnumerable<Vector3D> vertices)
        {
            return vertices.Select(v => Vector3ToVector4(v));
        }
    }
}