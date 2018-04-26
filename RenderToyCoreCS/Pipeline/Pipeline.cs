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
    /// Representation of a colored vertex with a given vector representation.
    /// </summary>
    /// <typeparam name="VERTEX">The underlying vector storage type.</typeparam>
    public struct Vertex<VERTEX>
    {
        public VERTEX Position;
        public uint Color;
    }
    /// <summary>
    /// Representation of a colored line with a given vector representation.
    /// </summary>
    /// <typeparam name="VERTEX">The underlying vector storage type.</typeparam>
    public struct Line<VERTEX>
    {
        public VERTEX P0, P1;
        public uint Color;
    }
    /// <summary>
    /// Representation of a colored triangle with a given vector representation.
    /// </summary>
    /// <typeparam name="VERTEX">The underlying vector storage type.</typeparam>
    public struct Triangle<VERTEX>
    {
        public VERTEX P0, P1, P2;
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
        public static IEnumerable<Vertex<Vector4D>> Clip(IEnumerable<Vertex<Vector4D>> vertices)
        {
            return vertices.Where(v => v.Position.W > 0);
        }
        /// <summary>
        /// Clip a list of lines.
        /// </summary>
        /// <param name="lines">The line source to be clipped.</param>
        /// <returns>A stream of line segments completely clipped by and contained in clip space.</returns>
        public static IEnumerable<Line<Vector4D>> Clip(IEnumerable<Line<Vector4D>> lines)
        {
            foreach (var line in lines)
            {
                Vector4D p0 = line.P0;
                Vector4D p1 = line.P1;
                if (ClipHelp.ClipLine3D(ref p0, ref p1))
                {
                    yield return new Line<Vector4D> { P0 = p0, P1 = p1, Color = line.Color };
                }
            }
        }
        /// <summary>
        /// Clip a list of triangles.
        /// </summary>
        /// <param name="triangles">The triangle source to be clipped.</param>
        /// <returns>A stream of triangles completely clipped by and contained in clip space.</returns>
        public static IEnumerable<Triangle<Vector4D>> Clip(IEnumerable<Triangle<Vector4D>> triangles)
        {
            foreach (var triangle in triangles)
            {
                Triangle4D triangle4 = new Triangle4D(triangle.P0, triangle.P1, triangle.P2);
                foreach (var clipped in ClipHelp.ClipTriangle4D(triangle4))
                {
                    yield return new Triangle<Vector4D> { P0 = clipped.P0, P1 = clipped.P1, P2 = clipped.P2, Color = triangle.Color };
                }
            }
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
        public static IEnumerable<Vertex<Vector4D>> HomogeneousDivide(IEnumerable<Vertex<Vector4D>> vertices)
        {
            return vertices.Select(v => new Vertex<Vector4D> { Position = HomogeneousDivide(v.Position), Color = v.Color });
        }
        /// <summary>
        /// Perform a homogeneous divide on a line list.
        /// </summary>
        /// <param name="lines">The stream of lines to be clipped.</param>
        /// <returns>A stream of clipped lines guaranteed to be contained completely in clip space.</returns>
        public static IEnumerable<Line<Vector4D>> HomogeneousDivide(IEnumerable<Line<Vector4D>> lines)
        {
            return lines.Select(v => new Line<Vector4D> { P0 = HomogeneousDivide(v.P0), P1 = HomogeneousDivide(v.P1), Color = v.Color });
        }
        /// <summary>
        /// Perform a homogeneous divide on a triangle list.
        /// </summary>
        /// <param name="lines">The stream of triangles to be clipped.</param>
        /// <returns>A stream of clipped triangles guaranteed to be contained completely in clip space.</returns>
        public static IEnumerable<Triangle<Vector4D>> HomogeneousDivide(IEnumerable<Triangle<Vector4D>> lines)
        {
            return lines.Select(v => new Triangle<Vector4D> { P0 = HomogeneousDivide(v.P0), P1 = HomogeneousDivide(v.P1), P2 = HomogeneousDivide(v.P2), Color = v.Color });
        }
        /// <summary>
        /// Transform a stream of vertices by an arbitrary 4D matrix.
        /// </summary>
        /// <param name="vertices">The vertex source to be transformed.</param>
        /// <param name="transform">The transform to apply to each vertex.</param>
        /// <returns>A stream of transformed vertices.</returns>
        public static IEnumerable<Vertex<Vector4D>> Transform(IEnumerable<Vertex<Vector4D>> vertices, Matrix3D transform)
        {
            return vertices.Select(v => new Vertex<Vector4D> { Position = MathHelp.Transform(transform, v.Position), Color = v.Color });
        }
        /// <summary>
        /// Transform a stream of line segments by an arbitrary 4D matrix.
        /// </summary>
        /// <param name="lines">The lines to be transformed.</param>
        /// <param name="transform">The transformation to be applied.</param>
        /// <returns>A stream of line segments transformed by the supplied matrix.</returns>
        public static IEnumerable<Line<Vector4D>> Transform(IEnumerable<Line<Vector4D>> lines, Matrix3D transform)
        {
            return lines.Select(v => new Line<Vector4D> { P0 = MathHelp.Transform(transform, v.P0), P1 = MathHelp.Transform(transform, v.P1), Color = v.Color });
        }
        /// <summary>
        /// Transform a stream of triangles by an arbitrary 4D matrix.
        /// </summary>
        /// <param name="triangles">The triangles to be transformed.</param>
        /// <param name="transform">The transformation to be applied.</param>
        /// <returns>A stream of triangles transformed by the supplied matrix.</returns>
        public static IEnumerable<Triangle<Vector4D>> Transform(IEnumerable<Triangle<Vector4D>> triangles, Matrix3D transform)
        {
            return triangles.Select(v => new Triangle<Vector4D> { P0 = MathHelp.Transform(transform, v.P0), P1 = MathHelp.Transform(transform, v.P1), P2 = MathHelp.Transform(transform, v.P2), Color = v.Color });
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
        public static IEnumerable<Vertex<Vector4D>> TransformToScreen(IEnumerable<Vertex<Vector4D>> vertices, double width, double height)
        {
            return vertices.Select(v => new Vertex<Vector4D> { Position = TransformToScreen(v.Position, width, height), Color = v.Color });
        }
        /// <summary>
        /// Transform a list of lines into screen space.
        /// </summary>
        /// <param name="lines">The lines to be transformed.</param>
        /// <param name="width">The width of the screen area in pixels.</param>
        /// <param name="height">The height of the screen area in pixels.</param>
        /// <returns>A stream of screen-space transformed lines.</returns>
        public static IEnumerable<Line<Vector4D>> TransformToScreen(IEnumerable<Line<Vector4D>> lines, double width, double height)
        {
            return lines.Select(v => new Line<Vector4D> { P0 = TransformToScreen(v.P0, width, height), P1 = TransformToScreen(v.P1, width, height), Color = v.Color });
        }
        /// <summary>
        /// Transform a list of triangles into screen space.
        /// </summary>
        /// <param name="triangles">The triangles to be transformed.</param>
        /// <param name="width">The width of the screen area in pixels.</param>
        /// <param name="height">The height of the screen area in pixels.</param>
        /// <returns>A stream of screen-space transformed triangles.</returns>
        public static IEnumerable<Triangle<Vector4D>> TransformToScreen(IEnumerable<Triangle<Vector4D>> triangles, double width, double height)
        {
            return triangles.Select(v => new Triangle<Vector4D> { P0 = TransformToScreen(v.P0, width, height), P1 = TransformToScreen(v.P1, width, height), P2 = TransformToScreen(v.P2, width, height), Color = v.Color });
        }
        /// <summary>
        /// Rasterize a single screen-space points into colored pixels.
        /// </summary>
        /// <param name="point">The point to be rasterized.</param>
        /// <returns>A stream of pixels to write to the framebuffer.</returns>
        public static IEnumerable<PixelBgra32> Rasterize(Vertex<Vector4D> point)
        {
           yield return new PixelBgra32 { X = (ushort)point.Position.X, Y = (ushort)point.Position.Y, Color = point.Color };
        }
        /// <summary>
        /// Rasterize a stream of screen-space points and emit pixels for all.
        /// </summary>
        /// <param name="vertices">The vertex source to be transformed.</param>
        /// <returns>A stream of screen transformed pixels.</returns>
        public static IEnumerable<PixelBgra32> Rasterize(IEnumerable<Vertex<Vector4D>> vertices)
        {
            return vertices.SelectMany(p => Rasterize(p));
        }
        /// <summary>
        /// Rasterize a stream of clip-space points and emit pixels.
        /// </summary>
        /// <param name="vertices">The vertex source to be transformed.</param>
        /// <param name="width">The width of the screen in pixels.</param>
        /// <param name="height">The height of the screen in pixels.</param>
        /// <returns>A stream of pixels to be written to the framebuffer.</returns>
        public static IEnumerable<PixelBgra32> Rasterize(IEnumerable<Vertex<Vector4D>> vertices, ushort width, ushort height)
        {
            return vertices.Select(v => new PixelBgra32 { X = (ushort)((v.Position.X + 1) * width / 2), Y = (ushort)((1 - v.Position.Y) * height / 2), Color = v.Color });
        }
        /// <summary>
        /// Rasterize a line in sceen space and emit pixels.
        /// </summary>
        /// <param name="line"></param>
        /// <returns>A stream of pixels to be written to the framebuffer.</returns>
        public static IEnumerable<PixelBgra32> Rasterize(Line<Vector4D> line)
        {
            if (Math.Abs(line.P1.X - line.P0.X) > Math.Abs(line.P1.Y - line.P0.Y))
            {
                // X spanning line; this line is longer in the X axis.
                // Scan in the X direction plotting Y points.
                var p0 = new Vector4D();
                var p1 = new Vector4D();
                if (line.P0.X < line.P1.X)
                {
                    p0 = line.P0;
                    p1 = line.P1;
                }
                else
                {
                    p0 = line.P1;
                    p1 = line.P0;
                }
                for (int x = 0; x <= p1.X - p0.X; ++x)
                {
                    yield return new PixelBgra32 { X = (ushort)(p0.X + x), Y = (ushort)(p0.Y + (p1.Y - p0.Y) * x / (p1.X - p0.X)), Color = line.Color };
                }
            }
            else
            {
                // Y spanning line; this line is longer in the Y axis.
                // Scan in the Y direction plotting X points.
                var p0 = new Vector4D();
                var p1 = new Vector4D();
                if (line.P0.Y < line.P1.Y)
                {
                    p0 = line.P0;
                    p1 = line.P1;
                }
                else
                {
                    p0 = line.P1;
                    p1 = line.P0;
                }
                for (int y = 0; y <= p1.Y - p0.Y; ++y)
                {
                    yield return new PixelBgra32 { X = (ushort)(p0.X + (p1.X - p0.X) * y / (p1.Y - p0.Y)), Y = (ushort)(p0.Y + y), Color = line.Color };
                }
            }
        }
        /// <summary>
        /// Rasterize a stream of lines into pixels.
        /// </summary>
        /// <param name="lines">The lines to be rasterized.</param>
        /// <returns>A stream of pixels to be written to the framebuffer.</returns>
        public static IEnumerable<PixelBgra32> Rasterize(IEnumerable<Line<Vector4D>> lines)
        {
            return lines.SelectMany(l => Rasterize(l));
        }
        /// <summary>
        /// Rasterize a triangle in screen-space and emit pixels.
        /// </summary>
        /// <param name="triangle">The triangle to be rasterized.</param>
        /// <returns>A stream of pixels to be written to the framebuffer.</returns>
        public static IEnumerable<PixelBgra32> Rasterize(Triangle<Vector4D> triangle)
        {
            // Calculate edge lines.
            var edges = new[]
            {
                new { Org = triangle.P0, Dir = triangle.P1 - triangle.P0 },
                new { Org = triangle.P1, Dir = triangle.P2 - triangle.P1 },
                new { Org = triangle.P2, Dir = triangle.P0 - triangle.P2 }
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
                    double alpha = MathHelp.Dot(new Vector2D(xline - triangle.P0.X, yline - triangle.P0.Y), new Vector2D(triangle.P1.X - triangle.P0.X, triangle.P1.Y - triangle.P0.Y));
                    double beta = MathHelp.Dot(new Vector2D(xline - triangle.P0.X, yline - triangle.P0.Y), new Vector2D(triangle.P2.X - triangle.P0.X, triangle.P2.Y - triangle.P0.Y));
                    yield return new PixelBgra32 { X = (ushort)x, Y = (ushort)y, Color = triangle.Color };
                }
            }
        }
        /// <summary>
        /// Rasterize a stream of triangles into pixels.
        /// </summary>
        /// <param name="triangles">The triangles to be rasterized.</param>
        /// <returns>A stream of pixels to be written to the framebuffer.</returns>
        public static IEnumerable<PixelBgra32> Rasterize(IEnumerable<Triangle<Vector4D>> triangles)
        {
            return triangles.SelectMany(t => Rasterize(t));
        }
        /// <summary>
        /// Convert an input scene into a point list.
        /// </summary>
        /// <param name="scene">The source scene.</param>
        /// <returns>A stream of colorer vertices.</returns>
        public static IEnumerable<Vertex<Vector3D>> SceneToVertices(Scene scene)
        {
            foreach (var transformedobject in TransformedObject.Enumerate(scene))
            {
                Matrix3D model_mvp = transformedobject.Transform;
                uint color = DrawHelp.ColorToUInt32(transformedobject.Node.WireColor);
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
                            Vector3D p = MathHelp.TransformPoint(model_mvp, uv.GetPointUV((double)u / USEGMENTS, (double)v / VSEGMENTS));
                            yield return new Vertex<Vector3D> { Position = p, Color = color };
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
                                Vector3D p = MathHelp.TransformPoint(model_mvp, uvw.GetPointUVW((double)u / USEGMENTS, (double)v / VSEGMENTS, (double)w / WSEGMENTS));
                                yield return new Vertex<Vector3D> { Position = p, Color = color };
                            }
                        }
                    }
                }
                Mesh mesh = transformedobject.Node.Primitive as Mesh;
                if (mesh != null)
                {
                    foreach (var p in mesh.Vertices)
                    {
                        yield return new Vertex<Vector3D> { Position = p, Color = color };
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
        public static IEnumerable<Line<Vector3D>> SceneToLines(Scene scene)
        {
            foreach (var transformedobject in TransformedObject.Enumerate(scene))
            {
                Matrix3D model_mvp = transformedobject.Transform;
                uint color = DrawHelp.ColorToUInt32(transformedobject.Node.WireColor);
                IParametricUV uv = transformedobject.Node.Primitive as IParametricUV;
                if (uv != null)
                {
                    int USEGMENTS = 10;
                    int VSEGMENTS = 20;
                    for (int u = 0; u <= USEGMENTS; ++u)
                    {
                        for (int l = 0; l < VSEGMENTS; ++l)
                        {
                            // Draw U Lines.
                            {
                                Vector3D p3u1 = MathHelp.TransformPoint(model_mvp, uv.GetPointUV((u + 0.0) / USEGMENTS, (l + 0.0) / VSEGMENTS));
                                Vector3D p3u2 = MathHelp.TransformPoint(model_mvp, uv.GetPointUV((u + 0.0) / USEGMENTS, (l + 1.0) / VSEGMENTS));
                                yield return new Line<Vector3D> { P0 = p3u1, P1 = p3u2, Color = color };
                            }
                            // Draw V Lines.
                            {
                                Vector3D p3u1 = MathHelp.TransformPoint(model_mvp, uv.GetPointUV((l + 0.0) / VSEGMENTS, (u + 0.0) / USEGMENTS));
                                Vector3D p3u2 = MathHelp.TransformPoint(model_mvp, uv.GetPointUV((l + 1.0) / VSEGMENTS, (u + 0.0) / USEGMENTS));
                                yield return new Line<Vector3D> { P0 = p3u1, P1 = p3u2, Color = color };
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
        public static IEnumerable<Triangle<Vector3D>> SceneToTriangles(Scene scene)
        {
            foreach (var transformedobject in TransformedObject.Enumerate(scene))
            {
                Matrix3D model_mvp = transformedobject.Transform;
                uint color = DrawHelp.ColorToUInt32(transformedobject.Node.WireColor);
                IParametricUV uv = transformedobject.Node.Primitive as IParametricUV;
                if (uv != null)
                {
                    int USEGMENTS = 10;
                    int VSEGMENTS = 20;
                    for (int u = 0; u <= USEGMENTS; ++u)
                    {
                        for (int l = 0; l < VSEGMENTS; ++l)
                        {
                            Vector3D p300 = MathHelp.TransformPoint(model_mvp, uv.GetPointUV((u + 0.0) / USEGMENTS, (l + 0.0) / VSEGMENTS));
                            Vector3D p310 = MathHelp.TransformPoint(model_mvp, uv.GetPointUV((u + 1.0) / USEGMENTS, (l + 0.0) / VSEGMENTS));
                            Vector3D p301 = MathHelp.TransformPoint(model_mvp, uv.GetPointUV((u + 0.0) / USEGMENTS, (l + 1.0) / VSEGMENTS));
                            Vector3D p311 = MathHelp.TransformPoint(model_mvp, uv.GetPointUV((u + 1.0) / USEGMENTS, (l + 1.0) / VSEGMENTS));
                            yield return new Triangle<Vector3D> { P0 = p300, P1 = p310, P2 = p311, Color = color };
                            yield return new Triangle<Vector3D> { P0 = p311, P1 = p301, P2 = p300, Color = color };
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
        public static IEnumerable<Vertex<Vector4D>> Vector3ToVector4(IEnumerable<Vertex<Vector3D>> vertices)
        {
            return vertices.Select(v => new Vertex<Vector4D> { Position = Vector3ToVector4(v.Position), Color = v.Color });
        }
        /// <summary>
        /// Cast a sequence of Vector3 lines to their homogeneous representation [x,y,z,1].
        /// </summary>
        /// <param name="lines">The lines to be cast.</param>
        /// <returns>A stream of homoegeneous lines expanded as [x,y,z,1].</returns>
        public static IEnumerable<Line<Vector4D>> Vector3ToVector4(IEnumerable<Line<Vector3D>> lines)
        {
            return lines.Select(v => new Line<Vector4D> { P0 = Vector3ToVector4(v.P0), P1 = Vector3ToVector4(v.P1), Color = v.Color });
        }
        /// <summary>
        /// Cast a sequence of Vector3 triangles to their homogeneous representation [x,y,z,1].
        /// </summary>
        /// <param name="triangles">The triangles to be cast.</param>
        /// <returns>A stream of homogeneous triangles expanded as [x,y,z,1].</returns>
        public static IEnumerable<Triangle<Vector4D>> Vector3ToVector4(IEnumerable<Triangle<Vector3D>> triangles)
        {
            return triangles.Select(v => new Triangle<Vector4D> { P0 = Vector3ToVector4(v.P0), P1 = Vector3ToVector4(v.P1), P2 = Vector3ToVector4(v.P2), Color = v.Color });
        }
        /// <summary>
        /// TEST FUNCTION: Generate a pixel for every pixel in a screen area.
        /// </summary>
        /// <param name="width">The width of the screen.</param>
        /// <param name="height">The height of the screen.</param>
        /// <returns>A stream of magenta pixels as big as the screen.</returns>
        public static IEnumerable<PixelBgra32> TestFillScreen(ushort width, ushort height)
        {
            for (ushort y = 0; y < height; ++y)
            {
                for (ushort x = 0; x < width; ++x)
                {
                    yield return new PixelBgra32 { X = x, Y = y, Color = 0xFF00FFFF };
                }
            }
        }
    }
}